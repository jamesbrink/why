//! Shell hook integration for automatic error explanation.

use anyhow::{Context, Result};
use clap_complete::Shell;
use colored::Colorize;
use std::path::{Path, PathBuf};

/// State directory for hook settings
fn get_state_dir() -> Option<PathBuf> {
    // XDG_STATE_HOME or fallback to ~/.local/state
    if let Ok(state_home) = std::env::var("XDG_STATE_HOME") {
        Some(PathBuf::from(state_home).join("why"))
    } else {
        dirs::home_dir().map(|h| h.join(".local").join("state").join("why"))
    }
}

/// Path to the hook enabled state file
fn get_hook_state_path() -> Option<PathBuf> {
    get_state_dir().map(|d| d.join("hook_enabled"))
}

/// Check if hooks are enabled
pub fn is_hook_enabled() -> bool {
    // Environment variable takes precedence
    if std::env::var("WHY_HOOK_DISABLE")
        .map(|v| v == "1")
        .unwrap_or(false)
    {
        return false;
    }

    // Check state file (default: enabled)
    if let Some(state_path) = get_hook_state_path() {
        if state_path.exists() {
            return std::fs::read_to_string(state_path)
                .map(|s| s.trim() != "0")
                .unwrap_or(true);
        }
    }

    // Default: enabled
    true
}

/// Enable hook functionality
pub fn enable_hook() -> Result<()> {
    let state_dir =
        get_state_dir().ok_or_else(|| anyhow::anyhow!("Could not determine state directory"))?;

    std::fs::create_dir_all(&state_dir)
        .with_context(|| format!("Failed to create state directory: {}", state_dir.display()))?;

    let state_path = state_dir.join("hook_enabled");
    std::fs::write(&state_path, "1\n")
        .with_context(|| format!("Failed to write state file: {}", state_path.display()))?;

    println!("{} Shell hook enabled", "✓".green());
    println!();
    println!("  The why shell hook will now explain command failures.");
    println!(
        "  To disable temporarily: {}",
        "export WHY_HOOK_DISABLE=1".cyan()
    );
    println!();

    Ok(())
}

/// Disable hook functionality
pub fn disable_hook() -> Result<()> {
    let state_dir =
        get_state_dir().ok_or_else(|| anyhow::anyhow!("Could not determine state directory"))?;

    std::fs::create_dir_all(&state_dir)
        .with_context(|| format!("Failed to create state directory: {}", state_dir.display()))?;

    let state_path = state_dir.join("hook_enabled");
    std::fs::write(&state_path, "0\n")
        .with_context(|| format!("Failed to write state file: {}", state_path.display()))?;

    println!("{} Shell hook disabled", "✓".green());
    println!();
    println!("  The why shell hook will no longer explain command failures.");
    println!("  To re-enable: {}", "why --enable".cyan());
    println!();

    Ok(())
}

/// Print hook status
pub fn print_hook_status() {
    let enabled = is_hook_enabled();
    let env_disabled = std::env::var("WHY_HOOK_DISABLE")
        .map(|v| v == "1")
        .unwrap_or(false);

    println!("{}", "Shell Hook Status".bold());
    println!();

    // Enabled/disabled state
    if enabled {
        println!("  {} {}", "Status:".blue().bold(), "Enabled".green().bold());
    } else {
        println!("  {} {}", "Status:".blue().bold(), "Disabled".red().bold());
    }

    // Environment variable status
    if env_disabled {
        println!(
            "  {} {} (WHY_HOOK_DISABLE=1)",
            "Env override:".blue().bold(),
            "Disabled".red()
        );
    }

    println!();

    // Installation status per shell
    println!("{}", "Installation Status".bold());
    println!();

    for shell in [Shell::Bash, Shell::Zsh, Shell::Fish] {
        if let Some(config_path) = get_shell_config_path(shell) {
            let installed = hooks_already_installed(&config_path);
            let status = if installed {
                "Installed".green().to_string()
            } else {
                "Not installed".dimmed().to_string()
            };
            println!(
                "  {:<12} {} ({})",
                format!("{:?}:", shell),
                status,
                config_path.display()
            );
        }
    }

    println!();
    println!("{}", "Commands".bold());
    println!();
    println!("  {} - Enable hook", "why --enable".cyan());
    println!("  {} - Disable hook", "why --disable".cyan());
    println!(
        "  {} - Install hook for shell",
        "why --hook-install <shell>".cyan()
    );
    println!(
        "  {} - Uninstall hook from shell",
        "why --hook-uninstall <shell>".cyan()
    );
    println!();
}

/// Marker comment for detecting existing hook installations
pub const HOOK_MARKER_START: &str = "# >>> why shell hook >>>";
pub const HOOK_MARKER_END: &str = "# <<< why shell hook <<<";

/// Get the shell config file path for a given shell
pub fn get_shell_config_path(shell: Shell) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    match shell {
        Shell::Bash => Some(home.join(".bashrc")),
        Shell::Zsh => Some(home.join(".zshrc")),
        Shell::Fish => dirs::config_dir().map(|p| p.join("fish").join("conf.d").join("why.fish")),
        Shell::PowerShell => dirs::config_dir().map(|p| {
            if cfg!(windows) {
                dirs::document_dir()
                    .unwrap_or_else(|| p.clone())
                    .join("PowerShell")
                    .join("Microsoft.PowerShell_profile.ps1")
            } else {
                p.join("powershell")
                    .join("Microsoft.PowerShell_profile.ps1")
            }
        }),
        _ => None,
    }
}

/// Generate the hook script for a given shell
pub fn generate_hook_script(shell: Shell) -> &'static str {
    match shell {
        Shell::Bash => {
            r#"# why shell hook - captures command output for error explanation
__why_stderr_file="/tmp/why_stderr_$$"
__why_last_cmd=""

# Capture stderr while still displaying it
exec 2> >(tee -a "$__why_stderr_file" >&2)

__why_preexec() {
    __why_last_cmd="$1"
    # Clear stderr capture file before each command
    : > "$__why_stderr_file" 2>/dev/null
}

__why_prompt_command() {
    local exit_code=$?
    if [[ $exit_code -ne 0 && $exit_code -ne 130 && -n "$__why_last_cmd" ]]; then
        local output=""
        if [[ -f "$__why_stderr_file" && -s "$__why_stderr_file" ]]; then
            output=$(tail -100 "$__why_stderr_file" 2>/dev/null)
        fi
        if [[ -n "$output" ]]; then
            why --exit-code "$exit_code" --last-command "$__why_last_cmd" --last-output "$output" 2>/dev/null
        else
            why --exit-code "$exit_code" --last-command "$__why_last_cmd" 2>/dev/null
        fi
    fi
    __why_last_cmd=""
}

trap '__why_preexec "$BASH_COMMAND"' DEBUG
PROMPT_COMMAND="__why_prompt_command${PROMPT_COMMAND:+;$PROMPT_COMMAND}"

# Cleanup on exit
trap 'rm -f "$__why_stderr_file" 2>/dev/null' EXIT
"#
        }
        Shell::Zsh => {
            r#"# why shell hook - captures command output for error explanation
__why_stderr_file="/tmp/why_stderr_$$"
__why_last_cmd=""

# Capture stderr while still displaying it
exec 2> >(tee -a "$__why_stderr_file" >&2)

__why_preexec() {
    __why_last_cmd="$1"
    # Clear stderr capture file before each command
    : > "$__why_stderr_file" 2>/dev/null
}

__why_precmd() {
    local exit_code=$?
    if [[ $exit_code -ne 0 && $exit_code -ne 130 && -n "$__why_last_cmd" ]]; then
        local output=""
        if [[ -f "$__why_stderr_file" && -s "$__why_stderr_file" ]]; then
            output=$(tail -100 "$__why_stderr_file" 2>/dev/null)
        fi
        if [[ -n "$output" ]]; then
            why --exit-code "$exit_code" --last-command "$__why_last_cmd" --last-output "$output" 2>/dev/null
        else
            why --exit-code "$exit_code" --last-command "$__why_last_cmd" 2>/dev/null
        fi
    fi
    __why_last_cmd=""
}

autoload -Uz add-zsh-hook
add-zsh-hook preexec __why_preexec
add-zsh-hook precmd __why_precmd

# Cleanup on exit
trap 'rm -f "$__why_stderr_file" 2>/dev/null' EXIT
"#
        }
        Shell::Fish => {
            r#"# why shell hook - captures command output for error explanation
set -g __why_stderr_file "/tmp/why_stderr_"(echo %self)

function __why_preexec --on-event fish_preexec
    # Clear stderr capture file before each command
    echo -n > $__why_stderr_file 2>/dev/null
end

function __why_postexec --on-event fish_postexec
    set -l exit_code $status
    if test $exit_code -ne 0 -a $exit_code -ne 130
        set -l output ""
        if test -f $__why_stderr_file -a -s $__why_stderr_file
            set output (tail -100 $__why_stderr_file 2>/dev/null | string collect)
        end
        if test -n "$output"
            why --exit-code $exit_code --last-command "$argv" --last-output "$output" 2>/dev/null
        else
            why --exit-code $exit_code --last-command "$argv" 2>/dev/null
        end
    end
end

# Cleanup on exit
function __why_cleanup --on-event fish_exit
    rm -f $__why_stderr_file 2>/dev/null
end
"#
        }
        Shell::PowerShell => {
            r#"# why shell hook (PowerShell integration is limited)
function global:__why_prompt {
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -and $exitCode -ne 130) {
        why --exit-code $exitCode --last-command $MyInvocation.MyCommand 2>$null
    }
}
"#
        }
        _ => "",
    }
}

/// Generate the hook code wrapped with markers
pub fn generate_hook_with_markers(shell: Shell) -> String {
    let mut output = String::new();
    output.push_str(HOOK_MARKER_START);
    output.push('\n');
    output.push_str(generate_hook_script(shell));
    output.push_str(HOOK_MARKER_END);
    output.push('\n');
    output
}

/// Check if hooks are already installed in a config file
pub fn hooks_already_installed(config_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(config_path) {
        contents.contains(HOOK_MARKER_START)
    } else {
        false
    }
}

/// Install hook integration into shell config file
pub fn install_hook(shell: Shell) -> Result<()> {
    let config_path = get_shell_config_path(shell)
        .ok_or_else(|| anyhow::anyhow!("Could not determine config path for {}", shell))?;

    if hooks_already_installed(&config_path) {
        println!(
            "{} Why hooks are already installed in {}",
            "✓".green(),
            config_path.display()
        );
        return Ok(());
    }

    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    let mut content = std::fs::read_to_string(&config_path).unwrap_or_default();

    if !content.is_empty() && !content.ends_with('\n') {
        content.push('\n');
    }
    content.push('\n');

    content.push_str(&generate_hook_with_markers(shell));

    std::fs::write(&config_path, content)
        .with_context(|| format!("Failed to write to: {}", config_path.display()))?;

    println!(
        "{} {}",
        "✓".green(),
        "Why shell hooks installed successfully!".green().bold()
    );
    println!();
    println!(
        "  {} {}",
        "Config file:".blue().bold(),
        config_path.display()
    );
    println!();
    println!("  {} To activate, run:", "Next steps:".yellow().bold());
    match shell {
        Shell::Bash => println!("    source ~/.bashrc"),
        Shell::Zsh => println!("    source ~/.zshrc"),
        Shell::Fish => println!("    source {}", config_path.display()),
        Shell::PowerShell => println!("    . $PROFILE"),
        _ => {}
    }
    println!();
    println!("  Or open a new terminal session.");
    println!();

    Ok(())
}

/// Uninstall hook integration from shell config file
pub fn uninstall_hook(shell: Shell) -> Result<()> {
    let config_path = get_shell_config_path(shell)
        .ok_or_else(|| anyhow::anyhow!("Could not determine config path for {}", shell))?;

    if !config_path.exists() {
        println!(
            "{} Config file does not exist: {}",
            "?".yellow(),
            config_path.display()
        );
        return Ok(());
    }

    if !hooks_already_installed(&config_path) {
        println!(
            "{} Why hooks are not installed in {}",
            "?".yellow(),
            config_path.display()
        );
        return Ok(());
    }

    let content = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read: {}", config_path.display()))?;

    let mut new_content = String::new();
    let mut in_hook_block = false;

    for line in content.lines() {
        if line.trim() == HOOK_MARKER_START {
            in_hook_block = true;
            continue;
        }
        if line.trim() == HOOK_MARKER_END {
            in_hook_block = false;
            continue;
        }
        if !in_hook_block {
            new_content.push_str(line);
            new_content.push('\n');
        }
    }

    while new_content.ends_with("\n\n\n") {
        new_content.pop();
    }

    std::fs::write(&config_path, new_content)
        .with_context(|| format!("Failed to write to: {}", config_path.display()))?;

    println!(
        "{} {}",
        "✓".green(),
        "Why shell hooks uninstalled successfully!".green().bold()
    );
    println!();
    println!(
        "  {} {}",
        "Config file:".blue().bold(),
        config_path.display()
    );
    println!();
    println!("  Restart your terminal or source the config file to apply changes.");
    println!();

    Ok(())
}

/// Print the hook script to stdout
pub fn print_hook_script(shell: Shell) {
    print!("{}", generate_hook_with_markers(shell));
}
