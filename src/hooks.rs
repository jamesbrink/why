//! Shell hook integration for automatic error explanation.

use anyhow::{Context, Result};
use clap_complete::Shell;
use colored::Colorize;
use std::path::{Path, PathBuf};

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
            r#"__why_prompt_command() {
    local exit_code=$?
    if [[ $exit_code -ne 0 && $exit_code -ne 130 ]]; then
        why --exit-code "$exit_code" --last-command "$BASH_COMMAND" 2>/dev/null
    fi
}
PROMPT_COMMAND="__why_prompt_command${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
"#
        }
        Shell::Zsh => {
            r#"__why_precmd() {
    local exit_code=$?
    if [[ $exit_code -ne 0 && $exit_code -ne 130 ]]; then
        why --exit-code "$exit_code" --last-command "$__why_last_cmd" 2>/dev/null
    fi
}
__why_preexec() {
    __why_last_cmd="$1"
}
autoload -Uz add-zsh-hook
add-zsh-hook precmd __why_precmd
add-zsh-hook preexec __why_preexec
"#
        }
        Shell::Fish => {
            r#"function __why_postexec --on-event fish_postexec
    set -l exit_code $status
    if test $exit_code -ne 0 -a $exit_code -ne 130
        why --exit-code $exit_code --last-command "$argv" 2>/dev/null
    end
end
"#
        }
        Shell::PowerShell => {
            r#"function global:__why_prompt {
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -and $exitCode -ne 130) {
        why --exit-code $exitCode --last-command $MyInvocation.MyCommand 2>$null
    }
}
# Note: PowerShell hook integration is limited
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
