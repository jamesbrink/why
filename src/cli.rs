//! Command-line interface definitions for the `why` tool.

use clap::{Parser, Subcommand};
use clap_complete::Shell;
use std::path::PathBuf;

use crate::model::ModelFamily;

/// Quick error explanation using local LLM
#[derive(Parser, Debug)]
#[command(
    name = "why",
    version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("WHY_GIT_SHA"), ")"),
    about,
    long_about = None
)]
#[command(
    after_help = "EXAMPLES:\n    why \"segmentation fault\"\n    cargo build 2>&1 | why\n    why --json \"null pointer exception\""
)]
pub struct Cli {
    /// Error message to explain
    #[arg(trailing_var_arg = true)]
    pub error: Vec<String>,

    /// Output as JSON
    #[arg(long, short = 'j')]
    pub json: bool,

    /// Show debug info (raw prompt and model response)
    #[arg(long, short = 'd')]
    pub debug: bool,

    /// Show performance stats
    #[arg(long)]
    pub stats: bool,

    /// Path to GGUF model file (overrides embedded model)
    #[arg(long, short = 'm', value_name = "PATH")]
    pub model: Option<PathBuf>,

    /// Model family for prompt template (auto-detected if not specified)
    #[arg(long, short = 't', value_enum, value_name = "FAMILY")]
    pub template: Option<ModelFamily>,

    /// List available model variants and exit
    #[arg(long)]
    pub list_models: bool,

    /// Generate shell completions
    #[arg(long, value_enum, value_name = "SHELL")]
    pub completions: Option<Shell>,

    /// Enable streaming output (display tokens as they're generated)
    #[arg(long, short = 's')]
    pub stream: bool,

    /// Generate shell hook integration script for the specified shell
    #[arg(long, value_enum, value_name = "SHELL")]
    pub hook: Option<Shell>,

    /// Exit code from the failed command (used by shell hooks)
    #[arg(long, value_name = "CODE")]
    pub exit_code: Option<i32>,

    /// The command that failed (used by shell hooks)
    #[arg(long, value_name = "CMD")]
    pub last_command: Option<String>,

    /// Enable source context injection (read source files referenced in stack traces)
    #[arg(long, short = 'c')]
    pub context: bool,

    /// Number of lines to show around error location (default: 5)
    #[arg(long, default_value = "5", value_name = "N")]
    pub context_lines: usize,

    /// Root path for resolving relative file paths in stack traces
    #[arg(long, value_name = "PATH")]
    pub context_root: Option<PathBuf>,

    /// Show parsed stack trace frames (requires stack trace in input)
    #[arg(long)]
    pub show_frames: bool,

    /// Run a command and explain its error output if it fails
    /// Example: why --capture npm run build
    #[arg(long)]
    pub capture: bool,

    /// Also capture and analyze stdout in capture mode (default: stderr only)
    #[arg(long)]
    pub capture_all: bool,

    /// Ask for confirmation before explaining errors (interactive mode)
    #[arg(long)]
    pub confirm: bool,

    /// Always explain without prompting (opposite of --confirm)
    #[arg(long)]
    pub auto: bool,

    /// Output default hook configuration to stdout
    #[arg(long)]
    pub hook_config: bool,

    /// Install hook integration into shell config file
    #[arg(long, value_enum, value_name = "SHELL")]
    pub hook_install: Option<Shell>,

    /// Uninstall hook integration from shell config file
    #[arg(long, value_enum, value_name = "SHELL")]
    pub hook_uninstall: Option<Shell>,

    // ========================================================================
    // Watch Mode (Feature 2)
    // ========================================================================
    /// Watch a file or command for errors and explain them as they appear
    /// Examples:
    ///   why --watch /var/log/app.log
    ///   why --watch "npm run dev"
    #[arg(long, short = 'w', value_name = "TARGET")]
    pub watch: Option<String>,

    /// Debounce time in milliseconds for watch mode (default: 500)
    #[arg(long, default_value = "500", value_name = "MS")]
    pub debounce: u64,

    /// Disable duplicate error suppression in watch mode
    #[arg(long)]
    pub no_dedup: bool,

    /// Custom regex pattern for error detection in watch mode
    #[arg(long, value_name = "REGEX")]
    pub pattern: Option<String>,

    /// Clear screen between errors in watch mode
    #[arg(long)]
    pub clear: bool,

    /// Quiet mode - only show explanations, no status messages
    #[arg(long, short = 'q')]
    pub quiet: bool,

    // ========================================================================
    // Daemon Mode (Feature 5)
    // ========================================================================
    /// Daemon management subcommand
    #[command(subcommand)]
    pub daemon: Option<DaemonCommand>,

    /// Prefer daemon connection for inference (falls back to direct if unavailable)
    #[arg(long, short = 'D')]
    pub use_daemon: bool,

    /// Require daemon connection (fail if daemon unavailable)
    #[arg(long)]
    pub daemon_required: bool,

    /// Don't auto-start daemon when using --daemon
    #[arg(long)]
    pub no_auto_start: bool,
}

/// Daemon management subcommand
#[derive(Subcommand, Debug, Clone)]
pub enum DaemonCommand {
    /// Start the daemon in the background
    Start {
        /// Run in foreground instead of daemonizing
        #[arg(long, short = 'f')]
        foreground: bool,

        /// Idle timeout in minutes (default: 30)
        #[arg(long, default_value = "30", value_name = "MINUTES")]
        idle_timeout: u64,
    },
    /// Stop the running daemon
    Stop {
        /// Force stop with SIGKILL if graceful shutdown fails
        #[arg(long)]
        force: bool,
    },
    /// Restart the daemon
    Restart {
        /// Run in foreground instead of daemonizing
        #[arg(long, short = 'f')]
        foreground: bool,
    },
    /// Show daemon status and statistics
    Status,
    /// Install system service (systemd/launchd)
    InstallService,
    /// Uninstall system service
    UninstallService,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_cli_parses_error_args() {
        let cli = Cli::parse_from(["why", "segmentation", "fault"]);
        assert_eq!(cli.error, vec!["segmentation", "fault"]);
        assert!(!cli.json);
    }

    #[test]
    fn test_cli_parses_json_flag() {
        let cli = Cli::parse_from(["why", "--json", "error"]);
        assert!(cli.json);
    }

    #[test]
    fn test_cli_parses_short_json_flag() {
        let cli = Cli::parse_from(["why", "-j", "error"]);
        assert!(cli.json);
    }

    #[test]
    fn test_cli_parses_stats_flag() {
        let cli = Cli::parse_from(["why", "--stats", "error"]);
        assert!(cli.stats);
    }

    #[test]
    fn test_cli_parses_completions() {
        let cli = Cli::parse_from(["why", "--completions", "bash"]);
        assert_eq!(cli.completions, Some(Shell::Bash));
    }

    #[test]
    fn test_cli_long_error_args() {
        let long_msg = "a]".repeat(1000);
        let cli = Cli::parse_from(["why", &long_msg]);
        assert_eq!(cli.error.len(), 1);
        assert_eq!(cli.error[0].len(), 2000);
    }

    #[test]
    fn test_cli_multiline_quoted_arg() {
        let cli = Cli::parse_from(["why", "line1\nline2\nline3"]);
        assert_eq!(cli.error.len(), 1);
        assert!(cli.error[0].contains('\n'));
        assert!(cli.error[0].contains("line1"));
        assert!(cli.error[0].contains("line3"));
    }

    #[test]
    fn test_cli_multiple_long_args() {
        let arg1 = "a".repeat(500);
        let arg2 = "b".repeat(500);
        let cli = Cli::parse_from(["why", &arg1, &arg2]);

        assert_eq!(cli.error.len(), 2);
        assert_eq!(cli.error[0].len(), 500);
        assert_eq!(cli.error[1].len(), 500);
    }

    #[test]
    fn test_cli_parses_debug_flag() {
        let cli = Cli::parse_from(["why", "--debug", "error"]);
        assert!(cli.debug);
    }

    #[test]
    fn test_cli_parses_short_debug_flag() {
        let cli = Cli::parse_from(["why", "-d", "error"]);
        assert!(cli.debug);
    }

    #[test]
    fn test_cli_debug_and_json_together() {
        let cli = Cli::parse_from(["why", "-d", "-j", "error"]);
        assert!(cli.debug);
        assert!(cli.json);
    }

    #[test]
    fn test_cli_parses_model_flag() {
        let cli = Cli::parse_from(["why", "--model", "/path/to/model.gguf", "error"]);
        assert_eq!(cli.model, Some(PathBuf::from("/path/to/model.gguf")));
    }

    #[test]
    fn test_cli_parses_short_model_flag() {
        let cli = Cli::parse_from(["why", "-m", "/path/to/model.gguf", "error"]);
        assert_eq!(cli.model, Some(PathBuf::from("/path/to/model.gguf")));
    }

    #[test]
    fn test_cli_parses_template_flag() {
        let cli = Cli::parse_from(["why", "--template", "gemma", "error"]);
        assert_eq!(cli.template, Some(ModelFamily::Gemma));
    }

    #[test]
    fn test_cli_parses_short_template_flag() {
        let cli = Cli::parse_from(["why", "-t", "qwen", "error"]);
        assert_eq!(cli.template, Some(ModelFamily::Qwen));
    }

    #[test]
    fn test_cli_parses_template_smollm() {
        let cli = Cli::parse_from(["why", "--template", "smollm", "error"]);
        assert_eq!(cli.template, Some(ModelFamily::Smollm));
    }

    #[test]
    fn test_cli_parses_list_models_flag() {
        let cli = Cli::parse_from(["why", "--list-models"]);
        assert!(cli.list_models);
    }

    #[test]
    fn test_cli_model_and_template_together() {
        let cli = Cli::parse_from([
            "why",
            "--model",
            "/path/to/gemma.gguf",
            "--template",
            "gemma",
            "error",
        ]);
        assert_eq!(cli.model, Some(PathBuf::from("/path/to/gemma.gguf")));
        assert_eq!(cli.template, Some(ModelFamily::Gemma));
    }

    #[test]
    fn test_cli_parses_stream_flag() {
        let cli = Cli::parse_from(["why", "--stream", "error"]);
        assert!(cli.stream);
    }

    #[test]
    fn test_cli_parses_short_stream_flag() {
        let cli = Cli::parse_from(["why", "-s", "error"]);
        assert!(cli.stream);
    }

    #[test]
    fn test_cli_stream_and_json_together() {
        let cli = Cli::parse_from(["why", "-s", "-j", "error"]);
        assert!(cli.stream);
        assert!(cli.json);
    }

    #[test]
    fn test_cli_parses_hook_bash() {
        let cli = Cli::parse_from(["why", "--hook", "bash"]);
        assert_eq!(cli.hook, Some(Shell::Bash));
    }

    #[test]
    fn test_cli_parses_hook_zsh() {
        let cli = Cli::parse_from(["why", "--hook", "zsh"]);
        assert_eq!(cli.hook, Some(Shell::Zsh));
    }

    #[test]
    fn test_cli_parses_hook_fish() {
        let cli = Cli::parse_from(["why", "--hook", "fish"]);
        assert_eq!(cli.hook, Some(Shell::Fish));
    }

    #[test]
    fn test_cli_parses_exit_code() {
        let cli = Cli::parse_from(["why", "--exit-code", "127"]);
        assert_eq!(cli.exit_code, Some(127));
    }

    #[test]
    fn test_cli_parses_last_command() {
        let cli = Cli::parse_from(["why", "--last-command", "npm run build"]);
        assert_eq!(cli.last_command, Some("npm run build".to_string()));
    }

    #[test]
    fn test_cli_parses_hook_mode_full() {
        let cli = Cli::parse_from(["why", "--exit-code", "1", "--last-command", "cargo build"]);
        assert_eq!(cli.exit_code, Some(1));
        assert_eq!(cli.last_command, Some("cargo build".to_string()));
    }
}
