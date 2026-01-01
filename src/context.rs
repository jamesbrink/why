//! Command context for shell integration.
//!
//! This module provides structures for capturing shell command context
//! to provide richer error explanations.

use serde::{Deserialize, Serialize};

/// Context from a shell command execution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CommandContext {
    /// The command that was executed
    pub command: Option<String>,

    /// Exit code from the command
    pub exit_code: Option<i32>,

    /// Standard error output (last N lines)
    pub stderr: Option<String>,

    /// Standard output (if captured)
    pub stdout: Option<String>,

    /// Working directory where command was run
    pub working_dir: Option<String>,

    /// Shell type (bash, zsh, fish, etc.)
    pub shell: Option<String>,

    /// Timestamp when command was executed
    pub timestamp: Option<String>,
}

impl CommandContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self::default()
    }

    /// Create context with just a command and exit code
    pub fn with_command(command: impl Into<String>, exit_code: i32) -> Self {
        Self {
            command: Some(command.into()),
            exit_code: Some(exit_code),
            ..Default::default()
        }
    }

    /// Builder: set command
    pub fn command(mut self, cmd: impl Into<String>) -> Self {
        self.command = Some(cmd.into());
        self
    }

    /// Builder: set exit code
    pub fn exit_code(mut self, code: i32) -> Self {
        self.exit_code = Some(code);
        self
    }

    /// Builder: set stderr
    pub fn stderr(mut self, stderr: impl Into<String>) -> Self {
        self.stderr = Some(stderr.into());
        self
    }

    /// Builder: set stdout
    pub fn stdout(mut self, stdout: impl Into<String>) -> Self {
        self.stdout = Some(stdout.into());
        self
    }

    /// Builder: set working directory
    pub fn working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Builder: set shell type
    pub fn shell(mut self, shell: impl Into<String>) -> Self {
        self.shell = Some(shell.into());
        self
    }

    /// Builder: set timestamp
    pub fn timestamp(mut self, ts: impl Into<String>) -> Self {
        self.timestamp = Some(ts.into());
        self
    }

    /// Check if context has any useful information
    pub fn is_empty(&self) -> bool {
        self.command.is_none()
            && self.exit_code.is_none()
            && self.stderr.is_none()
            && self.stdout.is_none()
    }

    /// Format context for inclusion in a prompt
    pub fn format_for_prompt(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref cmd) = self.command {
            parts.push(format!("Command: {}", cmd));
        }

        if let Some(code) = self.exit_code {
            let interpretation = interpret_exit_code(code);
            parts.push(format!("Exit code: {} ({})", code, interpretation));
        }

        if let Some(ref dir) = self.working_dir {
            parts.push(format!("Working directory: {}", dir));
        }

        if let Some(ref shell) = self.shell {
            parts.push(format!("Shell: {}", shell));
        }

        if let Some(ref stderr) = self.stderr {
            if !stderr.trim().is_empty() {
                parts.push(format!("\nStderr:\n{}", stderr));
            }
        }

        if let Some(ref stdout) = self.stdout {
            if !stdout.trim().is_empty() {
                parts.push(format!("\nStdout:\n{}", stdout));
            }
        }

        parts.join("\n")
    }

    /// Get last N lines of stderr
    pub fn last_stderr_lines(&self, n: usize) -> Option<String> {
        self.stderr.as_ref().map(|s| {
            let lines: Vec<&str> = s.lines().collect();
            if lines.len() <= n {
                s.clone()
            } else {
                lines[lines.len() - n..].join("\n")
            }
        })
    }
}

/// Interpret common exit codes
fn interpret_exit_code(code: i32) -> &'static str {
    match code {
        0 => "success",
        1 => "general error",
        2 => "misuse of shell command",
        126 => "permission problem or command not executable",
        127 => "command not found",
        128 => "invalid exit argument",
        130 => "terminated by Ctrl+C (SIGINT)",
        137 => "killed (SIGKILL)",
        139 => "segmentation fault (SIGSEGV)",
        143 => "terminated (SIGTERM)",
        255 => "exit status out of range",
        _ if code > 128 && code < 256 => "terminated by signal",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_new() {
        let ctx = CommandContext::new();
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_context_with_command() {
        let ctx = CommandContext::with_command("ls -la", 0);
        assert_eq!(ctx.command, Some("ls -la".to_string()));
        assert_eq!(ctx.exit_code, Some(0));
        assert!(!ctx.is_empty());
    }

    #[test]
    fn test_context_builder() {
        let ctx = CommandContext::new()
            .command("cargo build")
            .exit_code(1)
            .stderr("error[E0382]: borrow of moved value")
            .working_dir("/home/user/project")
            .shell("bash");

        assert_eq!(ctx.command, Some("cargo build".to_string()));
        assert_eq!(ctx.exit_code, Some(1));
        assert!(ctx.stderr.as_ref().unwrap().contains("E0382"));
        assert_eq!(ctx.working_dir, Some("/home/user/project".to_string()));
        assert_eq!(ctx.shell, Some("bash".to_string()));
    }

    #[test]
    fn test_format_for_prompt() {
        let ctx = CommandContext::with_command("npm run build", 1)
            .stderr("Error: Cannot find module 'react'");

        let prompt = ctx.format_for_prompt();
        assert!(prompt.contains("Command: npm run build"));
        assert!(prompt.contains("Exit code: 1"));
        assert!(prompt.contains("Cannot find module"));
    }

    #[test]
    fn test_last_stderr_lines() {
        let ctx = CommandContext::new().stderr("line 1\nline 2\nline 3\nline 4\nline 5");

        let last_3 = ctx.last_stderr_lines(3);
        assert_eq!(last_3, Some("line 3\nline 4\nline 5".to_string()));

        let last_10 = ctx.last_stderr_lines(10);
        assert_eq!(
            last_10,
            Some("line 1\nline 2\nline 3\nline 4\nline 5".to_string())
        );
    }

    #[test]
    fn test_interpret_exit_code() {
        assert_eq!(interpret_exit_code(0), "success");
        assert_eq!(interpret_exit_code(1), "general error");
        assert_eq!(interpret_exit_code(127), "command not found");
        assert_eq!(interpret_exit_code(130), "terminated by Ctrl+C (SIGINT)");
        assert_eq!(interpret_exit_code(139), "segmentation fault (SIGSEGV)");
    }
}
