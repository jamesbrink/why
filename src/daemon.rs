//! Daemon mode for pre-loaded model inference.
//!
//! This module provides the types and utilities for daemon mode, which keeps
//! the model loaded in memory for faster inference times.
//!
//! Note: The main daemon implementation is currently in main.rs due to
//! tight coupling with the model and CLI systems. This module exports
//! the core types used by daemon mode.

use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;

use crate::output::ErrorExplanation;

/// Get the daemon socket path
/// Uses XDG_RUNTIME_DIR if available, falls back to /tmp
#[cfg(unix)]
pub fn get_socket_path() -> PathBuf {
    if let Ok(runtime_dir) = env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(runtime_dir).join("why.sock");
    }

    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/tmp/why-{}.sock", uid))
}

#[cfg(not(unix))]
pub fn get_socket_path() -> PathBuf {
    env::temp_dir().join("why.sock")
}

/// Get the daemon PID file path
#[cfg(unix)]
pub fn get_pid_path() -> PathBuf {
    if let Ok(runtime_dir) = env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(runtime_dir).join("why.pid");
    }

    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/tmp/why-{}.pid", uid))
}

#[cfg(not(unix))]
pub fn get_pid_path() -> PathBuf {
    env::temp_dir().join("why.pid")
}

/// Get the daemon log file path
#[cfg(unix)]
pub fn get_log_path() -> Option<PathBuf> {
    dirs::cache_dir().map(|p| p.join("why").join("daemon.log"))
}

#[cfg(not(unix))]
pub fn get_log_path() -> Option<PathBuf> {
    dirs::cache_dir().map(|p| p.join("why").join("daemon.log"))
}

/// Daemon request protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRequest {
    /// Action to perform
    pub action: DaemonAction,
    /// Input for explain action
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    /// Options for the request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<DaemonRequestOptions>,
}

/// Daemon request options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DaemonRequestOptions {
    /// Enable streaming response
    #[serde(default)]
    pub stream: bool,
    /// Output as JSON
    #[serde(default)]
    pub json: bool,
    /// Enable source context
    #[serde(default)]
    pub context: bool,
    /// Context lines
    #[serde(default)]
    pub context_lines: Option<usize>,
    /// Context root path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_root: Option<String>,
}

/// Daemon action types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DaemonAction {
    /// Explain an error
    Explain,
    /// Ping the daemon (health check)
    Ping,
    /// Request daemon shutdown
    Shutdown,
    /// Get daemon statistics
    Stats,
}

/// Daemon response protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonResponse {
    /// Response type
    #[serde(rename = "type")]
    pub response_type: DaemonResponseType,
    /// Token content (for streaming)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Complete explanation (for final response)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<ErrorExplanationResponse>,
    /// Error message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Stats (for stats response)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<DaemonStats>,
}

impl DaemonResponse {
    /// Create a token response
    pub fn token(content: &str) -> Self {
        Self {
            response_type: DaemonResponseType::Token,
            content: Some(content.to_string()),
            explanation: None,
            error: None,
            stats: None,
        }
    }

    /// Create a complete response with explanation
    pub fn complete(explanation: ErrorExplanationResponse) -> Self {
        Self {
            response_type: DaemonResponseType::Complete,
            content: None,
            explanation: Some(explanation),
            error: None,
            stats: None,
        }
    }

    /// Create an error response
    pub fn error(message: &str) -> Self {
        Self {
            response_type: DaemonResponseType::Error,
            content: None,
            explanation: None,
            error: Some(message.to_string()),
            stats: None,
        }
    }

    /// Create a pong response
    pub fn pong() -> Self {
        Self {
            response_type: DaemonResponseType::Pong,
            content: None,
            explanation: None,
            error: None,
            stats: None,
        }
    }

    /// Create a stats response
    pub fn stats(stats: DaemonStats) -> Self {
        Self {
            response_type: DaemonResponseType::Stats,
            content: None,
            explanation: None,
            error: None,
            stats: Some(stats),
        }
    }

    /// Create a shutdown acknowledgment
    pub fn shutdown_ack() -> Self {
        Self {
            response_type: DaemonResponseType::ShutdownAck,
            content: None,
            explanation: None,
            error: None,
            stats: None,
        }
    }
}

/// Daemon response types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DaemonResponseType {
    /// Streaming token
    Token,
    /// Complete response
    Complete,
    /// Error response
    Error,
    /// Pong (ping response)
    Pong,
    /// Stats response
    Stats,
    /// Shutdown acknowledgment
    #[serde(rename = "shutdown_ack")]
    ShutdownAck,
}

/// Error explanation for daemon response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorExplanationResponse {
    pub error: String,
    pub summary: String,
    pub explanation: String,
    pub suggestion: String,
}

impl From<&ErrorExplanation> for ErrorExplanationResponse {
    fn from(exp: &ErrorExplanation) -> Self {
        Self {
            error: exp.error.clone(),
            summary: exp.summary.clone(),
            explanation: exp.explanation.clone(),
            suggestion: exp.suggestion.clone(),
        }
    }
}

/// Daemon statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DaemonStats {
    /// Daemon uptime in seconds
    pub uptime_seconds: u64,
    /// Total requests served
    pub requests_served: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Current memory usage (RSS) in MB
    pub memory_mb: f64,
    /// Model family
    pub model_family: String,
    /// Whether model is loaded
    pub model_loaded: bool,
}
