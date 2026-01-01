//! Why - Error explanation tool using local LLM
//!
//! This library provides the core functionality for the `why` CLI tool,
//! including stack trace parsing, model inference, and error explanation.

pub mod cli;
pub mod config;
pub mod context;
pub mod daemon;
pub mod hooks;
pub mod model;
pub mod output;
pub mod providers;
pub mod stack_trace;
pub mod watch;

// Re-export commonly used types
pub use cli::{Cli, DaemonCommand};
pub use config::Config;
pub use context::CommandContext;
pub use model::{ModelFamily, SamplingParams};
pub use output::ErrorExplanation;
pub use providers::ProviderType;
pub use stack_trace::{Language, StackFrame, StackTrace, StackTraceParserRegistry};
