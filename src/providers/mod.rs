//! AI Provider abstraction for error explanation.
//!
//! This module provides a unified interface for different AI providers
//! including local LLM inference and external APIs (Anthropic, OpenAI, OpenRouter).

pub mod anthropic;
pub mod local;
pub mod openai;
pub mod openrouter;

use anyhow::Result;
use async_trait::async_trait;
use clap::ValueEnum;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::context::CommandContext;

/// Available AI providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ProviderType {
    /// Local LLM inference using embedded or external GGUF model
    #[default]
    Local,
    /// Anthropic Claude API
    Anthropic,
    /// OpenAI API
    OpenAI,
    /// OpenRouter API (access to multiple models)
    OpenRouter,
}

impl fmt::Display for ProviderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProviderType::Local => write!(f, "local"),
            ProviderType::Anthropic => write!(f, "anthropic"),
            ProviderType::OpenAI => write!(f, "openai"),
            ProviderType::OpenRouter => write!(f, "openrouter"),
        }
    }
}

impl std::str::FromStr for ProviderType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "local" => Ok(ProviderType::Local),
            "anthropic" => Ok(ProviderType::Anthropic),
            "openai" => Ok(ProviderType::OpenAI),
            "openrouter" => Ok(ProviderType::OpenRouter),
            _ => Err(format!(
                "Unknown provider: {}. Valid options: local, anthropic, openai, openrouter",
                s
            )),
        }
    }
}

/// Result of an explanation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationResult {
    /// Raw response from the provider
    pub raw_response: String,
    /// Whether the response was streamed
    pub streamed: bool,
    /// Provider that generated the response
    pub provider: ProviderType,
    /// Model used for generation
    pub model: String,
}

/// Callback for streaming tokens
pub type StreamCallback = Box<dyn FnMut(&str) -> Result<bool> + Send>;

/// Provider trait for AI-powered error explanation
///
/// This trait defines the interface that all AI providers must implement.
/// It supports both synchronous and streaming modes.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Get the provider type
    fn provider_type(&self) -> ProviderType;

    /// Get the provider name for display
    fn name(&self) -> &'static str;

    /// Check if this provider requires an API key
    fn requires_api_key(&self) -> bool;

    /// Check if the provider is available (API key present, model loaded, etc.)
    fn is_available(&self) -> bool;

    /// Explain an error message
    ///
    /// # Arguments
    /// * `error` - The error message to explain
    /// * `context` - Optional command context (command, exit code, output, etc.)
    ///
    /// # Returns
    /// The explanation result containing the response
    async fn explain(
        &self,
        error: &str,
        context: Option<&CommandContext>,
    ) -> Result<ExplanationResult>;

    /// Explain an error message with streaming output
    ///
    /// # Arguments
    /// * `error` - The error message to explain
    /// * `context` - Optional command context
    /// * `callback` - Callback function for each token, returns false to stop streaming
    ///
    /// # Returns
    /// The complete response after streaming finishes
    async fn explain_streaming(
        &self,
        error: &str,
        context: Option<&CommandContext>,
        callback: StreamCallback,
    ) -> Result<ExplanationResult>;

    /// Get the current model name/identifier
    fn model_name(&self) -> &str;
}

/// Provider configuration for API-based providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// API key (if required)
    pub api_key: Option<String>,
    /// Model identifier
    pub model: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Base URL override (for custom endpoints)
    pub base_url: Option<String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            model: String::new(),
            max_tokens: 1024,
            base_url: None,
        }
    }
}

/// Error types for provider operations
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("API key not configured for {provider}. Set {env_var} environment variable.")]
    MissingApiKey { provider: String, env_var: String },

    #[error("Provider not available: {0}")]
    NotAvailable(String),

    #[error("API error from {provider}: {message}")]
    ApiError { provider: String, message: String },

    #[error("Rate limited by {provider}. Please wait and try again.")]
    RateLimited { provider: String },

    #[error("Network error: {0}")]
    NetworkError(String),

    #[error("Invalid response from {provider}: {message}")]
    InvalidResponse { provider: String, message: String },

    #[error("Model not found: {0}")]
    ModelNotFound(String),
}

/// Get the appropriate environment variable name for a provider's API key
pub fn get_api_key_env_var(provider: ProviderType) -> &'static str {
    match provider {
        ProviderType::Anthropic => "ANTHROPIC_API_KEY",
        ProviderType::OpenAI => "OPENAI_API_KEY",
        ProviderType::OpenRouter => "OPENROUTER_API_KEY",
        ProviderType::Local => "",
    }
}

/// Get API key from environment for a provider
pub fn get_api_key(provider: ProviderType) -> Option<String> {
    let env_var = get_api_key_env_var(provider);
    if env_var.is_empty() {
        return None;
    }
    std::env::var(env_var).ok().filter(|s| !s.is_empty())
}

/// List all available providers
pub fn list_providers() -> Vec<(ProviderType, &'static str, bool)> {
    vec![
        (
            ProviderType::Local,
            "Local LLM (embedded/external GGUF)",
            true,
        ),
        (
            ProviderType::Anthropic,
            "Anthropic Claude API",
            get_api_key(ProviderType::Anthropic).is_some(),
        ),
        (
            ProviderType::OpenAI,
            "OpenAI API",
            get_api_key(ProviderType::OpenAI).is_some(),
        ),
        (
            ProviderType::OpenRouter,
            "OpenRouter API",
            get_api_key(ProviderType::OpenRouter).is_some(),
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_type_display() {
        assert_eq!(format!("{}", ProviderType::Local), "local");
        assert_eq!(format!("{}", ProviderType::Anthropic), "anthropic");
        assert_eq!(format!("{}", ProviderType::OpenAI), "openai");
        assert_eq!(format!("{}", ProviderType::OpenRouter), "openrouter");
    }

    #[test]
    fn test_provider_type_from_str() {
        assert_eq!(
            "local".parse::<ProviderType>().unwrap(),
            ProviderType::Local
        );
        assert_eq!(
            "anthropic".parse::<ProviderType>().unwrap(),
            ProviderType::Anthropic
        );
        assert_eq!(
            "openai".parse::<ProviderType>().unwrap(),
            ProviderType::OpenAI
        );
        assert_eq!(
            "openrouter".parse::<ProviderType>().unwrap(),
            ProviderType::OpenRouter
        );
        assert_eq!(
            "ANTHROPIC".parse::<ProviderType>().unwrap(),
            ProviderType::Anthropic
        );
        assert!("invalid".parse::<ProviderType>().is_err());
    }

    #[test]
    fn test_provider_type_default() {
        assert_eq!(ProviderType::default(), ProviderType::Local);
    }

    #[test]
    fn test_get_api_key_env_var() {
        assert_eq!(
            get_api_key_env_var(ProviderType::Anthropic),
            "ANTHROPIC_API_KEY"
        );
        assert_eq!(get_api_key_env_var(ProviderType::OpenAI), "OPENAI_API_KEY");
        assert_eq!(
            get_api_key_env_var(ProviderType::OpenRouter),
            "OPENROUTER_API_KEY"
        );
        assert_eq!(get_api_key_env_var(ProviderType::Local), "");
    }
}
