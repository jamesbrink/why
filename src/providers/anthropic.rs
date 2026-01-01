//! Anthropic Claude API provider.
//!
//! This provider uses the Anthropic Messages API for error explanation.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{
    get_api_key, ExplanationResult, Provider, ProviderError, ProviderType, StreamCallback,
};
use crate::context::CommandContext;

/// Anthropic API endpoint
const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";

/// Anthropic API version
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// System prompt for error explanation
const SYSTEM_PROMPT: &str = r#"You are a helpful programming assistant that explains error messages.
When given an error message or stack trace, provide a clear explanation with:

1. SUMMARY: A one-line summary of what went wrong
2. EXPLANATION: A detailed explanation of why this error occurred
3. SUGGESTION: Concrete steps to fix the issue

Be concise and practical. Focus on the most likely cause and solution."#;

/// Anthropic provider
pub struct AnthropicProvider {
    /// HTTP client
    client: Client,
    /// API key
    api_key: String,
    /// Model to use
    model: String,
    /// Maximum tokens to generate
    max_tokens: u32,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider
    pub fn new(api_key: String, model: String, max_tokens: u32) -> Self {
        Self {
            client: Client::new(),
            api_key,
            model,
            max_tokens,
        }
    }

    /// Create from environment variables and config
    pub fn from_env(model: String, max_tokens: u32) -> Result<Self> {
        let api_key =
            get_api_key(ProviderType::Anthropic).ok_or_else(|| ProviderError::MissingApiKey {
                provider: "Anthropic".to_string(),
                env_var: "ANTHROPIC_API_KEY".to_string(),
            })?;

        Ok(Self::new(api_key, model, max_tokens))
    }

    /// Build the request body
    fn build_request(&self, content: &str, stream: bool) -> MessagesRequest {
        MessagesRequest {
            model: self.model.clone(),
            max_tokens: self.max_tokens,
            system: Some(SYSTEM_PROMPT.to_string()),
            messages: vec![Message {
                role: "user".to_string(),
                content: content.to_string(),
            }],
            stream: Some(stream),
        }
    }

    /// Format input with context
    fn format_input(&self, error: &str, context: Option<&CommandContext>) -> String {
        if let Some(ctx) = context {
            if !ctx.is_empty() {
                return format!(
                    "{}\n\nError to explain:\n{}",
                    ctx.format_for_prompt(),
                    error
                );
            }
        }
        format!("Please explain this error:\n\n{}", error)
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Anthropic
    }

    fn name(&self) -> &'static str {
        "Anthropic Claude"
    }

    fn requires_api_key(&self) -> bool {
        true
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    async fn explain(
        &self,
        error: &str,
        context: Option<&CommandContext>,
    ) -> Result<ExplanationResult> {
        let input = self.format_input(error, context);
        let request = self.build_request(&input, false);

        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Anthropic API")?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                bail!(ProviderError::RateLimited {
                    provider: "Anthropic".to_string()
                });
            }
            bail!(ProviderError::ApiError {
                provider: "Anthropic".to_string(),
                message: format!("HTTP {}: {}", status, error_body),
            });
        }

        let response_body: MessagesResponse = response
            .json()
            .await
            .context("Failed to parse Anthropic response")?;

        let content = response_body
            .content
            .into_iter()
            .filter_map(|block| {
                if block.content_type == "text" {
                    Some(block.text)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");

        Ok(ExplanationResult {
            raw_response: content,
            streamed: false,
            provider: ProviderType::Anthropic,
            model: self.model.clone(),
        })
    }

    async fn explain_streaming(
        &self,
        error: &str,
        context: Option<&CommandContext>,
        mut callback: StreamCallback,
    ) -> Result<ExplanationResult> {
        let input = self.format_input(error, context);
        let request = self.build_request(&input, true);

        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to Anthropic API")?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                bail!(ProviderError::RateLimited {
                    provider: "Anthropic".to_string()
                });
            }
            bail!(ProviderError::ApiError {
                provider: "Anthropic".to_string(),
                message: format!("HTTP {}: {}", status, error_body),
            });
        }

        // Parse SSE stream
        let mut full_response = String::new();
        let mut stream = response.bytes_stream();

        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Failed to read stream chunk")?;
            let text = String::from_utf8_lossy(&chunk);
            buffer.push_str(&text);

            // Process complete SSE events
            while let Some(pos) = buffer.find("\n\n") {
                let event = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                // Parse SSE event
                for line in event.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            continue;
                        }

                        if let Ok(event) = serde_json::from_str::<StreamEvent>(data) {
                            if event.event_type == "content_block_delta" {
                                if let Some(delta) = event.delta {
                                    if delta.delta_type == "text_delta" {
                                        if let Some(text) = delta.text {
                                            full_response.push_str(&text);
                                            if !callback(&text)? {
                                                // Callback requested stop
                                                return Ok(ExplanationResult {
                                                    raw_response: full_response,
                                                    streamed: true,
                                                    provider: ProviderType::Anthropic,
                                                    model: self.model.clone(),
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(ExplanationResult {
            raw_response: full_response,
            streamed: true,
            provider: ProviderType::Anthropic,
            model: self.model.clone(),
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// API types

#[derive(Debug, Serialize)]
struct MessagesRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
}

#[derive(Debug, Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    content_type: String,
    #[serde(default)]
    text: String,
}

#[derive(Debug, Deserialize)]
struct StreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    #[serde(default)]
    delta: Option<Delta>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    #[serde(rename = "type")]
    delta_type: String,
    #[serde(default)]
    text: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anthropic_provider_creation() {
        let provider = AnthropicProvider::new(
            "test-key".to_string(),
            "claude-sonnet-4-20250514".to_string(),
            1024,
        );
        assert_eq!(provider.provider_type(), ProviderType::Anthropic);
        assert_eq!(provider.name(), "Anthropic Claude");
        assert!(provider.requires_api_key());
        assert!(provider.is_available());
    }

    #[test]
    fn test_format_input_without_context() {
        let provider = AnthropicProvider::new(
            "test-key".to_string(),
            "claude-sonnet-4-20250514".to_string(),
            1024,
        );
        let input = provider.format_input("segmentation fault", None);
        assert!(input.contains("segmentation fault"));
    }

    #[test]
    fn test_format_input_with_context() {
        let provider = AnthropicProvider::new(
            "test-key".to_string(),
            "claude-sonnet-4-20250514".to_string(),
            1024,
        );
        let ctx = CommandContext::with_command("cargo build", 1);
        let input = provider.format_input("error[E0382]", Some(&ctx));
        assert!(input.contains("cargo build"));
        assert!(input.contains("E0382"));
    }
}
