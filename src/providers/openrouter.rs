//! OpenRouter API provider.
//!
//! This provider uses the OpenRouter API for access to multiple AI models.

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use super::{
    get_api_key, ExplanationResult, Provider, ProviderError, ProviderType, StreamCallback,
};
use crate::context::CommandContext;

/// OpenRouter API endpoint
const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// System prompt for error explanation
const SYSTEM_PROMPT: &str = r#"You are a helpful programming assistant that explains error messages.
When given an error message or stack trace, provide a clear explanation with:

1. SUMMARY: A one-line summary of what went wrong
2. EXPLANATION: A detailed explanation of why this error occurred
3. SUGGESTION: Concrete steps to fix the issue

Be concise and practical. Focus on the most likely cause and solution."#;

/// OpenRouter provider
pub struct OpenRouterProvider {
    /// HTTP client
    client: Client,
    /// API key
    api_key: String,
    /// Model to use (format: provider/model)
    model: String,
    /// Maximum tokens to generate
    max_tokens: u32,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider
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
            get_api_key(ProviderType::OpenRouter).ok_or_else(|| ProviderError::MissingApiKey {
                provider: "OpenRouter".to_string(),
                env_var: "OPENROUTER_API_KEY".to_string(),
            })?;

        Ok(Self::new(api_key, model, max_tokens))
    }

    /// Build the request body
    fn build_request(&self, content: &str, stream: bool) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: self.model.clone(),
            max_tokens: Some(self.max_tokens),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: SYSTEM_PROMPT.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: content.to_string(),
                },
            ],
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
impl Provider for OpenRouterProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::OpenRouter
    }

    fn name(&self) -> &'static str {
        "OpenRouter"
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
            .post(OPENROUTER_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "https://github.com/jamesbrink/why")
            .header("X-Title", "why - Error Explanation Tool")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to OpenRouter API")?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                bail!(ProviderError::RateLimited {
                    provider: "OpenRouter".to_string()
                });
            }
            bail!(ProviderError::ApiError {
                provider: "OpenRouter".to_string(),
                message: format!("HTTP {}: {}", status, error_body),
            });
        }

        let response_body: ChatCompletionResponse = response
            .json()
            .await
            .context("Failed to parse OpenRouter response")?;

        let content = response_body
            .choices
            .into_iter()
            .filter_map(|choice| choice.message.map(|m| m.content))
            .collect::<Vec<_>>()
            .join("");

        Ok(ExplanationResult {
            raw_response: content,
            streamed: false,
            provider: ProviderType::OpenRouter,
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
            .post(OPENROUTER_API_URL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("HTTP-Referer", "https://github.com/jamesbrink/why")
            .header("X-Title", "why - Error Explanation Tool")
            .json(&request)
            .send()
            .await
            .context("Failed to send request to OpenRouter API")?;

        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_default();
            if status.as_u16() == 429 {
                bail!(ProviderError::RateLimited {
                    provider: "OpenRouter".to_string()
                });
            }
            bail!(ProviderError::ApiError {
                provider: "OpenRouter".to_string(),
                message: format!("HTTP {}: {}", status, error_body),
            });
        }

        // Parse SSE stream (OpenAI-compatible)
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

                        if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                            for choice in chunk.choices {
                                if let Some(delta) = choice.delta {
                                    if let Some(content) = delta.content {
                                        full_response.push_str(&content);
                                        if !callback(&content)? {
                                            return Ok(ExplanationResult {
                                                raw_response: full_response,
                                                streamed: true,
                                                provider: ProviderType::OpenRouter,
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

        Ok(ExplanationResult {
            raw_response: full_response,
            streamed: true,
            provider: ProviderType::OpenRouter,
            model: self.model.clone(),
        })
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

// API types (OpenAI-compatible)

#[derive(Debug, Serialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Option<ChatMessage>,
}

#[derive(Debug, Deserialize)]
struct StreamChunk {
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: Option<Delta>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openrouter_provider_creation() {
        let provider = OpenRouterProvider::new(
            "test-key".to_string(),
            "anthropic/claude-sonnet-4".to_string(),
            1024,
        );
        assert_eq!(provider.provider_type(), ProviderType::OpenRouter);
        assert_eq!(provider.name(), "OpenRouter");
        assert!(provider.requires_api_key());
        assert!(provider.is_available());
    }

    #[test]
    fn test_format_input_without_context() {
        let provider = OpenRouterProvider::new(
            "test-key".to_string(),
            "anthropic/claude-sonnet-4".to_string(),
            1024,
        );
        let input = provider.format_input("segmentation fault", None);
        assert!(input.contains("segmentation fault"));
    }
}
