//! Local LLM provider using llama-cpp.
//!
//! This provider uses the embedded or external GGUF model for inference.

use anyhow::Result;
use async_trait::async_trait;

use super::{ExplanationResult, Provider, ProviderType, StreamCallback};
use crate::context::CommandContext;

/// Local LLM provider
pub struct LocalProvider {
    /// Path to the model file
    model_path: Option<std::path::PathBuf>,
    /// Model family for prompt template selection
    model_family: Option<crate::model::ModelFamily>,
}

impl LocalProvider {
    /// Create a new local provider
    pub fn new() -> Self {
        Self {
            model_path: None,
            model_family: None,
        }
    }

    /// Create with a specific model path
    pub fn with_model(model_path: std::path::PathBuf) -> Self {
        Self {
            model_path: Some(model_path),
            model_family: None,
        }
    }

    /// Set the model family
    pub fn with_family(mut self, family: crate::model::ModelFamily) -> Self {
        self.model_family = Some(family);
        self
    }
}

impl Default for LocalProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for LocalProvider {
    fn provider_type(&self) -> ProviderType {
        ProviderType::Local
    }

    fn name(&self) -> &'static str {
        "Local LLM"
    }

    fn requires_api_key(&self) -> bool {
        false
    }

    fn is_available(&self) -> bool {
        // Local provider is always available if we can find/load a model
        true
    }

    async fn explain(
        &self,
        error: &str,
        context: Option<&CommandContext>,
    ) -> Result<ExplanationResult> {
        // Build enhanced input with context
        let input = if let Some(ctx) = context {
            if ctx.is_empty() {
                error.to_string()
            } else {
                format!("{}\n\n{}", ctx.format_for_prompt(), error)
            }
        } else {
            error.to_string()
        };

        // Get model path
        let model_info = crate::model::get_model_path(self.model_path.as_ref())?;
        let model_path = &model_info.path;

        // Determine model family
        let model_family = self.model_family.unwrap_or_else(|| {
            model_info
                .embedded_family
                .unwrap_or_else(|| crate::model::detect_model_family(model_path))
        });

        // Build prompt
        let prompt = crate::model::build_prompt(&input, model_family);

        // Run inference
        let params = crate::model::SamplingParams::default();
        let (response, _stats) =
            crate::model::run_inference_with_callback(model_path, &prompt, &params, None)?;

        Ok(ExplanationResult {
            raw_response: response,
            streamed: false,
            provider: ProviderType::Local,
            model: model_path.display().to_string(),
        })
    }

    async fn explain_streaming(
        &self,
        error: &str,
        context: Option<&CommandContext>,
        mut callback: StreamCallback,
    ) -> Result<ExplanationResult> {
        // Build enhanced input with context
        let input = if let Some(ctx) = context {
            if ctx.is_empty() {
                error.to_string()
            } else {
                format!("{}\n\n{}", ctx.format_for_prompt(), error)
            }
        } else {
            error.to_string()
        };

        // Get model path
        let model_info = crate::model::get_model_path(self.model_path.as_ref())?;
        let model_path = &model_info.path;

        // Determine model family
        let model_family = self.model_family.unwrap_or_else(|| {
            model_info
                .embedded_family
                .unwrap_or_else(|| crate::model::detect_model_family(model_path))
        });

        // Build prompt
        let prompt = crate::model::build_prompt(&input, model_family);

        // Create callback wrapper
        let token_callback: crate::model::TokenCallback =
            Box::new(move |token: &str| callback(token));

        // Run inference with streaming
        let params = crate::model::SamplingParams::default();
        let (response, _stats) = crate::model::run_inference_with_callback(
            model_path,
            &prompt,
            &params,
            Some(token_callback),
        )?;

        Ok(ExplanationResult {
            raw_response: response,
            streamed: true,
            provider: ProviderType::Local,
            model: model_path.display().to_string(),
        })
    }

    fn model_name(&self) -> &str {
        "local"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_provider_creation() {
        let provider = LocalProvider::new();
        assert_eq!(provider.provider_type(), ProviderType::Local);
        assert_eq!(provider.name(), "Local LLM");
        assert!(!provider.requires_api_key());
    }
}
