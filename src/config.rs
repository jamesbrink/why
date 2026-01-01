//! Configuration system for the `why` tool.

use regex::Regex;
use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;

use crate::providers::ProviderType;

/// Configuration for hook behavior
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct HookConfig {
    /// Always explain without prompting
    pub auto_explain: bool,
    /// Exit codes to skip (e.g., [0, 130] for success and Ctrl+C)
    pub skip_exit_codes: Vec<i32>,
    /// Minimum number of stderr lines to trigger explanation
    pub min_stderr_lines: usize,
    /// Command patterns to ignore
    pub ignore_commands: IgnoreCommandsConfig,
}

impl Default for HookConfig {
    fn default() -> Self {
        Self {
            auto_explain: false,
            skip_exit_codes: vec![0, 130], // Success and Ctrl+C
            min_stderr_lines: 1,
            ignore_commands: IgnoreCommandsConfig::default(),
        }
    }
}

/// Command patterns to ignore
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct IgnoreCommandsConfig {
    /// Regex patterns for commands to ignore
    pub patterns: Vec<String>,
}

impl Default for IgnoreCommandsConfig {
    fn default() -> Self {
        Self {
            patterns: vec![
                "^cd ".to_string(),
                "^ls".to_string(),
                "^echo ".to_string(),
                "^pwd$".to_string(),
                "^clear$".to_string(),
            ],
        }
    }
}

/// Provider-specific configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct ProviderConfig {
    /// Default provider to use
    pub default: ProviderType,
    /// Anthropic-specific settings
    pub anthropic: AnthropicConfig,
    /// OpenAI-specific settings
    pub openai: OpenAIConfig,
    /// OpenRouter-specific settings
    pub openrouter: OpenRouterConfig,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            default: ProviderType::Local,
            anthropic: AnthropicConfig::default(),
            openai: OpenAIConfig::default(),
            openrouter: OpenRouterConfig::default(),
        }
    }
}

/// Anthropic provider configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct AnthropicConfig {
    /// Model to use (e.g., "claude-sonnet-4-20250514")
    pub model: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 1024,
        }
    }
}

/// OpenAI provider configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct OpenAIConfig {
    /// Model to use (e.g., "gpt-4o-mini")
    pub model: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
}

impl Default for OpenAIConfig {
    fn default() -> Self {
        Self {
            model: "gpt-4o-mini".to_string(),
            max_tokens: 1024,
        }
    }
}

/// OpenRouter provider configuration
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct OpenRouterConfig {
    /// Model to use (e.g., "anthropic/claude-sonnet-4")
    pub model: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
}

impl Default for OpenRouterConfig {
    fn default() -> Self {
        Self {
            model: "anthropic/claude-sonnet-4".to_string(),
            max_tokens: 1024,
        }
    }
}

/// Root configuration structure
#[derive(Debug, Deserialize, Clone, Default)]
#[serde(default)]
pub struct Config {
    pub hook: HookConfig,
    pub provider: ProviderConfig,
}

impl Config {
    /// Load config from file, returning default config if file doesn't exist
    pub fn load() -> Self {
        Self::load_from_path(Self::config_path())
    }

    /// Load config from a specific path
    pub fn load_from_path(path: Option<PathBuf>) -> Self {
        let Some(path) = path else {
            return Self::default();
        };

        if !path.exists() {
            return Self::default();
        }

        std::fs::read_to_string(&path)
            .ok()
            .and_then(|contents| toml::from_str(&contents).ok())
            .unwrap_or_default()
    }

    /// Get the config file path (~/.config/why/config.toml)
    pub fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("why").join("config.toml"))
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        // WHY_HOOK_AUTO=1 enables auto-explain
        if env::var("WHY_HOOK_AUTO").map(|v| v == "1").unwrap_or(false) {
            self.hook.auto_explain = true;
        }

        // WHY_PROVIDER overrides default provider
        if let Ok(provider_str) = env::var("WHY_PROVIDER") {
            if let Ok(provider) = provider_str.parse::<ProviderType>() {
                self.provider.default = provider;
            }
        }

        // WHY_MODEL overrides model for the configured provider
        if let Ok(model) = env::var("WHY_MODEL") {
            match self.provider.default {
                ProviderType::Anthropic => self.provider.anthropic.model = model,
                ProviderType::OpenAI => self.provider.openai.model = model,
                ProviderType::OpenRouter => self.provider.openrouter.model = model,
                ProviderType::Local => {} // Local model is set via --model flag
            }
        }
    }

    /// Get the configured provider type (with env override)
    pub fn get_provider(&self) -> ProviderType {
        self.provider.default
    }

    /// Get model for a specific provider
    pub fn get_model(&self, provider: ProviderType) -> &str {
        match provider {
            ProviderType::Anthropic => &self.provider.anthropic.model,
            ProviderType::OpenAI => &self.provider.openai.model,
            ProviderType::OpenRouter => &self.provider.openrouter.model,
            ProviderType::Local => "",
        }
    }

    /// Get max tokens for a specific provider
    pub fn get_max_tokens(&self, provider: ProviderType) -> u32 {
        match provider {
            ProviderType::Anthropic => self.provider.anthropic.max_tokens,
            ProviderType::OpenAI => self.provider.openai.max_tokens,
            ProviderType::OpenRouter => self.provider.openrouter.max_tokens,
            ProviderType::Local => 512, // Default for local
        }
    }

    /// Check if hook is disabled via environment variable
    pub fn is_hook_disabled() -> bool {
        env::var("WHY_HOOK_DISABLE")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    /// Check if a command matches any ignore patterns
    pub fn should_ignore_command(&self, command: &str) -> bool {
        for pattern in &self.hook.ignore_commands.patterns {
            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(command) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if an exit code should be skipped
    pub fn should_skip_exit_code(&self, code: i32) -> bool {
        self.hook.skip_exit_codes.contains(&code)
    }
}

/// Generate default config as TOML string
pub fn generate_default_config() -> String {
    r#"# Why - Error explanation tool configuration
# Place this file at ~/.config/why/config.toml

[hook]
# Always explain errors without prompting (default: false)
auto_explain = false

# Exit codes to skip (0 = success, 130 = Ctrl+C)
skip_exit_codes = [0, 130]

# Minimum stderr lines to trigger explanation
min_stderr_lines = 1

[hook.ignore_commands]
# Regex patterns for commands to ignore (won't be explained)
patterns = [
    "^cd ",      # cd command
    "^ls",       # ls command
    "^echo ",    # echo command
    "^pwd$",     # pwd command
    "^clear$",   # clear command
]

[provider]
# Default provider: local | anthropic | openai | openrouter
default = "local"

[provider.anthropic]
# Model to use with Anthropic
model = "claude-sonnet-4-20250514"
max_tokens = 1024

[provider.openai]
# Model to use with OpenAI
model = "gpt-4o-mini"
max_tokens = 1024

[provider.openrouter]
# Model to use with OpenRouter (format: provider/model)
model = "anthropic/claude-sonnet-4"
max_tokens = 1024

# Environment variable overrides:
# WHY_HOOK_AUTO=1      - Force auto-explain (overrides config)
# WHY_HOOK_DISABLE=1   - Temporarily disable hook explanations
# WHY_PROVIDER=<name>  - Override default provider
# WHY_MODEL=<model>    - Override model for current provider
# ANTHROPIC_API_KEY    - API key for Anthropic
# OPENAI_API_KEY       - API key for OpenAI
# OPENROUTER_API_KEY   - API key for OpenRouter
"#
    .to_string()
}

/// Print the default config to stdout
pub fn print_hook_config() {
    print!("{}", generate_default_config());
}
