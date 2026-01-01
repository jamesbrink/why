# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **External AI Providers**: Support for Anthropic, OpenAI, and OpenRouter APIs
  - `--provider` flag to select provider (local, anthropic, openai, openrouter)
  - `--api-key` flag for one-off API key usage
  - `--list-providers` command to show available providers
  - Environment variable support: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`
  - `WHY_PROVIDER` and `WHY_MODEL` environment variables for defaults
  - Streaming support for all external providers via SSE

- **Shell Hook Controls**: Enable/disable hook functionality
  - `--enable` flag to enable shell hook
  - `--disable` flag to disable shell hook
  - `--status` flag to show current hook status
  - State persisted in `$XDG_STATE_HOME/why/hook_enabled`
  - `WHY_HOOK_DISABLE=1` environment variable for temporary disable

- **Configuration File**: XDG-compliant config at `~/.config/why/config.toml`
  - Provider defaults and model selection
  - Per-provider settings (model, max_tokens)

- **CLI-Only Binary**: Separate build without embedded model (~5MB)
  - `nix build .#cli` for CLI-only build
  - Requires `--provider` or `--model` flag when no embedded model
  - Release binaries: `why-cli-{arch}-{os}`

- **aarch64-linux Support**: ARM64 Linux builds via QEMU emulation in CI

### Changed

- Updated CI/CD to build 8 binary variants (4 platforms × 2 variants)
- Release workflow now includes SHA256 checksums for all binaries
- Provider architecture refactored into `src/providers/` module

## [0.1.0] - 2024-12-XX

### Added

- Initial release
- Embedded LLM inference using Qwen2.5-Coder 0.5B
- Shell hook integration for Bash, Zsh, and Fish
- Streaming output with `--stream` flag
- Watch mode for log files and commands
- Daemon mode for faster subsequent queries
- JSON output for scripting
- Multiple model variants: Qwen2.5-Coder, Qwen3, SmolLM2, Gemma3
- GPU acceleration: Metal (macOS), Vulkan (Linux)
- Stack trace parsing for Python, Rust, JavaScript, Go, Java, C++
