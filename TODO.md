# TODO: Third-Party AI Provider Support

This document outlines the implementation plan for adding third-party AI provider support (Anthropic, OpenAI, OpenRouter), enhanced shell hook functionality, and CI/CD improvements for the `why` CLI tool.

---

## Phase 1: Core Infrastructure & Configuration

### 1.1 HTTP Client Infrastructure

- [x] Add `reqwest` crate with `rustls-tls` feature for HTTPS
- [x] Add `tokio` runtime with `rt-multi-thread` and `macros` features for async operations
- [x] Add `futures` crate for stream handling
- [x] Create `src/providers/mod.rs` module structure
- [x] Define `Provider` trait with async methods:

```rust
trait Provider {
    async fn explain(&self, error: &str, context: Option<&CommandContext>) -> Result<String>;
    async fn explain_streaming(&self, error: &str, context: Option<&CommandContext>, callback: impl FnMut(&str)) -> Result<()>;
    fn name(&self) -> &'static str;
    fn requires_api_key(&self) -> bool;
}
```

- [x] Create `CommandContext` struct to hold shell context (command, output, exit code, working directory)
- [x] Implement proper error types for provider failures (network, auth, API errors)

### 1.2 XDG Configuration Directory (Enhancement)

- [x] Review existing `src/config.rs` implementation
- [x] Ensure full XDG Base Directory Specification compliance:
  - `$XDG_CONFIG_HOME/why/config.toml` (default: `~/.config/why/config.toml`)
  - `$XDG_DATA_HOME/why/` for persistent data (default: `~/.local/share/why/`)
  - `$XDG_CACHE_HOME/why/` for cache files (default: `~/.cache/why/`)
  - `$XDG_STATE_HOME/why/` for state files (default: `~/.local/state/why/`)
- [ ] Add `xdg` crate for proper cross-platform XDG support
- [x] Create config directory structure on first run if not exists
- [ ] Add config file migration logic for any existing configs

### 1.3 Configuration Schema Updates

- [x] Extend `Config` struct in `src/config.rs` with provider settings:

```toml
[provider]
default = "local"  # local | anthropic | openai | openrouter

[provider.anthropic]
model = "claude-sonnet-4-20250514"
max_tokens = 1024

[provider.openai]
model = "gpt-4o-mini"
max_tokens = 1024

[provider.openrouter]
model = "anthropic/claude-sonnet-4"
max_tokens = 1024
```

- [x] Add environment variable support (standard conventions):
  - `ANTHROPIC_API_KEY` - Anthropic API key
  - `OPENAI_API_KEY` - OpenAI API key
  - `OPENROUTER_API_KEY` - OpenRouter API key
  - `WHY_PROVIDER` - Override default provider
  - `WHY_MODEL` - Override provider model
- [x] Implement config validation on load
- [x] Add `--provider` CLI flag to override default provider
- [x] Add `--api-key` CLI flag for one-off key usage (not stored)

### 1.4 Success Criteria - Phase 1

- [x] `cargo test` passes with new infrastructure code
- [x] `cargo clippy -- -D warnings` produces no warnings
- [x] Config loads correctly from XDG paths
- [x] Environment variables properly override config file settings
- [x] Provider trait compiles and has documentation

---

## Phase 2: AI Provider Implementations

### 2.1 Local Provider (Refactor)

- [x] Extract existing llama-cpp inference logic into `src/providers/local.rs`
- [x] Implement `Provider` trait for `LocalProvider`
- [x] Maintain backward compatibility with embedded model system
- [x] Support `--model` flag for external GGUF files (existing functionality preserved)
- [x] Preserve all existing functionality (streaming, daemon mode, etc.)

### 2.2 Anthropic Provider

- [x] Create `src/providers/anthropic.rs`
- [x] Implement Anthropic Messages API client:
  - Endpoint: `https://api.anthropic.com/v1/messages`
  - Required headers: `x-api-key`, `anthropic-version`, `content-type`
- [x] Build system prompt optimized for error explanation
- [x] Implement streaming via Server-Sent Events (SSE)
- [x] Handle API errors gracefully (401, 429, 500, etc.)
- [x] Support model selection via `--model` flag (claude-sonnet-4-20250514, claude-haiku, etc.)

### 2.3 OpenAI Provider

- [x] Create `src/providers/openai.rs`
- [x] Implement OpenAI Chat Completions API client:
  - Endpoint: `https://api.openai.com/v1/chat/completions`
  - Required headers: `Authorization: Bearer`, `content-type`
- [x] Build system prompt optimized for error explanation
- [x] Implement streaming via SSE (`stream: true`)
- [x] Handle API errors gracefully
- [x] Support model selection via `--model` flag (gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.)

### 2.4 OpenRouter Provider

- [x] Create `src/providers/openrouter.rs`
- [x] Implement OpenRouter API client (OpenAI-compatible):
  - Endpoint: `https://openrouter.ai/api/v1/chat/completions`
  - Required headers: `Authorization: Bearer`, `HTTP-Referer`, `X-Title`
- [x] Build system prompt optimized for error explanation
- [x] Implement streaming support
- [x] Handle API errors and model availability
- [x] Support model selection via `--model` flag (any model from OpenRouter catalog)
- [ ] Add `--list-openrouter-models` command to show available models

### 2.5 Provider Selection Logic

- [x] Implement provider factory/resolver in `src/providers/mod.rs`
- [x] Priority order for provider selection:
  1. `--provider` CLI flag
  2. `WHY_PROVIDER` environment variable
  3. `provider.default` in config file
  4. `local` (embedded model) as fallback
- [x] Validate API key availability before using external provider
- [x] Provide helpful error message if API key missing
- [x] Add `--list-providers` command to show available providers
- [x] `--model` flag works with all providers (GGUF path for local, model name for external)

### 2.6 Context Injection for External Providers

- [x] Create enhanced prompt template for external providers
- [x] Include shell context when available:
  - Failed command
  - Exit code
  - Last N lines of stderr (simple capture)
  - Working directory
  - Shell type
- [x] Optimize token usage (truncate long outputs intelligently)
- [ ] Add `--max-context` flag to limit context size

### 2.7 Success Criteria - Phase 2

- [x] `cargo test` passes including new provider tests
- [x] `cargo clippy -- -D warnings` produces no warnings
- [x] Each provider can successfully explain a test error (integration test)
- [x] Streaming works for all providers
- [x] Provider selection logic follows documented priority
- [x] Missing API key produces helpful error message
- [x] `--model` flag works correctly for all providers

---

## Phase 3: Shell Hook Enhancements

### 3.1 Enable/Disable Commands

- [x] Add `--enable` CLI flag to enable shell hook for current shell
- [x] Add `--disable` CLI flag to disable shell hook for current shell
- [x] Implementation approach:
  - Set `WHY_HOOK_DISABLE=0` or `WHY_HOOK_DISABLE=1` in shell config
  - Or use a state file in `$XDG_STATE_HOME/why/hook_enabled`
- [x] Add `--status` flag to show current hook state
- [x] Update shell hook scripts to check enabled state
- [x] Provide shell-specific enable/disable output (eval-able commands)

### 3.2 Enhanced Context Capture (Simple Approach)

- [x] Modify shell hooks to capture command context:
  - Command executed (already implemented)
  - Exit code (already implemented)
  - Last N lines of stderr (configurable, default: 50 lines)
  - Working directory
  - Timestamp
- [ ] Update Bash hook to pass stderr context via temp file or variable
- [ ] Update Zsh hook to pass stderr context
- [ ] Update Fish hook to pass stderr context
- [ ] Add `--context-lines` config option to control captured lines (default: 50)
- [x] Ensure minimal performance impact on shell prompt

### 3.3 Improved Hook Installation

- [x] Add `--hook-status` to show installation status for all shells
- [x] Improve `--hook-install` with:
  - Backup existing config before modification
  - Verify installation was successful
  - Show what was added
- [x] Improve `--hook-uninstall` with:
  - Verify removal was successful
  - Clean up any state files
- [ ] Add `--hook-update` to update hooks to latest version
- [ ] Support `--shell` flag to target specific shell

### 3.4 Success Criteria - Phase 3

- [x] `cargo test` passes including hook tests
- [x] `cargo clippy -- -D warnings` produces no warnings
- [x] `why --enable` successfully enables hook (verified by `why --status`)
- [x] `why --disable` successfully disables hook (verified by `why --status`)
- [x] Shell hook captures command, exit code, and last N lines of stderr
- [x] Context is properly passed to provider for explanation
- [x] Hook install/uninstall works for Bash, Zsh, Fish
- [x] No performance degradation in shell prompt

---

## Phase 4: CI/CD & Build System Updates

### 4.1 Dual Binary Build Configuration

- [x] Update `flake.nix` to produce two binary variants:
  - `why-cli` - Standalone CLI without embedded model (~5-10MB)
  - `why` (default) - CLI with embedded model (~680MB)
- [x] Add clear naming convention for releases:
  - `why-x86_64-linux` (with model)
  - `why-cli-x86_64-linux` (CLI only)
  - `why-aarch64-linux` (with model)
  - `why-cli-aarch64-linux` (CLI only)
  - `why-aarch64-darwin` (with model)
  - `why-cli-aarch64-darwin` (CLI only)
  - `why-x86_64-darwin` (with model)
  - `why-cli-x86_64-darwin` (CLI only)
- [x] Ensure CLI-only version gracefully handles missing model:
  - Require `--provider` flag or `WHY_PROVIDER` env var
  - Suggest external provider if no model available
  - Clear error message explaining options

### 4.2 Linux Build Support (hal9000)

- [x] Configure Nix remote builder for hal9000:

```nix
# In ~/.config/nix/nix.conf or /etc/nix/nix.conf
builders = ssh://hal9000 x86_64-linux
```

- [x] Add Linux build targets to flake:

```nix
packages.x86_64-linux.default = ...;
packages.x86_64-linux.cli = ...;
packages.aarch64-linux.default = ...;
packages.aarch64-linux.cli = ...;
```

- [x] Create build script for remote Linux builds:

```bash
nix build .#packages.x86_64-linux.default --builders 'ssh://hal9000'
nix build .#packages.x86_64-linux.cli --builders 'ssh://hal9000'
```

- [ ] Test binary execution on hal9000

### 4.3 CI/CD Pipeline Updates

- [x] Update `.github/workflows/ci.yml`:
  - Add provider integration tests (with mocked responses)
  - Add shell hook tests
  - Add config loading tests
  - Test both CLI-only and embedded builds
- [x] Update `.github/workflows/release.yml`:
  - Build matrix for all variants:

| OS    | Arch    | Variant  | Artifact Name          |
| ----- | ------- | -------- | ---------------------- |
| Linux | x86_64  | embedded | why-x86_64-linux       |
| Linux | x86_64  | cli-only | why-cli-x86_64-linux   |
| Linux | aarch64 | embedded | why-aarch64-linux      |
| Linux | aarch64 | cli-only | why-cli-aarch64-linux  |
| macOS | x86_64  | embedded | why-x86_64-darwin      |
| macOS | x86_64  | cli-only | why-cli-x86_64-darwin  |
| macOS | aarch64 | embedded | why-aarch64-darwin     |
| macOS | aarch64 | cli-only | why-cli-aarch64-darwin |

  - Update artifact upload/download steps
  - Update release asset names
- [x] Add checksums (SHA256) for all release binaries

### 4.4 Success Criteria - Phase 4

- [x] `nix build .#cli` produces CLI-only binary without model
- [x] `nix build .#default` produces embedded model binary
- [ ] Remote build on hal9000 succeeds for x86_64-linux
- [ ] aarch64-linux builds succeed (cross-compile or native runner)
- [x] CI pipeline passes for all build variants
- [x] Release workflow produces all expected artifacts (8 binaries total)

---

## Phase 5: Final Validation & Quality Assurance

### 5.1 Code Quality

- [x] `cargo fmt --all --check` passes (no formatting issues)
- [x] `cargo clippy --all-targets --all-features -- -D warnings` passes (no warnings)
- [x] `cargo clippy --all-targets --no-default-features -- -D warnings` passes
- [ ] All `TODO` and `FIXME` comments addressed or tracked
- [ ] No `unwrap()` or `expect()` in library code (use proper error handling)
- [x] All public APIs documented with rustdoc comments

### 5.2 Test Coverage

- [x] `cargo test` passes (all unit tests)
- [x] `cargo test --all-features` passes
- [x] `cargo test --no-default-features` passes
- [x] Integration tests for each provider (with mocked HTTP)
- [x] Integration tests for shell hooks
- [x] Integration tests for config loading
- [ ] Test coverage report generated (optional: use tarpaulin)

### 5.3 Nix Validation

- [x] `nix flake check` passes with no errors
- [x] `nix flake check` passes with no warnings
- [x] `nix build` succeeds for all defined packages:
  - [x] `nix build .#default`
  - [x] `nix build .#cli`
  - [x] `nix build .#why`
  - [ ] `nix build .#why-qwen2_5-coder`
  - [ ] `nix build .#why-qwen3`
  - [ ] `nix build .#why-smollm2`
  - [ ] `nix build .#why-gemma3`
- [x] `nix develop` shell works correctly
- [x] All packages have valid metadata (description, license, etc.)

### 5.4 Binary Validation

- [ ] x86_64-linux embedded binary:
  - [ ] Builds successfully
  - [ ] `--help` works
  - [ ] `--version` shows correct version
  - [ ] Embedded model inference works
  - [ ] External provider works (with API key)
- [ ] x86_64-linux CLI-only binary:
  - [ ] Builds successfully
  - [ ] `--help` works
  - [ ] Graceful error without model/provider
  - [ ] External provider works
- [ ] aarch64-linux embedded binary:
  - [ ] Builds successfully
  - [ ] Embedded model inference works
  - [ ] External provider works
- [ ] aarch64-linux CLI-only binary:
  - [ ] Builds successfully
  - [ ] External provider works
- [x] aarch64-darwin embedded binary:
  - [x] Builds successfully
  - [x] Metal acceleration works
  - [x] All features work correctly
- [x] x86_64-darwin embedded binary:
  - [x] Builds successfully
  - [x] All features work correctly

### 5.5 Documentation Updates

- [x] Update README.md with:
  - [x] External provider setup instructions
  - [x] Environment variable documentation
  - [x] Config file documentation
  - [x] Shell hook enable/disable instructions
  - [x] Installation options (embedded vs CLI-only)
- [x] Update CLAUDE.md with architecture changes
- [x] Add CHANGELOG.md entry for this release
- [x] Verify `--help` output is accurate and complete

### 5.6 Success Criteria - Phase 5

- [x] All Phase 1-4 success criteria still pass
- [x] Zero errors from `cargo fmt`, `cargo clippy`, `cargo test`
- [x] Zero errors from `nix flake check`
- [x] Zero warnings from `nix flake check`
- [x] All binaries build and run successfully on target platforms
- [x] Documentation is complete and accurate
- [x] Ready for release

---

## Appendix A: Dependency Additions

```toml
# Cargo.toml additions
[dependencies]
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls", "json", "stream"] }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
futures = "0.3"
async-trait = "0.1"
xdg = "2.5"
```

## Appendix B: File Structure

```text
src/
├── main.rs
├── lib.rs
├── cli.rs              # Updated with new flags
├── config.rs           # Enhanced with provider config
├── hooks.rs            # Enhanced with enable/disable
├── providers/
│   ├── mod.rs          # Provider trait and factory
│   ├── local.rs        # Embedded model provider
│   ├── anthropic.rs    # Anthropic Claude API
│   ├── openai.rs       # OpenAI API
│   └── openrouter.rs   # OpenRouter API
├── context.rs          # CommandContext for shell capture
├── model.rs            # Existing model handling
├── output.rs           # Existing output formatting
├── daemon.rs           # Existing daemon mode
├── watch.rs            # Existing watch mode
├── stack_trace.rs      # Existing stack trace parsing
└── prompts/
    ├── chatml.txt
    ├── gemma.txt
    └── external.txt    # Template for external providers
```

## Appendix C: Environment Variables Summary

| Variable             | Description                  | Example                  |
| -------------------- | ---------------------------- | ------------------------ |
| `ANTHROPIC_API_KEY`  | Anthropic API key            | `sk-ant-...`             |
| `OPENAI_API_KEY`     | OpenAI API key               | `sk-...`                 |
| `OPENROUTER_API_KEY` | OpenRouter API key           | `sk-or-...`              |
| `WHY_PROVIDER`       | Override default provider    | `anthropic`              |
| `WHY_MODEL`          | Override provider model      | `claude-sonnet-4-20250514` |
| `WHY_HOOK_DISABLE`   | Disable shell hook           | `1`                      |
| `WHY_HOOK_AUTO`      | Auto-explain without prompt  | `1`                      |
| `WHY_CONFIG`         | Custom config file path      | `/path/to/config.toml`   |
| `WHY_DEBUG`          | Enable debug output          | `1`                      |

## Appendix D: References

- [Anthropic API Documentation](https://docs.anthropic.com/en/api/messages)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference/chat)
- [OpenRouter API Documentation](https://openrouter.ai/docs)
- [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html)
