# why

[![CI](https://github.com/jamesbrink/why/actions/workflows/ci.yml/badge.svg)](https://github.com/jamesbrink/why/actions/workflows/ci.yml)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Nix Flake](https://img.shields.io/badge/Nix-Flake-5277C3?logo=nixos)](https://nixos.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model: Qwen2.5-Coder](https://img.shields.io/badge/Model-Qwen2.5--Coder-green.svg)](https://github.com/QwenLM/Qwen2.5-Coder)

Simple CLI to explain errors using an embedded LLM, so you don't lose your shit.

Works offline with an embedded model, or connect to cloud providers (Anthropic, OpenAI, OpenRouter) when you want more power.

<p align="center">
  <img src="why.gif" alt="why demo">
</p>

## Quick Start

```bash
# Install it (one command, no wizardry required)
curl -sSfL https://raw.githubusercontent.com/jamesbrink/why/main/install.sh | sh

# Now yell at your errors
why "segmentation fault"
```

Already have Nix? You can skip the install entirely:
```bash
nix run github:jamesbrink/why -- "segmentation fault"
```

## Usage

<p align="center">
  <img src="demo.gif" alt="why demo">
</p>

```bash
# Direct error
why "segmentation fault"

# Pipe your failures (use 2>&1 to capture stderr)
cargo build 2>&1 | why
python script.py 2>&1 | why

# Stream tokens as they generate (fancy)
why --stream "null pointer exception"

# Watch a log file for errors
why --watch /var/log/app.log

# Watch a command's output
why --watch "npm run dev"

# Capture and explain failures automatically
why --capture -- cargo build

# For the robots
why --json "null pointer exception"

# Use external model (CLI-only build)
why --model /path/to/model.gguf "error message"

# Override template for non-standard models
why --template gemma --model /path/to/gemma.gguf "error"

# Use cloud providers (see External Providers section)
why --provider anthropic "segmentation fault"
why --provider openai "null pointer exception"
```

See the [examples/](examples/) directory for sample scripts in various languages that produce common errors.

## Features

- **Single binary** - Model embedded right in the executable. One file, zero dependencies.
- **Offline** - Works on airplanes, in bunkers, or when your ISP decides to take a nap.
- **Cloud providers** - Connect to Anthropic, OpenAI, or OpenRouter for more powerful explanations.
- **Fast** - Local inference with Metal (macOS) or Vulkan (Linux). CPU-only works everywhere.
- **Streaming** - Watch tokens appear in real-time with `--stream`. Feels like magic, but it's just inference.
- **Watch mode** - Monitor log files or commands with `--watch`. Errors explained as they happen.
- **Stack trace parsing** - Understands Python, Rust, JavaScript, Go, Java, and C++ stack traces.
- **Shell integration** - Auto-explain failed commands. Your shell becomes slightly less hostile.
- **Daemon mode** - Keep the model loaded with `why daemon start`. Sub-second responses.
- **Structured output** - Clean, colored terminal output or JSON for scripting.
- **Shell completions** - Tab completion for bash, zsh, fish, and friends.

## Installation

### Quick Install (Recommended)

```bash
curl -sSfL https://raw.githubusercontent.com/jamesbrink/why/main/install.sh | sh
```

This downloads the latest release binary (~680MB, includes the model) and installs it to `~/.local/bin` or `/usr/local/bin`.

**Options:**

```bash
# Install to a specific directory
WHY_INSTALL_DIR=/opt/bin curl -sSfL https://raw.githubusercontent.com/jamesbrink/why/main/install.sh | sh

# Install a specific version
WHY_VERSION=v0.1.0 curl -sSfL https://raw.githubusercontent.com/jamesbrink/why/main/install.sh | sh
```

### Pre-built Binary (Manual)

Download the binary for your platform from [Releases](https://github.com/jamesbrink/why/releases):

**Embedded (with model, ~680MB):**
- `why-x86_64-linux` - Linux (x86_64)
- `why-aarch64-linux` - Linux (ARM64)
- `why-aarch64-darwin` - macOS (Apple Silicon)
- `why-x86_64-darwin` - macOS (Intel)

**CLI-only (no model, ~5MB):**
- `why-cli-x86_64-linux` - Linux (x86_64)
- `why-cli-aarch64-linux` - Linux (ARM64)
- `why-cli-aarch64-darwin` - macOS (Apple Silicon)
- `why-cli-x86_64-darwin` - macOS (Intel)

The CLI-only build requires either a local model file (`--model`) or an external provider (`--provider`).

```bash
# Example for Linux x86_64 (embedded)
curl -L -o why https://github.com/jamesbrink/why/releases/latest/download/why-x86_64-linux
chmod +x why
sudo mv why /usr/local/bin/

# Example for CLI-only build with external provider
curl -L -o why https://github.com/jamesbrink/why/releases/latest/download/why-cli-x86_64-linux
chmod +x why
export ANTHROPIC_API_KEY="your-key"
./why --provider anthropic "error message"
```

### Build from Source

Requires [Nix](https://nixos.org/) with flakes enabled.

On Linux, the flake enables Vulkan GPU acceleration by default (works on NVIDIA/AMD/Intel). CPU-only still works without Vulkan. On macOS, it uses Metal.

```bash
git clone https://github.com/jamesbrink/why.git
cd why

# Build embedded binary (~680MB with model)
nix build
./result/bin/why "segmentation fault"
```

## Uninstall

```bash
# Remove the binary
rm $(which why)

# Or if you used the installer with default paths
rm ~/.local/bin/why
# or
sudo rm /usr/local/bin/why
```

## Shell Completions

Because typing is hard.

```bash
# Bash
why --completions bash > ~/.local/share/bash-completion/completions/why

# Zsh
why --completions zsh > ~/.zfunc/_why

# Fish
why --completions fish > ~/.config/fish/completions/why.fish
```

## Shell Integration

Make your shell explain failures automatically. No more copy-pasting error messages like a peasant.

```bash
# Generate hook for your shell
why --hook bash >> ~/.bashrc
why --hook zsh >> ~/.zshrc
why --hook fish >> ~/.config/fish/config.fish

# Or install directly
why --hook-install bash

# Now when commands fail, why explains them
$ npm run build
# (command fails)
# why automatically explains what went wrong

# Enable/disable hook without uninstalling
why --enable    # Enable auto-explain
why --disable   # Disable auto-explain
why --status    # Show current hook status
```

Wrap commands to capture and explain failures:

```bash
# Run a command, explain if it fails
why --capture -- cargo build

# Prompt before explaining
why --capture --confirm -- make test

# Capture both stdout and stderr
why --capture-all -- ./my-script.sh
```

## External Providers

Need more brain power than a 0.5B model can muster? Phone a friend in the cloud:

```bash
# Set your API key (one of these)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."

# Use a specific provider
why --provider anthropic "segmentation fault"
why --provider openai "null pointer exception"
why --provider openrouter "memory leak"

# List available providers
why --list-providers

# Override the model
why --provider anthropic --model claude-haiku-4-20250514 "error"
why --provider openai --model gpt-4o "complex error"
```

### Supported Providers

| Provider | Env Variable | Default Model |
| -------- | ------------ | ------------- |
| `local` | - | Embedded Qwen2.5-Coder |
| `anthropic` | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 |
| `openai` | `OPENAI_API_KEY` | gpt-4o-mini |
| `openrouter` | `OPENROUTER_API_KEY` | anthropic/claude-sonnet-4 |

### Environment Variables

| Variable | Description |
| -------- | ----------- |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `WHY_PROVIDER` | Override default provider |
| `WHY_MODEL` | Override provider model |
| `WHY_HOOK_DISABLE` | Disable shell hook (`1` to disable) |
| `WHY_DEBUG` | Enable debug output |

### Configuration File

Create `~/.config/why/config.toml`:

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

## Daemon Mode

Cold starts are for chumps. Keep the model loaded and get sub-second responses.

```bash
# Start the daemon
why daemon start

# Now queries are fast
why "segmentation fault"  # Uses daemon if running

# Check status
why daemon status

# Stop when you're done
why daemon stop

# Run in foreground (for debugging)
why daemon start --foreground

# Install as a system service
why daemon install-service  # systemd on Linux, launchd on macOS
```

The daemon auto-shuts down after 30 minutes of inactivity. Configure with `--idle-timeout`.

## Nix Build Targets

```bash
nix build               # Default (Qwen2.5-Coder, ~680MB)
nix build .#cli         # CLI only, no model (~4.5MB)
nix build .#why-qwen3   # Qwen3 0.6B (~644MB)
nix build .#why-gemma3  # Gemma 3 270M (~297MB)
nix build .#why-smollm2 # SmolLM2 135M (~149MB)

# Use CLI with external model
nix run .#cli -- --model /path/to/model.gguf "error"
```

## NixOS / Home Manager

Add the overlay to your flake:

```nix
{
  inputs.why.url = "github:jamesbrink/why";

  outputs = { nixpkgs, why, ... }: {
    # NixOS
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      modules = [{
        nixpkgs.overlays = [ why.overlays.default ];
        environment.systemPackages = [ pkgs.why ];
      }];
    };

    # Home Manager (standalone)
    homeConfigurations.myuser = home-manager.lib.homeManagerConfiguration {
      modules = [{
        nixpkgs.overlays = [ why.overlays.default ];
        home.packages = [ pkgs.why ];
      }];
    };
  };
}
```

## Development

```bash
nix develop

build                    # Build + embed model (auto-downloads if needed)
cargo build              # Build CLI only
cargo test               # Run tests
cargo clippy             # Lint
cargo tarpaulin          # Coverage report
```

### Local Testing

After building with `nix build`, test the binary without installing:

```bash
# Test the binary
./result/bin/why "segmentation fault"

# Source completions for your current shell (zsh)
source <(./result/bin/why --completions zsh)

# Source shell hook for local testing (rewrites 'why' to use local binary)
source <(./result/bin/why --hook zsh | sed 's|why --exit|'$(pwd)'/result/bin/why --exit|')

# Same for bash
source <(./result/bin/why --completions bash)
source <(./result/bin/why --hook bash | sed 's|why --exit|'$(pwd)'/result/bin/why --exit|')

# Now failed commands will auto-explain using your local build
$ cat /nonexistent
# (fails, why explains it)
```

### Manual Model Download

The `build` command auto-downloads the model, but you can also download it manually:

```bash
curl -L -o qwen2.5-coder-0.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-0.5b-instruct-q8_0.gguf
```

## How It Works

1. You give it an error message
2. A tiny LLM (Qwen2.5-Coder 0.5B) thinks about it locally (or a cloud model if you're fancy)
3. You get a summary, explanation, and suggestion
4. You feel slightly less like throwing your laptop

The model is embedded directly in the binary using a custom trailer format. On first run, it extracts to a temp file for inference. Subsequent runs skip extraction. Or skip all that and just use `--provider anthropic` like someone with an API budget.

## License

This project is licensed under the [MIT License](LICENSE).

### Models

Available model variants:

| Model | Size | License | Note |
| ----- | ---- | ------- | ---- |
| [Qwen2.5-Coder 0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF) | ~530MB | Apache 2.0 | Default |
| [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) | ~639MB | Apache 2.0 | Newest Qwen |
| [Gemma 3 270M](https://huggingface.co/unsloth/gemma-3-270m-it-GGUF) | ~292MB | [Gemma Terms](https://ai.google.dev/gemma/terms) | Google |
| [SmolLM2 135M](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF) | ~145MB | Apache 2.0 | Smallest |

When distributing binaries with an embedded model, both the MIT (CLI) and model licenses apply. Note that Gemma models use Google's custom terms, not Apache 2.0.
