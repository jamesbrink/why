# why

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Model: Qwen2.5-Coder](https://img.shields.io/badge/Model-Qwen2.5--Coder-green.svg)](https://github.com/QwenLM/Qwen2.5-Coder)

Simple CLI to explain errors using an embedded LLM, so you don't lose your shit.

No API keys. No internet. No patience required.

![why demo](why.gif)

## Quick Start

```bash
# Run directly from GitHub (no install needed)
nix run github:jamesbrink/why -- "segmentation fault"
```

## Usage

```bash
# Direct error
why "segmentation fault"

# Pipe your failures (use 2>&1 to capture stderr)
cargo build 2>&1 | why
python script.py 2>&1 | why

# For the robots
why --json "null pointer exception"
```

See the [examples/](examples/) directory for sample scripts in various languages that produce common errors.

## Features

- **Single binary** - Model embedded right in the executable. One file, zero dependencies.
- **Offline** - Works on airplanes, in bunkers, or when your ISP decides to take a nap.
- **Fast** - Local inference with Metal/CUDA acceleration. No round trips to the cloud.
- **Structured output** - Clean, colored terminal output or JSON for scripting.
- **Shell completions** - Tab completion for bash, zsh, fish, and friends.

## Installation

### Pre-built Binary

Grab the embedded binary from releases (includes the model, ~400MB).

### Build from Source

Requires [Nix](https://nixos.org/) with flakes enabled and [git-lfs](https://git-lfs.com/) for the model.

```bash
git clone https://github.com/jamesbrink/why.git
cd why

# Build embedded binary (~400MB with model)
nix build
./result/bin/why "segmentation fault"
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

## Nix Build Targets

```bash
# Build embedded binary (default) - includes model (~400MB)
nix build

# Build CLI only (no embedded model)
nix build .#why

# Run directly
nix run . -- "segmentation fault"
```

## Development

```bash
nix develop

cargo build              # Build CLI
cargo test               # Run tests
cargo clippy             # Lint
cargo tarpaulin          # Coverage report
```

## How It Works

1. You give it an error message
2. A tiny LLM (Qwen2.5-Coder 0.5B) thinks about it locally
3. You get a summary, explanation, and suggestion
4. You feel slightly less like throwing your laptop

The model is embedded directly in the binary using a custom trailer format. On first run, it extracts to a temp file for inference. Subsequent runs skip extraction.

## License

This project is licensed under the [MIT License](LICENSE).

### Model

Uses [Qwen2.5-Coder 0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF) (Q4_K_M quantization, ~398MB) tracked via git-lfs.

This project uses [Qwen2.5-Coder](https://github.com/QwenLM/Qwen2.5-Coder) by the Qwen Team (Alibaba Group), licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

When distributing binaries with an embedded model, both licenses apply:

- The `why` CLI code: MIT License
- The Qwen2.5-Coder model: Apache License 2.0
