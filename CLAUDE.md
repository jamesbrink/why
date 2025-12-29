# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`why` is a CLI tool that explains programming errors using a locally-embedded LLM (Qwen2.5-Coder 0.5B). It runs entirely offline with the model baked into the binary.

## Build Commands

```bash
# Enter development environment (provides rust toolchain, llama-cpp, etc.)
nix develop

# Build and run during development
cargo build
cargo run -- "segmentation fault"

# Run tests
cargo test

# Run single test
cargo test test_name

# Lint
cargo clippy

# Build embedded binary with nix (includes model, ~400MB)
nix build

# Build CLI only (no model)
nix build .#why

# Manual embed for development
cargo build --release
./scripts/embed.sh target/release/why qwen2.5-coder-0.5b.gguf why-embedded
```

## Architecture

### Binary Embedding Format

The model is embedded in the binary using a trailer format:
```
[original binary][model bytes][8-byte magic "WHYMODEL"][8-byte offset][8-byte size]
```

At runtime, `find_embedded_model()` reads the trailer to locate and extract the model to a temp file for inference.

### Key Components (src/main.rs)

- **CLI parsing**: Uses clap with `--json`, `--debug`, `--completions` flags
- **Input handling**: Accepts error as args or via stdin pipe
- **Prompt building**: Uses ChatML format template from `src/prompt.txt`
- **Inference**: llama-cpp-2 bindings, 2048 context, greedy sampling
- **Response parsing**: Extracts SUMMARY/EXPLANATION/SUGGESTION sections
- **Echo detection**: Catches when model repeats input (indicates non-error input)

### Response Detection Logic

1. Empty response → "No error detected"
2. Starts with "NO_ERROR" → "No error detected"
3. Echo of input → "No error detected"
4. Has SUMMARY/EXPLANATION/SUGGESTION → Parse and display
5. No parseable content → "Could not analyze input"

## Model

The GGUF model (`qwen2.5-coder-0.5b.gguf`, ~398MB) is tracked via git-lfs. Ensure git-lfs is installed before cloning.
