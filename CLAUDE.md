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

# Build + embed model (auto-downloads model if needed)
build

# Build embedded binary with nix (includes model, ~680MB)
nix build

# Build CLI only (no model)
nix build .#why

# Run evaluation suite
python3 scripts/eval.py          # All cases
python3 scripts/eval.py -d       # Detailed output with markdown
python3 scripts/eval.py -f rust  # Filter by language
```

## Architecture

### Binary Embedding Format

The model is embedded in the binary using a trailer format (25 bytes):
```
[original binary][model bytes][8-byte magic "WHYMODEL"][8-byte offset][8-byte size][1-byte family]
```

Family byte: 0=qwen, 1=gemma, 2=smollm

At runtime, `find_embedded_model()` reads the trailer to locate and extract the model to a temp file for inference.

### Prompt Templates

Different model families require different prompt formats. Templates are in `src/prompts/`:
- **chatml.txt** - Used by Qwen and SmolLM models (`<|im_start|>`, `<|im_end|>`)
- **gemma.txt** - Used by Gemma models (`<start_of_turn>`, `<end_of_turn>`)

The template is selected based on: CLI `--template` flag > embedded family byte > auto-detection from filename.

### Key Components (src/main.rs)

- **CLI parsing**: Uses clap with `--json`, `--debug`, `--stats`, `--model`, `--template`, `--completions` flags
- **Input handling**: Accepts error as args or via stdin pipe
- **Model family detection**: Auto-detects from embedded trailer or filename, can be overridden with `--template`
- **Prompt building**: Selects appropriate template based on model family
- **Inference**: llama-cpp-2 bindings, 2048 context, temp=0.7 sampling
- **Response parsing**: Extracts SUMMARY/EXPLANATION/SUGGESTION sections (handles both uppercase and markdown `**Bold:**` formats)
- **Echo detection**: Catches when model repeats input (indicates non-error input)

### Response Detection Logic

1. Empty response → "No error detected"
2. Starts with "NO_ERROR" → "No error detected"
3. Echo of input → "No error detected"
4. Has SUMMARY/EXPLANATION/SUGGESTION → Parse and display
5. No parseable content → "Could not analyze input"

### GPU Acceleration

Feature flags in Cargo.toml control GPU backend:
- `metal` - macOS (Apple Silicon/Intel)
- `vulkan` - Linux (AMD/Intel/NVIDIA)
- `cuda` - NVIDIA (optional)

The nix flake auto-selects: Metal on Darwin, Vulkan on Linux.

## Dependencies

Always use the most recent versions of dependencies. Check latest versions with:
```bash
cargo search <crate-name> --limit 1
```

## Model

The GGUF model (`qwen2.5-coder-0.5b-instruct-q8_0.gguf`, ~676MB) is fetched from HuggingFace during the nix build. For local development, the `build` script auto-downloads it.
