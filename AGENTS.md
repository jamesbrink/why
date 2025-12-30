# Repository Guidelines

## Project Structure & Module Organization
- `src/main.rs` contains the CLI, prompt handling, inference wiring, and response parsing.
- `src/prompt.txt` is the ChatML-style prompt template.
- `scripts/embed.sh` embeds the GGUF model into the release binary.
- `examples/` holds sample inputs (e.g., Python error fixtures).
- The GGUF model is fetched from HuggingFace during the nix build.

## Build, Test, and Development Commands
- `nix develop` sets up the Rust toolchain and llama-cpp dependencies.
- `cargo build` builds the CLI for local development.
- `cargo run -- "segmentation fault"` runs the CLI with a sample error.
- `cargo test` runs unit tests embedded in `src/main.rs`.
- `cargo clippy` runs lint checks.
- `cargo tarpaulin` generates coverage reports.
- `nix build` produces the embedded binary (~680MB with model).
- `nix build .#why` builds the CLI only (no embedded model).

## Coding Style & Naming Conventions
- Rust 2021 edition; follow `rustfmt` defaults (4-space indentation, trailing commas).
- Use `snake_case` for functions and modules, `PascalCase` for types.
- Keep CLI flags and outputs consistent with existing naming in `src/main.rs`.
- Prefer small, focused helper functions with clear names.

## Testing Guidelines
- Tests live in `src/main.rs` under `#[cfg(test)]`.
- Name tests `test_*` and keep fixtures inline for clarity.
- Run `cargo test` before submitting changes that affect parsing or CLI behavior.

## Commit & Pull Request Guidelines
- Use short, sentence-case, imperative summaries (e.g., "Add …", "Fix …").
- Keep commits focused; avoid mixing refactors and behavior changes.
- PRs should include a brief problem/solution summary, test commands run, and any CLI output changes. Screenshots are optional unless output formatting changes.

## Model & Licensing Notes
- The embedded model is licensed separately (Apache 2.0). Keep `LICENSE` and model attribution intact when distributing binaries.
- The model is automatically fetched from HuggingFace during `nix build`.
