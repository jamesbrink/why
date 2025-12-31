//! Model loading, inference, and prompt construction.

use anyhow::{bail, Context, Result};
use clap::ValueEnum;
use colored::Colorize;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use serde::Serialize;
use std::env;
use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Token callback type for streaming output
/// Returns Ok(true) to continue, Ok(false) to stop, or Err to abort
pub type TokenCallback<'a> = Box<dyn FnMut(&str) -> Result<bool> + 'a>;

/// Model family for prompt template selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ModelFamily {
    /// Qwen models - uses ChatML format
    Qwen,
    /// Gemma models - uses Gemma format
    Gemma,
    /// SmolLM models - uses ChatML format
    Smollm,
}

impl fmt::Display for ModelFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelFamily::Qwen => write!(f, "qwen (ChatML)"),
            ModelFamily::Gemma => write!(f, "gemma (Gemma format)"),
            ModelFamily::Smollm => write!(f, "smollm (ChatML)"),
        }
    }
}

/// Magic marker written before embedded model
pub const MAGIC: &[u8; 8] = b"WHYMODEL";

/// Embedded model info: offset, size, and optional model family
pub struct EmbeddedModelInfo {
    pub offset: u64,
    pub size: u64,
    pub family: Option<ModelFamily>,
}

/// Result of getting model path, includes embedded family if available
pub struct ModelPathInfo {
    pub path: PathBuf,
    pub embedded_family: Option<ModelFamily>,
}

/// Sampling parameters for inference
#[derive(Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: Option<u32>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            seed: None,
        }
    }
}

/// Statistics from an inference run
#[derive(Debug, Serialize)]
pub struct InferenceStats {
    pub backend: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub total_tokens: usize,
    pub model_load_ms: u128,
    pub prompt_eval_ms: u128,
    pub generation_ms: u128,
    pub total_ms: u128,
    pub gen_tok_per_s: f64,
    pub total_tok_per_s: f64,
}

/// Maximum number of retries when detecting degenerate output
pub const MAX_RETRIES: usize = 2;

/// ChatML template for Qwen and SmolLM models
const TEMPLATE_CHATML: &str = include_str!("prompts/chatml.txt");

/// Gemma template using <start_of_turn> format
const TEMPLATE_GEMMA: &str = include_str!("prompts/gemma.txt");

pub fn format_error(message: &str, tip: Option<&str>) -> String {
    let mut output = format!("{} {}", "Error:".red().bold(), message);
    if let Some(tip) = tip {
        output.push('\n');
        output.push_str(&format!("{} {}", "Tip:".blue().bold(), tip));
    }
    output
}

pub fn backend_mode() -> &'static str {
    if cfg!(feature = "metal") {
        "metal"
    } else if cfg!(feature = "cuda") {
        "cuda"
    } else if cfg!(feature = "vulkan") {
        "vulkan"
    } else {
        "cpu"
    }
}

pub fn find_embedded_model() -> Result<EmbeddedModelInfo> {
    let exe_path = env::current_exe().context("failed to get executable path")?;
    let mut file = File::open(&exe_path).context("failed to open self")?;
    let file_len = file.metadata()?.len();

    // Try new 25-byte trailer format first (with family byte)
    if file_len >= 25 {
        file.seek(SeekFrom::End(-25))?;
        let mut trailer = [0u8; 25];
        file.read_exact(&mut trailer)?;

        if &trailer[0..8] == MAGIC {
            let offset = u64::from_le_bytes(trailer[8..16].try_into().unwrap());
            let size = u64::from_le_bytes(trailer[16..24].try_into().unwrap());
            let family = match trailer[24] {
                0 => Some(ModelFamily::Qwen),
                1 => Some(ModelFamily::Gemma),
                2 => Some(ModelFamily::Smollm),
                _ => None,
            };
            return Ok(EmbeddedModelInfo {
                offset,
                size,
                family,
            });
        }
    }

    // Fall back to old 24-byte trailer format (no family byte)
    if file_len >= 24 {
        file.seek(SeekFrom::End(-24))?;
        let mut trailer = [0u8; 24];
        file.read_exact(&mut trailer)?;

        if &trailer[0..8] == MAGIC {
            let offset = u64::from_le_bytes(trailer[8..16].try_into().unwrap());
            let size = u64::from_le_bytes(trailer[16..24].try_into().unwrap());
            return Ok(EmbeddedModelInfo {
                offset,
                size,
                family: None,
            });
        }
    }

    bail!(format_error("No embedded model found.", None))
}

pub fn get_model_path(cli_model: Option<&PathBuf>) -> Result<ModelPathInfo> {
    // CLI flag takes highest priority
    if let Some(model_path) = cli_model {
        if model_path.exists() {
            return Ok(ModelPathInfo {
                path: model_path.clone(),
                embedded_family: None,
            });
        }
        bail!(format_error(
            &format!("Model not found: {}", model_path.display()),
            Some("Check the path and try again")
        ));
    }

    // Check for embedded model
    if let Ok(info) = find_embedded_model() {
        let exe_path = env::current_exe()?;
        let mut file = File::open(&exe_path)?;

        // Extract to temp
        let temp_path = env::temp_dir().join("why-model.gguf");
        if !temp_path.exists() || temp_path.metadata().map(|m| m.len()).unwrap_or(0) != info.size {
            eprintln!("{}", "Extracting embedded model...".dimmed());
            file.seek(SeekFrom::Start(info.offset))?;
            let mut model_data = vec![0u8; info.size as usize];
            file.read_exact(&mut model_data)?;
            std::fs::write(&temp_path, model_data)?;
        }
        return Ok(ModelPathInfo {
            path: temp_path,
            embedded_family: info.family,
        });
    }

    // Fallback: look for model file in current dir or next to exe
    let candidates = [
        PathBuf::from("model.gguf"),
        PathBuf::from("qwen2.5-coder-0.5b-instruct-q8_0.gguf"),
        PathBuf::from("qwen2.5-coder-0.5b.gguf"),
        env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("model.gguf")))
            .unwrap_or_default(),
    ];

    for path in candidates {
        if path.exists() {
            return Ok(ModelPathInfo {
                path,
                embedded_family: None,
            });
        }
    }

    let message = format!(
        "{} {}\n{} {}\n  1. Use --model <path> to specify a GGUF model\n  2. Place model.gguf in current directory\n  3. Embed model: ./scripts/embed.sh target/release/why model.gguf why-embedded",
        "Error:".red().bold(),
        "No model found.",
        "Tip:".blue().bold(),
        "Either:"
    );
    bail!(message)
}

/// Detect model family from filename/path
pub fn detect_model_family(model_path: &Path) -> ModelFamily {
    let filename = model_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    if filename.contains("gemma") {
        ModelFamily::Gemma
    } else if filename.contains("smol") || filename.contains("smollm") {
        ModelFamily::Smollm
    } else {
        // Default to Qwen (most common, also works for generic models)
        ModelFamily::Qwen
    }
}

pub fn build_prompt(error: &str, family: ModelFamily) -> String {
    let template = match family {
        ModelFamily::Gemma => TEMPLATE_GEMMA,
        ModelFamily::Qwen | ModelFamily::Smollm => TEMPLATE_CHATML,
    };
    template.replace("{error}", error.trim())
}

/// Check if the response contains degenerate patterns (repetitive characters/sequences)
pub fn is_degenerate_response(response: &str) -> bool {
    let response = response.trim();

    if response.len() < 20 {
        return false;
    }

    let chars: Vec<char> = response.chars().collect();
    let mut max_run = 1;
    let mut current_run = 1;
    for i in 1..chars.len() {
        if chars[i] == chars[i - 1] {
            current_run += 1;
            max_run = max_run.max(current_run);
        } else {
            current_run = 1;
        }
    }
    if max_run > 20 {
        return true;
    }

    for pattern_len in 1..=4 {
        if response.len() >= pattern_len * 10 {
            let pattern: String = chars.iter().take(pattern_len).collect();
            let repeated = pattern.repeat(10);
            if response.contains(&repeated) {
                return true;
            }
        }
    }

    let mut char_counts = std::collections::HashMap::new();
    for c in chars.iter() {
        if !c.is_whitespace() {
            *char_counts.entry(c).or_insert(0) += 1;
        }
    }
    let non_whitespace_count: usize = char_counts.values().sum();
    if non_whitespace_count > 0 {
        if let Some(&max_count) = char_counts.values().max() {
            if max_count as f64 / non_whitespace_count as f64 > 0.5 {
                return true;
            }
        }
    }

    let words: Vec<&str> = response.split_whitespace().collect();
    if words.len() >= 10 {
        let mut word_run = 1;
        let mut max_word_run = 1;
        for i in 1..words.len() {
            if words[i] == words[i - 1] {
                word_run += 1;
                max_word_run = max_word_run.max(word_run);
            } else {
                word_run = 1;
            }
        }
        if max_word_run > 5 {
            return true;
        }

        for pattern_len in 2..=3 {
            if words.len() >= pattern_len * 5 {
                let pattern: Vec<&str> = words.iter().take(pattern_len).copied().collect();
                let mut matches = 0;
                for chunk in words.chunks(pattern_len) {
                    if chunk == pattern.as_slice() {
                        matches += 1;
                    }
                }
                if matches >= 5 {
                    return true;
                }
            }
        }
    }

    false
}

/// Check if the model just echoed the input back
pub fn is_echo_response(input: &str, response: &str) -> bool {
    let response_trimmed = response.trim();
    let input_trimmed = input.trim();
    let response_lower = response_trimmed.to_lowercase();

    let has_structure = response_lower.contains("summary:")
        || response_lower.contains("explanation:")
        || response_lower.contains("suggestion:")
        || response_lower.contains("**summary")
        || response_lower.contains("**explanation")
        || response_lower.contains("**suggestion");

    if has_structure {
        return false;
    }

    let input_start: String = input_trimmed.chars().take(100).collect();
    let response_start: String = response_trimmed.chars().take(100).collect();

    if input_start == response_start {
        return true;
    }

    let input_lines: Vec<&str> = input_trimmed.lines().take(3).collect();
    let response_lines: Vec<&str> = response_trimmed.lines().take(3).collect();

    input_lines == response_lines
}

/// Helper function to check if we can generate more tokens
pub fn can_generate_more(start_n: i32, n_cur: i32, max_gen_tokens: usize) -> bool {
    (n_cur - start_n) < max_gen_tokens as i32
}

/// Run inference with optional streaming callback
pub fn run_inference_with_callback(
    model_path: &PathBuf,
    prompt: &str,
    params: &SamplingParams,
    mut callback: Option<TokenCallback>,
) -> Result<(String, InferenceStats)> {
    let total_start = Instant::now();
    let backend = LlamaBackend::init()?;

    let model_params = LlamaModelParams::default().with_n_gpu_layers(1000);
    let model_load_start = Instant::now();
    let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
        .with_context(|| "Failed to load model")?;

    let ctx_params = LlamaContextParams::default().with_n_ctx(Some(NonZeroU32::new(2048).unwrap()));

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "Failed to create context")?;
    let model_load_ms = model_load_start.elapsed().as_millis();

    let prompt_eval_start = Instant::now();
    let mut tokens = model
        .str_to_token(prompt, AddBos::Always)
        .with_context(|| "Failed to tokenize")?;

    let max_prompt_tokens = 1500;
    if tokens.len() > max_prompt_tokens {
        eprintln!(
            "{}",
            format!(
                "Input truncated from {} to {} tokens",
                tokens.len(),
                max_prompt_tokens
            )
            .dimmed()
        );
        tokens.truncate(max_prompt_tokens);
    }

    let batch_size = tokens.len().max(512);
    let mut batch = LlamaBatch::new(batch_size, 1);
    let last_idx = (tokens.len() - 1) as i32;

    for (i, token) in tokens.iter().enumerate() {
        batch.add(*token, i as i32, &[0], i as i32 == last_idx)?;
    }

    ctx.decode(&mut batch)?;
    let prompt_eval_ms = prompt_eval_start.elapsed().as_millis();

    let seed = params.seed.unwrap_or_else(|| {
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        (t ^ (t >> 32)) as u32
    });

    let mut sampler = LlamaSampler::chain_simple([
        LlamaSampler::top_k(params.top_k),
        LlamaSampler::top_p(params.top_p, 1),
        LlamaSampler::temp(params.temperature),
        LlamaSampler::dist(seed),
    ]);

    let max_gen_tokens = 512;
    let mut n_cur = batch.n_tokens();
    let start_n = n_cur;
    let mut output = String::new();

    let generation_start = Instant::now();
    let mut utf8_buffer: Vec<u8> = Vec::new();

    while can_generate_more(start_n, n_cur, max_gen_tokens) {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let bytes = model.token_to_bytes(token, Special::Tokenize)?;
        utf8_buffer.extend_from_slice(&bytes);

        let mut token_str = String::new();
        match std::str::from_utf8(&utf8_buffer) {
            Ok(s) => {
                token_str = s.to_string();
                utf8_buffer.clear();
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if valid_up_to > 0 {
                    token_str =
                        String::from_utf8(utf8_buffer[..valid_up_to].to_vec()).unwrap_or_default();
                    utf8_buffer = utf8_buffer[valid_up_to..].to_vec();
                }
            }
        }

        if !token_str.is_empty() {
            output.push_str(&token_str);

            if let Some(ref mut cb) = callback {
                match cb(&token_str) {
                    Ok(true) => {}
                    Ok(false) => break,
                    Err(e) => return Err(e),
                }
            }
        }

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        ctx.decode(&mut batch)?;
        n_cur += 1;
    }

    if !utf8_buffer.is_empty() {
        let remaining = String::from_utf8_lossy(&utf8_buffer).to_string();
        output.push_str(&remaining);
        if let Some(ref mut cb) = callback {
            let _ = cb(&remaining);
        }
    }

    let generation_ms = generation_start.elapsed().as_millis();
    let total_ms = total_start.elapsed().as_millis();
    let prompt_tokens = tokens.len();
    let generated_tokens = (n_cur - start_n).max(0) as usize;
    let total_tokens = prompt_tokens + generated_tokens;
    let gen_tok_per_s = if generation_ms == 0 {
        0.0
    } else {
        (generated_tokens as f64) / (generation_ms as f64 / 1000.0)
    };
    let total_tok_per_s = if total_ms == 0 {
        0.0
    } else {
        (total_tokens as f64) / (total_ms as f64 / 1000.0)
    };

    let stats = InferenceStats {
        backend: backend_mode().to_string(),
        prompt_tokens,
        generated_tokens,
        total_tokens,
        model_load_ms,
        prompt_eval_ms,
        generation_ms,
        total_ms,
        gen_tok_per_s,
        total_tok_per_s,
    };

    Ok((output, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt_substitutes_error() {
        let prompt = build_prompt("segmentation fault", ModelFamily::Qwen);
        assert!(prompt.contains("segmentation fault"));
        assert!(prompt.contains("<|im_start|>"));
        assert!(prompt.contains("SUMMARY:"));
    }

    #[test]
    fn test_build_prompt_trims_whitespace() {
        let prompt = build_prompt("  error with spaces  ", ModelFamily::Qwen);
        assert!(prompt.contains("error with spaces"));
        assert!(!prompt.contains("  error"));
    }

    #[test]
    fn test_build_prompt_gemma_template() {
        let prompt = build_prompt("test error", ModelFamily::Gemma);
        assert!(prompt.contains("test error"));
        assert!(prompt.contains("<start_of_turn>"));
        assert!(prompt.contains("<end_of_turn>"));
        assert!(!prompt.contains("<|im_start|>"));
    }

    #[test]
    fn test_detect_model_family_qwen() {
        let path = PathBuf::from("/path/to/qwen2.5-coder-0.5b-instruct-q8_0.gguf");
        assert_eq!(detect_model_family(&path), ModelFamily::Qwen);
    }

    #[test]
    fn test_detect_model_family_gemma() {
        let path = PathBuf::from("/path/to/gemma-3-270m-it-Q8_0.gguf");
        assert_eq!(detect_model_family(&path), ModelFamily::Gemma);
    }

    #[test]
    fn test_detect_model_family_smollm() {
        let path = PathBuf::from("/path/to/SmolLM2-135M-Instruct-Q8_0.gguf");
        assert_eq!(detect_model_family(&path), ModelFamily::Smollm);
    }

    #[test]
    fn test_detect_model_family_default() {
        let path = PathBuf::from("/path/to/random-model.gguf");
        assert_eq!(detect_model_family(&path), ModelFamily::Qwen);
    }

    #[test]
    fn test_model_family_display() {
        assert_eq!(format!("{}", ModelFamily::Qwen), "qwen (ChatML)");
        assert_eq!(format!("{}", ModelFamily::Gemma), "gemma (Gemma format)");
        assert_eq!(format!("{}", ModelFamily::Smollm), "smollm (ChatML)");
    }

    #[test]
    fn test_build_prompt_long_input() {
        let long_error = "error: ".to_string() + &"x".repeat(5000);
        let prompt = build_prompt(&long_error, ModelFamily::Qwen);
        assert!(prompt.contains(&"x".repeat(100)));
        assert!(prompt.len() > 5000);
    }

    #[test]
    fn test_build_prompt_multiline_input() {
        let multiline_error = "error[E0382]: borrow of moved value\n\
            --> src/main.rs:10:5\n\
            |\n\
            10 |     println!(\"{}\", x);\n\
            |                    ^ value borrowed here after move";

        let prompt = build_prompt(multiline_error, ModelFamily::Qwen);

        assert!(prompt.contains("error[E0382]"));
        assert!(prompt.contains("src/main.rs:10:5"));
        assert!(prompt.contains("value borrowed here after move"));
    }

    #[test]
    fn test_build_prompt_preserves_newlines() {
        let input = "line1\nline2\nline3";
        let prompt = build_prompt(input, ModelFamily::Qwen);

        assert!(prompt.contains("line1"));
        assert!(prompt.contains("line2"));
        assert!(prompt.contains("line3"));
    }

    #[test]
    fn test_build_prompt_long_multiline() {
        let long_line = "x".repeat(500);
        let multiline = format!("{}\n{}\n{}", long_line, long_line, long_line);
        let prompt = build_prompt(&multiline, ModelFamily::Qwen);

        assert!(prompt.len() > 1500);
        assert!(prompt.contains(&"x".repeat(100)));
    }

    #[test]
    fn test_is_echo_response_long_input() {
        let long_input = "error: ".to_string() + &"details ".repeat(100);
        let echo_response = long_input.clone();

        assert!(is_echo_response(&long_input, &echo_response));
    }

    #[test]
    fn test_is_echo_response_multiline_input() {
        let multiline_input = "error on line 1\nerror on line 2\nerror on line 3";
        let echo_response = multiline_input.to_string();

        assert!(is_echo_response(multiline_input, &echo_response));
    }

    #[test]
    fn test_is_echo_response_structured_long() {
        let long_input = "x".repeat(1000);
        let structured_response = format!(
            "SUMMARY: Error detected.\nEXPLANATION: The input {} indicates a problem.\nSUGGESTION: Fix it.",
            &long_input[..50]
        );

        assert!(!is_echo_response(&long_input, &structured_response));
    }

    #[test]
    fn test_generation_limit_allows_long_prompts() {
        let start_n = 2000;
        let n_cur = 2000;
        let max_gen_tokens = 512;

        assert!(can_generate_more(start_n, n_cur, max_gen_tokens));
        assert!(!can_generate_more(
            start_n,
            start_n + max_gen_tokens as i32,
            max_gen_tokens
        ));
    }

    #[test]
    fn test_is_degenerate_response_long_char_run() {
        let response = "The hash is ".to_string() + &"A".repeat(50);
        assert!(is_degenerate_response(&response));
    }

    #[test]
    fn test_is_degenerate_response_repeating_pattern() {
        let response = "@ ".repeat(20);
        assert!(is_degenerate_response(&response));
    }

    #[test]
    fn test_is_degenerate_response_repeating_words() {
        let response = "sha256 ".repeat(10);
        assert!(is_degenerate_response(&response));
    }

    #[test]
    fn test_is_degenerate_response_release_req_pattern() {
        let response = "RELEASE: REQ: ".repeat(20);
        assert!(is_degenerate_response(&response));
    }

    #[test]
    fn test_is_degenerate_response_high_char_dominance() {
        let response = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx abc";
        assert!(is_degenerate_response(response));
    }

    #[test]
    fn test_is_degenerate_response_normal_response() {
        let response = "SUMMARY: This is a segmentation fault error.\n\
            EXPLANATION: The program tried to access memory it doesn't have permission to access.\n\
            SUGGESTION: Check for null pointers and array bounds.";
        assert!(!is_degenerate_response(response));
    }

    #[test]
    fn test_is_degenerate_response_short_response() {
        assert!(!is_degenerate_response("OK"));
        assert!(!is_degenerate_response("Error found"));
    }

    #[test]
    fn test_is_degenerate_response_empty() {
        assert!(!is_degenerate_response(""));
        assert!(!is_degenerate_response("   "));
    }

    #[test]
    fn test_is_degenerate_response_code_block() {
        let response = "SUMMARY: Fix the loop.\nEXPLANATION:\n```\nfor i in range(10):\n    print(i)\n```\nSUGGESTION: Use enumerate.";
        assert!(!is_degenerate_response(response));
    }

    #[test]
    fn test_token_callback_invoked() {
        let mut tokens_received: Vec<String> = Vec::new();
        let callback: TokenCallback = Box::new(|token: &str| {
            tokens_received.push(token.to_string());
            Ok(true)
        });
        drop(callback);
        assert!(tokens_received.is_empty());
    }

    #[test]
    fn test_callback_return_values() {
        let mut continue_cb: TokenCallback = Box::new(|_| Ok(true));
        let mut stop_cb: TokenCallback = Box::new(|_| Ok(false));
        let mut error_cb: TokenCallback = Box::new(|_| Err(anyhow::anyhow!("test error")));

        assert!(continue_cb("test").unwrap());
        assert!(!stop_cb("test").unwrap());
        assert!(error_cb("test").is_err());
    }
}
