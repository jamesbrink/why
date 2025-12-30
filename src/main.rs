use anyhow::{bail, Context, Result};
use clap::{CommandFactory, Parser};
use clap_complete::{generate, Shell};
use colored::Colorize;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use serde::Serialize;
use std::env;
use std::fs::File;
use std::io::{self, BufRead, IsTerminal, Read, Seek, SeekFrom};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::time::Instant;
use std::time::{SystemTime, UNIX_EPOCH};

/// Magic marker written before embedded model
const MAGIC: &[u8; 8] = b"WHYMODEL";

/// Quick error explanation using local LLM
#[derive(Parser, Debug)]
#[command(
    name = "why",
    version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("WHY_GIT_SHA"), ")"),
    about,
    long_about = None
)]
#[command(
    after_help = "EXAMPLES:\n    why \"segmentation fault\"\n    cargo build 2>&1 | why\n    why --json \"null pointer exception\""
)]
struct Cli {
    /// Error message to explain
    #[arg(trailing_var_arg = true)]
    error: Vec<String>,

    /// Output as JSON
    #[arg(long, short = 'j')]
    json: bool,

    /// Show debug info (raw prompt and model response)
    #[arg(long, short = 'd')]
    debug: bool,

    /// Show performance stats
    #[arg(long)]
    stats: bool,

    /// Generate shell completions
    #[arg(long, value_enum, value_name = "SHELL")]
    completions: Option<Shell>,
}

#[derive(Debug, Serialize)]
struct ErrorExplanation {
    error: String,
    summary: String,
    explanation: String,
    suggestion: String,
}

#[derive(Debug, Serialize)]
struct InferenceStats {
    backend: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    total_tokens: usize,
    model_load_ms: u128,
    prompt_eval_ms: u128,
    generation_ms: u128,
    total_ms: u128,
    gen_tok_per_s: f64,
    total_tok_per_s: f64,
}

fn format_error(message: &str, tip: Option<&str>) -> String {
    let mut output = format!("{} {}", "Error:".red().bold(), message);
    if let Some(tip) = tip {
        output.push('\n');
        output.push_str(&format!("{} {}", "Tip:".blue().bold(), tip));
    }
    output
}

fn print_debug_section(title: &str, body: &str, footer: Option<String>) {
    eprintln!("{}", format!("=== DEBUG: {title} ===").yellow().bold());
    if body.trim().is_empty() {
        eprintln!("{}", "| <empty>".dimmed());
    } else {
        for line in body.lines() {
            eprintln!("{}", format!("| {line}").bright_white());
        }
    }
    if let Some(footer) = footer {
        eprintln!("{}", footer.dimmed());
    }
    eprintln!();
}

fn backend_mode() -> &'static str {
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

fn print_stats(stats: &InferenceStats) {
    println!("{} {}", "▸".magenta(), "Stats".magenta().bold());
    println!(
        "  {} {}",
        "Backend:".blue().bold(),
        stats.backend.bright_white()
    );
    println!(
        "  {} {}",
        "Tokens:".blue().bold(),
        format!(
            "prompt {}, generated {}, total {}",
            stats.prompt_tokens, stats.generated_tokens, stats.total_tokens
        )
        .bright_white()
    );
    println!(
        "  {} {}",
        "Timing:".blue().bold(),
        format!(
            "model {} ms, prompt {} ms, gen {} ms, total {} ms",
            stats.model_load_ms, stats.prompt_eval_ms, stats.generation_ms, stats.total_ms
        )
        .bright_white()
    );
    println!(
        "  {} {}",
        "Speed:".blue().bold(),
        format!(
            "gen {:.1} tok/s, total {:.1} tok/s",
            stats.gen_tok_per_s, stats.total_tok_per_s
        )
        .bright_white()
    );
    println!();
}

fn find_embedded_model() -> Result<(u64, u64)> {
    let exe_path = env::current_exe().context("failed to get executable path")?;
    let mut file = File::open(&exe_path).context("failed to open self")?;
    let file_len = file.metadata()?.len();

    if file_len < 24 {
        bail!(format_error("No embedded model found.", None));
    }

    file.seek(SeekFrom::End(-24))?;
    let mut trailer = [0u8; 24];
    file.read_exact(&mut trailer)?;

    if &trailer[0..8] != MAGIC {
        bail!(format_error(
            "No embedded model found (missing magic).",
            None
        ));
    }

    let offset = u64::from_le_bytes(trailer[8..16].try_into().unwrap());
    let size = u64::from_le_bytes(trailer[16..24].try_into().unwrap());

    Ok((offset, size))
}

fn get_model_path() -> Result<PathBuf> {
    // Check for embedded model first
    if let Ok((offset, size)) = find_embedded_model() {
        let exe_path = env::current_exe()?;
        let mut file = File::open(&exe_path)?;

        // Extract to temp
        let temp_path = env::temp_dir().join("why-model.gguf");
        if !temp_path.exists() || temp_path.metadata().map(|m| m.len()).unwrap_or(0) != size {
            eprintln!("{}", "Extracting embedded model...".dimmed());
            file.seek(SeekFrom::Start(offset))?;
            let mut model_data = vec![0u8; size as usize];
            file.read_exact(&mut model_data)?;
            std::fs::write(&temp_path, model_data)?;
        }
        return Ok(temp_path);
    }

    // Fallback: look for model file in current dir or next to exe
    let candidates = [
        PathBuf::from("qwen2.5-coder-0.5b.gguf"),
        PathBuf::from("model.gguf"),
        env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("model.gguf")))
            .unwrap_or_default(),
    ];

    for path in candidates {
        if path.exists() {
            return Ok(path);
        }
    }

    let message = format!(
        "{} {}\n{} {}\n  1. Place qwen2.5-coder-0.5b.gguf in current directory\n  2. Embed model: ./scripts/embed.sh target/release/why model.gguf why-embedded",
        "Error:".red().bold(),
        "No model found.",
        "Tip:".blue().bold(),
        "Either:"
    );
    bail!(message)
}

fn get_input(cli: &Cli) -> Result<String> {
    // If error args provided, use them
    if !cli.error.is_empty() {
        return Ok(cli.error.join(" "));
    }

    // Otherwise read from stdin if piped
    if !io::stdin().is_terminal() {
        let stdin = io::stdin();
        let mut input = String::new();
        for line in stdin.lock().lines() {
            input.push_str(&line?);
            input.push('\n');
        }
        let trimmed = input.trim().to_string();
        if !trimmed.is_empty() {
            return Ok(trimmed);
        }
    }

    let message = format!(
        "{} {}\n{} {}",
        "Error:".red().bold(),
        "No input provided. Usage: why <error message>",
        "Tip:".blue().bold(),
        "Use 2>&1 to capture stderr: command 2>&1 | why".dimmed()
    );
    bail!(message)
}

const PROMPT_TEMPLATE: &str = include_str!("prompt.txt");

fn build_prompt(error: &str) -> String {
    PROMPT_TEMPLATE.replace("{error}", error.trim())
}

/// Sampling parameters for inference
#[derive(Clone)]
struct SamplingParams {
    temperature: f32,
    top_p: f32,
    top_k: i32,
    seed: Option<u32>,
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

/// Maximum number of retries when detecting degenerate output
const MAX_RETRIES: usize = 2;

/// Check if the response contains degenerate patterns (repetitive characters/sequences)
/// that indicate the model is stuck in a loop
fn is_degenerate_response(response: &str) -> bool {
    let response = response.trim();

    // Empty or very short responses aren't degenerate, just empty
    if response.len() < 20 {
        return false;
    }

    // Check 1: Long runs of the same character (e.g., "AAAA..." or "@@@@...")
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
    // More than 20 identical characters in a row is degenerate
    if max_run > 20 {
        return true;
    }

    // Check 2: Short repeating patterns (e.g., "@ @ @ @ " or "ab ab ab ab")
    // Look for patterns of length 1-4 that repeat many times
    for pattern_len in 1..=4 {
        if response.len() >= pattern_len * 10 {
            let pattern: String = chars.iter().take(pattern_len).collect();
            let repeated = pattern.repeat(10);
            if response.contains(&repeated) {
                return true;
            }
        }
    }

    // Check 3: High single-character dominance
    // If any single character makes up more than 50% of the response, it's suspicious
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

    // Check 4: Repeating word/token patterns (e.g., "sha256 sha256 sha256")
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
        // More than 5 identical words in a row is degenerate
        if max_word_run > 5 {
            return true;
        }

        // Check 5: Repeating word pairs/triplets (e.g., "RELEASE: REQ: RELEASE: REQ:")
        for pattern_len in 2..=3 {
            if words.len() >= pattern_len * 5 {
                let pattern: Vec<&str> = words.iter().take(pattern_len).copied().collect();
                let mut matches = 0;
                for chunk in words.chunks(pattern_len) {
                    if chunk == pattern.as_slice() {
                        matches += 1;
                    }
                }
                // If the same word pattern repeats 5+ times, it's degenerate
                if matches >= 5 {
                    return true;
                }
            }
        }
    }

    false
}

/// Check if the model just echoed the input back (indicates confusion, not an error)
fn is_echo_response(input: &str, response: &str) -> bool {
    let response_trimmed = response.trim();
    let input_trimmed = input.trim();
    let response_lower = response_trimmed.to_lowercase();

    // If response has our expected structure markers (case-insensitive, with or without markdown)
    let has_structure = response_lower.contains("summary:")
        || response_lower.contains("explanation:")
        || response_lower.contains("suggestion:")
        || response_lower.contains("**summary")
        || response_lower.contains("**explanation")
        || response_lower.contains("**suggestion");

    if has_structure {
        return false;
    }

    // Check if response starts with significant portion of input (echo detection)
    let input_start: String = input_trimmed.chars().take(100).collect();
    let response_start: String = response_trimmed.chars().take(100).collect();

    // If first 100 chars match closely, it's likely an echo
    if input_start == response_start {
        return true;
    }

    // Check if response is suspiciously similar in length and content
    let input_lines: Vec<&str> = input_trimmed.lines().take(3).collect();
    let response_lines: Vec<&str> = response_trimmed.lines().take(3).collect();

    input_lines == response_lines
}

/// Extract section label from a line, handling various formats:
/// - "SUMMARY:" or "SUMMARY"
/// - "**Summary:**" or "**SUMMARY:**"
/// - "Summary:" (case-insensitive)
///
/// Returns (section_name, rest_of_line) if a label is found.
fn extract_section_label(line: &str) -> Option<(&'static str, String)> {
    // Strip markdown bold markers if present
    let cleaned = line.trim_start_matches("**").trim_start_matches('*');
    let cleaned_lower = cleaned.to_lowercase();

    for (label, section) in [
        ("summary", "summary"),
        ("explanation", "explanation"),
        ("suggestion", "suggestion"),
    ] {
        // Check for "label:" or "label" at start
        if cleaned_lower.starts_with(label) {
            let after_label = &cleaned[label.len()..];
            // Must be followed by ":", "**:", or end of string
            let rest = if let Some(stripped) = after_label.strip_prefix(':') {
                stripped.trim_start_matches("**").trim()
            } else if let Some(stripped) = after_label.strip_prefix("**:") {
                stripped.trim()
            } else if after_label.is_empty()
                || after_label.starts_with("**")
                || after_label
                    .chars()
                    .next()
                    .map(|c| c.is_whitespace())
                    .unwrap_or(false)
            {
                after_label.trim_start_matches("**").trim()
            } else {
                continue;
            };
            return Some((section, rest.to_string()));
        }
    }
    None
}

fn parse_response(error: &str, response: &str) -> ErrorExplanation {
    let mut summary = String::new();
    let mut explanation = String::new();
    let mut suggestion = String::new();
    let mut current_section = "summary"; // Start with summary since prompt ends with "SUMMARY:"

    for line in response.lines() {
        let line = line.trim();

        // Check for section headers using flexible matching
        if let Some((section, rest)) = extract_section_label(line) {
            current_section = section;
            let target = match section {
                "summary" => &mut summary,
                "explanation" => &mut explanation,
                "suggestion" => &mut suggestion,
                _ => &mut summary,
            };
            if !rest.is_empty() {
                *target = rest;
            }
        } else if !line.is_empty() {
            let target = match current_section {
                "summary" => &mut summary,
                "explanation" => &mut explanation,
                "suggestion" => &mut suggestion,
                _ => &mut summary,
            };
            if !target.is_empty() {
                target.push(' ');
            }
            target.push_str(line);
        }
    }

    // Fallback if parsing fails completely
    if summary.is_empty() && explanation.is_empty() && suggestion.is_empty() {
        explanation = response.trim().to_string();
        summary = response.lines().next().unwrap_or("").to_string();
    }

    ErrorExplanation {
        error: error.to_string(),
        summary,
        explanation,
        suggestion,
    }
}

/// Render markdown text to terminal with colored output.
/// Handles inline code (`code`), code blocks (```), bold (**), and italic (*).
fn render_markdown(text: &str, width: usize, indent: &str) {
    let mut in_code_block = false;
    let mut code_block_content: Vec<String> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Handle code block delimiters
        if trimmed.starts_with("```") {
            if in_code_block {
                // End of code block - print accumulated content
                for code_line in &code_block_content {
                    println!("{indent}  {}", code_line.cyan());
                }
                code_block_content.clear();
                in_code_block = false;
            } else {
                // Start of code block
                in_code_block = true;
            }
            continue;
        }

        if in_code_block {
            code_block_content.push(line.to_string());
            continue;
        }

        // Process inline markdown for regular text
        let processed = render_inline_markdown(line);

        // Wrap and print
        for wrapped_line in textwrap::wrap(&processed, width.saturating_sub(indent.len())) {
            println!("{indent}{wrapped_line}");
        }
    }

    // Handle unclosed code block
    if in_code_block {
        for code_line in &code_block_content {
            println!("{indent}  {}", code_line.cyan());
        }
    }
}

/// Process inline markdown: `code`, **bold**, *italic*
fn render_inline_markdown(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '`' {
            // Inline code
            let mut code = String::new();
            while let Some(&next) = chars.peek() {
                if next == '`' {
                    chars.next();
                    break;
                }
                code.push(chars.next().unwrap());
            }
            result.push_str(&code.cyan().to_string());
        } else if c == '*' {
            if chars.peek() == Some(&'*') {
                // Bold
                chars.next();
                let mut bold_text = String::new();
                while let Some(&next) = chars.peek() {
                    if next == '*' {
                        chars.next();
                        if chars.peek() == Some(&'*') {
                            chars.next();
                        }
                        break;
                    }
                    bold_text.push(chars.next().unwrap());
                }
                result.push_str(&bold_text.bold().to_string());
            } else {
                // Italic
                let mut italic_text = String::new();
                while let Some(&next) = chars.peek() {
                    if next == '*' {
                        chars.next();
                        break;
                    }
                    italic_text.push(chars.next().unwrap());
                }
                result.push_str(&italic_text.italic().to_string());
            }
        } else {
            result.push(c);
        }
    }

    result
}

fn print_colored(result: &ErrorExplanation) {
    // Get terminal width, default to 80
    let width = textwrap::termwidth().min(100);

    println!();
    println!("{} {}", "●".red(), result.error.bold());
    println!();

    if !result.summary.is_empty() {
        let processed = render_inline_markdown(&result.summary);
        for line in textwrap::wrap(&processed, width) {
            println!("{}", line.white().bold());
        }
        println!();
    }

    if !result.explanation.is_empty() {
        println!("{} {}", "▸".blue(), "Explanation".blue().bold());
        render_markdown(&result.explanation, width, "  ");
        println!();
    }

    if !result.suggestion.is_empty() {
        println!("{} {}", "▸".green(), "Suggestion".green().bold());
        render_markdown(&result.suggestion, width, "  ");
        println!();
    }
}

fn run_inference(
    model_path: &PathBuf,
    prompt: &str,
    params: &SamplingParams,
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

    // Truncate if input exceeds context window (leave room for generation)
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

    // Batch size must accommodate all prompt tokens plus room for generation
    let batch_size = tokens.len().max(512);
    let mut batch = LlamaBatch::new(batch_size, 1);
    let last_idx = (tokens.len() - 1) as i32;

    for (i, token) in tokens.iter().enumerate() {
        batch.add(*token, i as i32, &[0], i as i32 == last_idx)?;
    }

    ctx.decode(&mut batch)?;
    let prompt_eval_ms = prompt_eval_start.elapsed().as_millis();

    // Use provided seed or generate one from system time
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

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let max_gen_tokens = 512;
    let mut n_cur = batch.n_tokens();
    let start_n = n_cur;
    let mut output = String::new();

    let generation_start = Instant::now();
    while can_generate_more(start_n, n_cur, max_gen_tokens) {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let bytes = model.token_to_bytes(token, Special::Tokenize)?;
        let mut token_str = String::with_capacity(32);
        let _ = decoder.decode_to_string(&bytes, &mut token_str, false);
        output.push_str(&token_str);

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        ctx.decode(&mut batch)?;
        n_cur += 1;
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

    Ok((
        output,
        InferenceStats {
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
        },
    ))
}

fn can_generate_more(start_n: i32, n_cur: i32, max_gen_tokens: i32) -> bool {
    n_cur.saturating_sub(start_n) < max_gen_tokens
}

fn print_completions(shell: Shell) {
    let mut cmd = Cli::command();
    generate(shell, &mut cmd, "why", &mut io::stdout());
}

fn main() -> Result<()> {
    // Suppress verbose llama.cpp logs immediately
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

    let cli = Cli::parse();

    // Handle completions
    if let Some(shell) = cli.completions {
        print_completions(shell);
        return Ok(());
    }

    let input = get_input(&cli)?;
    let model_path = get_model_path()?;
    let prompt = build_prompt(&input);

    if cli.debug {
        print_debug_section(
            "Input",
            &input,
            Some(format!(
                "({} chars, {} lines)",
                input.len(),
                input.lines().count()
            )),
        );
        print_debug_section("Prompt", &prompt, Some(format!("({} chars)", prompt.len())));
        eprintln!("{}", "=== DEBUG: Model ===".yellow().bold());
        match std::fs::metadata(&model_path) {
            Ok(meta) => {
                let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
                eprintln!(
                    "{} {} ({:.1} MB)",
                    "Path:".blue().bold(),
                    model_path.display(),
                    size_mb
                );
            }
            Err(err) => {
                eprintln!(
                    "{} {} ({})",
                    "Path:".blue().bold(),
                    model_path.display(),
                    format!("size unavailable: {err}").dimmed()
                );
            }
        }
        eprintln!();
    }

    // Run inference with retry logic for degenerate outputs
    let mut params = SamplingParams::default();
    let mut response;
    let mut stats;
    let mut retries = 0;

    loop {
        (response, stats) = run_inference(&model_path, &prompt, &params)?;

        // Check for degenerate output (repetitive patterns)
        if is_degenerate_response(&response) {
            retries += 1;
            if retries > MAX_RETRIES {
                if cli.debug {
                    eprintln!(
                        "{}",
                        format!(
                            "Degenerate output detected after {} retries, giving up",
                            retries
                        )
                        .yellow()
                    );
                }
                break;
            }

            // Adjust sampling parameters for retry
            // Lower temperature and use a different seed to get more focused output
            params.temperature = 0.5 - (retries as f32 * 0.15); // 0.35, then 0.2
            params.temperature = params.temperature.max(0.1);
            params.top_p = 0.8;
            params.seed = Some(retries as u32 * 12345 + 42);

            if cli.debug {
                eprintln!(
                    "{}",
                    format!(
                        "Degenerate output detected, retrying ({}/{}) with temp={:.2}",
                        retries, MAX_RETRIES, params.temperature
                    )
                    .yellow()
                );
            } else {
                eprintln!(
                    "{}",
                    format!("Retrying inference ({}/{})...", retries, MAX_RETRIES).dimmed()
                );
            }
            continue;
        }

        break;
    }

    if cli.debug {
        print_debug_section(
            "Raw Response",
            &response,
            Some(format!(
                "({} chars, {} lines{})",
                response.len(),
                response.lines().count(),
                if retries > 0 {
                    format!(", {} retries", retries)
                } else {
                    String::new()
                }
            )),
        );
    }

    // Check if model detected no error, echoed input back, or returned nothing
    let is_no_error = response.trim().is_empty()
        || response.trim().starts_with("NO_ERROR")
        || is_echo_response(&input, &response)
        || (retries > MAX_RETRIES && is_degenerate_response(&response));

    if is_no_error {
        if cli.json {
            let mut payload = serde_json::json!({
                "input": input,
                "no_error": true,
                "message": "No error detected in input."
            });
            if cli.stats {
                payload["stats"] = serde_json::to_value(&stats)?;
            }
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            println!();
            println!("{} {}", "✓".green(), "No error detected".green().bold());
            println!();
            println!(
                "  {}",
                "The input doesn't appear to contain an error message.".dimmed()
            );
            println!();
            if cli.stats {
                print_stats(&stats);
            }
        }
        return Ok(());
    }

    let result = parse_response(&input, &response);

    // If parsing yielded no meaningful content, treat as no error / unusable input
    let has_content = !result.summary.is_empty()
        || !result.explanation.is_empty()
        || !result.suggestion.is_empty();

    if !has_content {
        if cli.json {
            let mut payload = serde_json::json!({
                "input": input,
                "no_error": true,
                "message": "Could not analyze input. It may not be an error message."
            });
            if cli.stats {
                payload["stats"] = serde_json::to_value(&stats)?;
            }
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            println!();
            println!(
                "{} {}",
                "?".yellow(),
                "Could not analyze input".yellow().bold()
            );
            println!();
            println!(
                "  {}",
                "The input may not be an error message, or is too complex to parse.".dimmed()
            );
            println!();
            if cli.stats {
                print_stats(&stats);
            }
        }
        return Ok(());
    }

    if cli.json {
        let mut payload = serde_json::json!({
            "input": input,
            "error": result.error,
            "summary": result.summary,
            "explanation": result.explanation,
            "suggestion": result.suggestion
        });
        if cli.stats {
            payload["stats"] = serde_json::to_value(&stats)?;
        }
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        print_colored(&result);
        if cli.stats {
            print_stats(&stats);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt_substitutes_error() {
        let prompt = build_prompt("segmentation fault");
        assert!(prompt.contains("segmentation fault"));
        assert!(prompt.contains("<|im_start|>"));
        assert!(prompt.contains("SUMMARY:"));
    }

    #[test]
    fn test_build_prompt_trims_whitespace() {
        let prompt = build_prompt("  error with spaces  ");
        assert!(prompt.contains("error with spaces"));
        assert!(!prompt.contains("  error")); // leading spaces trimmed
    }

    #[test]
    fn test_parse_response_standard_format() {
        let response = "Memory access violation at invalid address.\n\
            EXPLANATION: The program tried to access memory it doesn't own.\n\
            SUGGESTION: Check for null pointers and array bounds.";

        let result = parse_response("segfault", response);

        assert_eq!(result.error, "segfault");
        assert!(result.summary.contains("Memory access violation"));
        assert!(result.explanation.contains("memory it doesn't own"));
        assert!(result.suggestion.contains("null pointers"));
    }

    #[test]
    fn test_parse_response_with_headers() {
        let response = "SUMMARY: A null pointer was dereferenced.\n\
            EXPLANATION: You tried to use a pointer that points to nothing.\n\
            SUGGESTION: Initialize your pointers before use.";

        let result = parse_response("null pointer", response);

        assert_eq!(result.summary, "A null pointer was dereferenced.");
        assert!(result
            .explanation
            .contains("pointer that points to nothing"));
        assert!(result.suggestion.contains("Initialize"));
    }

    #[test]
    fn test_parse_response_inline_content() {
        let response = "SUMMARY: Stack overflow occurred.\n\
            EXPLANATION: Infinite recursion exhausted the stack.\n\
            SUGGESTION: Add a base case to your recursion.";

        let result = parse_response("stack overflow", response);

        assert_eq!(result.summary, "Stack overflow occurred.");
        assert!(result.explanation.contains("recursion"));
        assert!(result.suggestion.contains("base case"));
    }

    #[test]
    fn test_parse_response_multiline_sections() {
        let response = "SUMMARY: Type mismatch error.\n\
            EXPLANATION: The function expected an integer\n\
            but received a string instead.\n\
            SUGGESTION: Convert the string to int using parse().";

        let result = parse_response("type error", response);

        assert!(result.explanation.contains("expected an integer"));
        assert!(result.explanation.contains("string instead"));
    }

    #[test]
    fn test_parse_response_fallback_unstructured() {
        let response = "This error means something went wrong with your code.";

        let result = parse_response("generic error", response);

        // Unstructured text without headers goes to summary (default section)
        assert!(!result.summary.is_empty());
        assert!(result.summary.contains("something went wrong"));
    }

    #[test]
    fn test_parse_response_empty_sections() {
        let response = "SUMMARY:\nEXPLANATION: Something happened.\nSUGGESTION:";

        let result = parse_response("error", response);

        assert!(result.summary.is_empty());
        assert!(!result.explanation.is_empty());
        assert!(result.suggestion.is_empty());
    }

    #[test]
    fn test_error_explanation_serializes_to_json() {
        let explanation = ErrorExplanation {
            error: "test error".to_string(),
            summary: "test summary".to_string(),
            explanation: "test explanation".to_string(),
            suggestion: "test suggestion".to_string(),
        };

        let json = serde_json::to_string(&explanation).unwrap();

        assert!(json.contains("\"error\":\"test error\""));
        assert!(json.contains("\"summary\":\"test summary\""));
    }

    #[test]
    fn test_cli_parses_error_args() {
        let cli = Cli::parse_from(["why", "segmentation", "fault"]);
        assert_eq!(cli.error, vec!["segmentation", "fault"]);
        assert!(!cli.json);
    }

    #[test]
    fn test_cli_parses_json_flag() {
        let cli = Cli::parse_from(["why", "--json", "error"]);
        assert!(cli.json);
    }

    #[test]
    fn test_cli_parses_short_json_flag() {
        let cli = Cli::parse_from(["why", "-j", "error"]);
        assert!(cli.json);
    }

    #[test]
    fn test_cli_parses_stats_flag() {
        let cli = Cli::parse_from(["why", "--stats", "error"]);
        assert!(cli.stats);
    }

    #[test]
    fn test_cli_parses_completions() {
        let cli = Cli::parse_from(["why", "--completions", "bash"]);
        assert_eq!(cli.completions, Some(Shell::Bash));
    }

    // Long input tests
    #[test]
    fn test_build_prompt_long_input() {
        let long_error = "error: ".to_string() + &"x".repeat(5000);
        let prompt = build_prompt(&long_error);
        assert!(prompt.contains(&"x".repeat(100))); // Contains part of the long string
        assert!(prompt.len() > 5000);
    }

    #[test]
    fn test_parse_response_long_sections() {
        let long_explanation = "word ".repeat(500);
        let response = format!(
            "SUMMARY: Short summary.\nEXPLANATION: {}\nSUGGESTION: Fix it.",
            long_explanation
        );

        let result = parse_response("error", &response);

        assert_eq!(result.summary, "Short summary.");
        assert!(result.explanation.len() > 2000);
        assert!(result.explanation.contains("word word word"));
    }

    #[test]
    fn test_cli_long_error_args() {
        let long_msg = "a]".repeat(1000);
        let cli = Cli::parse_from(["why", &long_msg]);
        assert_eq!(cli.error.len(), 1);
        assert_eq!(cli.error[0].len(), 2000);
    }

    // Multiline input tests
    #[test]
    fn test_build_prompt_multiline_input() {
        let multiline_error = "error[E0382]: borrow of moved value\n\
            --> src/main.rs:10:5\n\
            |\n\
            10 |     println!(\"{}\", x);\n\
            |                    ^ value borrowed here after move";

        let prompt = build_prompt(multiline_error);

        assert!(prompt.contains("error[E0382]"));
        assert!(prompt.contains("src/main.rs:10:5"));
        assert!(prompt.contains("value borrowed here after move"));
    }

    #[test]
    fn test_build_prompt_preserves_newlines() {
        let input = "line1\nline2\nline3";
        let prompt = build_prompt(input);

        // The prompt should contain all lines
        assert!(prompt.contains("line1"));
        assert!(prompt.contains("line2"));
        assert!(prompt.contains("line3"));
    }

    #[test]
    fn test_parse_response_multiline_explanation() {
        let response = "SUMMARY: Compilation failed.\n\
            EXPLANATION: The compiler found multiple errors:\n\
            - Type mismatch on line 10\n\
            - Missing semicolon on line 15\n\
            - Undefined variable on line 20\n\
            SUGGESTION: Fix each error in order.";

        let result = parse_response("compile error", &response);

        assert_eq!(result.summary, "Compilation failed.");
        assert!(result.explanation.contains("Type mismatch"));
        assert!(result.explanation.contains("Missing semicolon"));
        assert!(result.explanation.contains("Undefined variable"));
    }

    #[test]
    fn test_cli_multiline_quoted_arg() {
        // When passed as a single quoted argument
        let cli = Cli::parse_from(["why", "line1\nline2\nline3"]);
        assert_eq!(cli.error.len(), 1);
        assert!(cli.error[0].contains('\n'));
        assert!(cli.error[0].contains("line1"));
        assert!(cli.error[0].contains("line3"));
    }

    #[test]
    fn test_parse_response_stack_trace() {
        let response = "SUMMARY: Null pointer exception in Java.\n\
            EXPLANATION: The stack trace shows:\n\
            at com.example.Main.process(Main.java:42)\n\
            at com.example.Main.main(Main.java:10)\n\
            The null reference originated in the process method.\n\
            SUGGESTION: Add null checks before calling methods on objects.";

        let result = parse_response("java.lang.NullPointerException", &response);

        assert!(result.explanation.contains("Main.java:42"));
        assert!(result.explanation.contains("null reference"));
    }

    // Combined long + multiline tests
    #[test]
    fn test_build_prompt_long_multiline() {
        let long_line = "x".repeat(500);
        let multiline = format!("{}\n{}\n{}", long_line, long_line, long_line);
        let prompt = build_prompt(&multiline);

        assert!(prompt.len() > 1500);
        assert!(prompt.contains(&"x".repeat(100)));
    }

    #[test]
    fn test_parse_response_real_rust_error() {
        let response = "SUMMARY: Ownership violation - value used after move.\n\
            EXPLANATION: In Rust, when you assign a value to another variable or pass it to a function, \
            ownership is transferred (moved). The original variable can no longer be used. \
            This error occurs at src/main.rs:10:5 where you tried to use 'x' after it was moved.\n\
            SUGGESTION: Consider using .clone() to create a copy, or borrow the value with & instead of moving it.";

        let result = parse_response("error[E0382]: borrow of moved value", &response);

        assert!(result.summary.contains("Ownership violation"));
        assert!(result.explanation.contains("ownership is transferred"));
        assert!(result.suggestion.contains("clone()"));
    }

    // Echo detection tests for long/multiline
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

    // Edge cases
    #[test]
    fn test_build_prompt_empty_lines_in_multiline() {
        let input = "error\n\n\nmore info\n\n";
        let prompt = build_prompt(input);

        assert!(prompt.contains("error"));
        assert!(prompt.contains("more info"));
    }

    #[test]
    fn test_parse_response_many_newlines() {
        let response = "SUMMARY: Test.\n\n\n\nEXPLANATION: Details.\n\n\nSUGGESTION: Fix.";
        let result = parse_response("error", &response);

        assert_eq!(result.summary, "Test.");
        assert_eq!(result.explanation, "Details.");
        assert_eq!(result.suggestion, "Fix.");
    }

    #[test]
    fn test_generation_limit_allows_long_prompts() {
        let start_n = 2000;
        let n_cur = 2000;
        let max_gen_tokens = 512;

        assert!(can_generate_more(start_n, n_cur, max_gen_tokens));
        assert!(!can_generate_more(
            start_n,
            start_n + max_gen_tokens,
            max_gen_tokens
        ));
    }

    #[test]
    fn test_cli_multiple_long_args() {
        let arg1 = "a".repeat(500);
        let arg2 = "b".repeat(500);
        let cli = Cli::parse_from(["why", &arg1, &arg2]);

        assert_eq!(cli.error.len(), 2);
        assert_eq!(cli.error[0].len(), 500);
        assert_eq!(cli.error[1].len(), 500);
    }

    #[test]
    fn test_cli_parses_debug_flag() {
        let cli = Cli::parse_from(["why", "--debug", "error"]);
        assert!(cli.debug);
    }

    #[test]
    fn test_cli_parses_short_debug_flag() {
        let cli = Cli::parse_from(["why", "-d", "error"]);
        assert!(cli.debug);
    }

    #[test]
    fn test_cli_debug_and_json_together() {
        let cli = Cli::parse_from(["why", "-d", "-j", "error"]);
        assert!(cli.debug);
        assert!(cli.json);
    }

    // Tests for extract_section_label guardrail
    #[test]
    fn test_extract_section_label_uppercase() {
        let (section, rest) = extract_section_label("SUMMARY: This is a summary").unwrap();
        assert_eq!(section, "summary");
        assert_eq!(rest, "This is a summary");
    }

    #[test]
    fn test_extract_section_label_lowercase() {
        let (section, rest) = extract_section_label("summary: This is a summary").unwrap();
        assert_eq!(section, "summary");
        assert_eq!(rest, "This is a summary");
    }

    #[test]
    fn test_extract_section_label_mixed_case() {
        let (section, rest) = extract_section_label("Summary: This is a summary").unwrap();
        assert_eq!(section, "summary");
        assert_eq!(rest, "This is a summary");
    }

    #[test]
    fn test_extract_section_label_markdown_bold() {
        let (section, rest) = extract_section_label("**Summary:** This is a summary").unwrap();
        assert_eq!(section, "summary");
        assert_eq!(rest, "This is a summary");
    }

    #[test]
    fn test_extract_section_label_markdown_bold_uppercase() {
        let (section, rest) = extract_section_label("**SUMMARY:** This is a summary").unwrap();
        assert_eq!(section, "summary");
        assert_eq!(rest, "This is a summary");
    }

    #[test]
    fn test_extract_section_label_explanation_markdown() {
        let (section, rest) = extract_section_label("**Explanation:** The error occurs").unwrap();
        assert_eq!(section, "explanation");
        assert_eq!(rest, "The error occurs");
    }

    #[test]
    fn test_extract_section_label_suggestion_markdown() {
        let (section, rest) = extract_section_label("**Suggestion:** Fix the code").unwrap();
        assert_eq!(section, "suggestion");
        assert_eq!(rest, "Fix the code");
    }

    #[test]
    fn test_extract_section_label_no_content() {
        let (section, rest) = extract_section_label("SUMMARY:").unwrap();
        assert_eq!(section, "summary");
        assert_eq!(rest, "");
    }

    #[test]
    fn test_extract_section_label_just_label() {
        let (section, rest) = extract_section_label("EXPLANATION").unwrap();
        assert_eq!(section, "explanation");
        assert_eq!(rest, "");
    }

    #[test]
    fn test_extract_section_label_not_a_section() {
        assert!(extract_section_label("This is just regular text").is_none());
        assert!(extract_section_label("summarizing the results").is_none());
        assert!(extract_section_label("explanatory note").is_none());
    }

    #[test]
    fn test_parse_response_markdown_format() {
        let response = "**Summary:** The error is a TypeError.\n\
            **Explanation:** You tried to access a property on undefined.\n\
            **Suggestion:** Check if the object exists first.";

        let result = parse_response("TypeError", response);

        assert_eq!(result.summary, "The error is a TypeError.");
        assert!(result.explanation.contains("access a property"));
        assert!(result.suggestion.contains("Check if the object"));
    }

    #[test]
    fn test_parse_response_mixed_formats() {
        // Model might mix formats within a response
        let response = "**Summary:** Mixed format test.\n\
            EXPLANATION: Using uppercase here.\n\
            **Suggestion:** And back to markdown.";

        let result = parse_response("error", response);

        assert_eq!(result.summary, "Mixed format test.");
        assert!(result.explanation.contains("uppercase"));
        assert!(result.suggestion.contains("back to markdown"));
    }

    #[test]
    fn test_parse_response_case_insensitive() {
        let response = "summary: lowercase labels work.\n\
            explanation: this should parse correctly.\n\
            suggestion: and so should this.";

        let result = parse_response("error", response);

        assert!(result.summary.contains("lowercase labels"));
        assert!(result.explanation.contains("parse correctly"));
        assert!(result.suggestion.contains("so should this"));
    }

    // Degenerate response detection tests
    #[test]
    fn test_is_degenerate_response_long_char_run() {
        // Long run of 'A' characters
        let response = "The hash is ".to_string() + &"A".repeat(50);
        assert!(is_degenerate_response(&response));
    }

    #[test]
    fn test_is_degenerate_response_repeating_pattern() {
        // Repeating "@ " pattern
        let response = "@ ".repeat(20);
        assert!(is_degenerate_response(&response));
    }

    #[test]
    fn test_is_degenerate_response_repeating_words() {
        // Same word repeated many times
        let response = "sha256 ".repeat(10);
        assert!(is_degenerate_response(&response));
    }

    #[test]
    fn test_is_degenerate_response_release_req_pattern() {
        // The actual pattern from user's report
        let response = "RELEASE: REQ: ".repeat(20);
        assert!(is_degenerate_response(&response));
    }

    #[test]
    fn test_is_degenerate_response_high_char_dominance() {
        // Single character makes up > 50% of response
        let response = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx abc";
        assert!(is_degenerate_response(response));
    }

    #[test]
    fn test_is_degenerate_response_normal_response() {
        // A normal, well-formed response should not be flagged
        let response = "SUMMARY: This is a segmentation fault error.\n\
            EXPLANATION: The program tried to access memory it doesn't have permission to access.\n\
            SUGGESTION: Check for null pointers and array bounds.";
        assert!(!is_degenerate_response(response));
    }

    #[test]
    fn test_is_degenerate_response_short_response() {
        // Very short responses aren't degenerate, just short
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
        // Code with repetitive structure shouldn't trigger false positives
        let response = "SUMMARY: Fix the loop.\nEXPLANATION:\n```\nfor i in range(10):\n    print(i)\n```\nSUGGESTION: Use enumerate.";
        assert!(!is_degenerate_response(response));
    }
}
