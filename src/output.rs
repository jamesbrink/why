//! Response parsing and terminal output formatting.

use colored::Colorize;
use serde::Serialize;
use std::path::Path;

use crate::model::InferenceStats;
use crate::stack_trace::StackTrace;

/// Parsed error explanation
#[derive(Debug, Serialize)]
pub struct ErrorExplanation {
    pub error: String,
    pub summary: String,
    pub explanation: String,
    pub suggestion: String,
}

/// Extract section label from a line, handling various formats:
/// - "SUMMARY:" or "SUMMARY"
/// - "**Summary:**" or "**SUMMARY:**"
/// - "Summary:" (case-insensitive)
///
/// Returns (section_name, rest_of_line) if a label is found.
pub fn extract_section_label(line: &str) -> Option<(&'static str, String)> {
    let cleaned = line.trim_start_matches("**").trim_start_matches('*');
    let cleaned_lower = cleaned.to_lowercase();

    for (label, section) in [
        ("summary", "summary"),
        ("explanation", "explanation"),
        ("suggestion", "suggestion"),
    ] {
        if cleaned_lower.starts_with(label) {
            let after_label = &cleaned[label.len()..];
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

pub fn parse_response(error: &str, response: &str) -> ErrorExplanation {
    let mut summary = String::new();
    let mut explanation = String::new();
    let mut suggestion = String::new();
    let mut current_section = "summary";

    for line in response.lines() {
        let line = line.trim();

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
pub fn render_markdown(text: &str, width: usize, indent: &str) {
    let mut in_code_block = false;
    let mut code_block_content: Vec<String> = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("```") {
            if in_code_block {
                for code_line in &code_block_content {
                    println!("{indent}  {}", code_line.cyan());
                }
                code_block_content.clear();
                in_code_block = false;
            } else {
                in_code_block = true;
            }
            continue;
        }

        if in_code_block {
            code_block_content.push(line.to_string());
            continue;
        }

        let processed = render_inline_markdown(line);

        for wrapped_line in textwrap::wrap(&processed, width.saturating_sub(indent.len())) {
            println!("{indent}{wrapped_line}");
        }
    }

    if in_code_block {
        for code_line in &code_block_content {
            println!("{indent}  {}", code_line.cyan());
        }
    }
}

/// Process inline markdown: `code`, **bold**, *italic*
pub fn render_inline_markdown(text: &str) -> String {
    let mut result = String::new();
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '`' {
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

pub fn print_colored(result: &ErrorExplanation) {
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

pub fn print_stats(stats: &InferenceStats) {
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

pub fn print_debug_section(title: &str, body: &str, footer: Option<String>) {
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

/// Print parsed stack trace frames as a formatted table
pub fn print_frames(trace: &StackTrace) {
    println!();
    println!("{} {}", "▸".cyan(), "Parsed Stack Trace".cyan().bold());
    println!(
        "  {} {} ({})",
        "Language:".blue().bold(),
        trace.language.to_string().bright_white(),
        if !trace.error_type.is_empty() {
            trace.error_type.clone()
        } else {
            "unknown error type".to_string()
        }
    );
    if !trace.error_message.is_empty() {
        println!(
            "  {} {}",
            "Message:".blue().bold(),
            trace.error_message.bright_white()
        );
    }
    println!();

    if trace.frames.is_empty() {
        println!("  {}", "No frames parsed.".dimmed());
        return;
    }

    println!(
        "  {} {:^4} {:40} {:30} {:>6}",
        "".normal(),
        "#".dimmed(),
        "Function".dimmed(),
        "File".dimmed(),
        "Line".dimmed()
    );
    println!("  {}", "─".repeat(85).dimmed());

    for (i, frame) in trace.frames.iter().enumerate() {
        let marker = if frame.is_user_code { ">" } else { " " };
        let marker_colored = if frame.is_user_code {
            marker.green().bold()
        } else {
            marker.normal()
        };

        let function = frame.function.as_deref().unwrap_or("<unknown>").to_string();
        let function_display = if function.len() > 38 {
            format!("{}...", &function[..35])
        } else {
            function
        };

        let file = frame
            .file
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "<unknown>".to_string());
        let file_display = if file.len() > 28 {
            format!("...{}", &file[file.len().saturating_sub(25)..])
        } else {
            file.clone()
        };

        let line = frame
            .line
            .map(|l| l.to_string())
            .unwrap_or_else(|| "-".to_string());

        let row = format!(
            "  {} {:>4} {:40} {:30} {:>6}",
            marker_colored,
            i + 1,
            function_display,
            file_display,
            line
        );

        if frame.is_user_code {
            println!("{}", row.green());
        } else {
            println!("{}", row);
        }
    }

    println!();

    if let Some(root_frame) = trace.root_cause_frame() {
        let file_line = match (&root_frame.file, root_frame.line) {
            (Some(file), Some(line)) => format!("{}:{}", file.display(), line),
            (Some(file), None) => file.display().to_string(),
            (None, Some(line)) => format!("line {}", line),
            (None, None) => "unknown location".to_string(),
        };

        println!(
            "  {} {} at {}",
            "Root cause:".yellow().bold(),
            root_frame
                .function
                .as_deref()
                .unwrap_or("<unknown>")
                .bright_white(),
            file_line.cyan()
        );
        println!();
    }
}

/// Format a file:line location with color highlighting for terminal output
pub fn format_file_line(file: &Path, line: Option<u32>, column: Option<u32>) -> String {
    let mut result = file.display().to_string().cyan().to_string();
    if let Some(l) = line {
        result.push_str(&format!(":{}", l.to_string().yellow()));
        if let Some(c) = column {
            result.push_str(&format!(":{}", c.to_string().yellow()));
        }
    }
    result
}

/// Check if a string contains common error patterns
pub fn contains_error_patterns(text: &str) -> bool {
    let error_patterns = [
        "error",
        "Error",
        "ERROR",
        "exception",
        "Exception",
        "EXCEPTION",
        "failed",
        "Failed",
        "FAILED",
        "panic",
        "Panic",
        "PANIC",
        "traceback",
        "Traceback",
        "stack trace",
        "undefined",
        "null pointer",
        "Null pointer",
        "segmentation fault",
        "Segmentation fault",
        "Segmentation Fault",
        "segfault",
        "SIGSEGV",
        "SIGABRT",
        "SIGBUS",
    ];

    error_patterns.iter().any(|pattern| text.contains(pattern))
}

/// Interpret exit codes with human-readable descriptions
pub fn interpret_exit_code(code: i32) -> &'static str {
    match code {
        0 => "Success",
        1 => "General error",
        2 => "Misuse of shell command",
        126 => "Command cannot execute (permission denied)",
        127 => "Command not found",
        128 => "Invalid exit argument",
        130 => "Terminated by Ctrl+C (SIGINT)",
        137 => "Killed (SIGKILL)",
        139 => "Segmentation fault (SIGSEGV)",
        141 => "Broken pipe (SIGPIPE)",
        143 => "Terminated (SIGTERM)",
        _ if code > 128 && code < 165 => match code - 128 {
            9 => "Killed (SIGKILL)",
            11 => "Segmentation fault (SIGSEGV)",
            13 => "Broken pipe (SIGPIPE)",
            15 => "Terminated (SIGTERM)",
            _ => "Signal received",
        },
        _ => "Unknown error",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_parse_response_multiline_explanation() {
        let response = "SUMMARY: Compilation failed.\n\
            EXPLANATION: The compiler found multiple errors:\n\
            - Type mismatch on line 10\n\
            - Missing semicolon on line 15\n\
            - Undefined variable on line 20\n\
            SUGGESTION: Fix each error in order.";

        let result = parse_response("compile error", response);

        assert_eq!(result.summary, "Compilation failed.");
        assert!(result.explanation.contains("Type mismatch"));
        assert!(result.explanation.contains("Missing semicolon"));
        assert!(result.explanation.contains("Undefined variable"));
    }

    #[test]
    fn test_parse_response_stack_trace() {
        let response = "SUMMARY: Null pointer exception in Java.\n\
            EXPLANATION: The stack trace shows:\n\
            at com.example.Main.process(Main.java:42)\n\
            at com.example.Main.main(Main.java:10)\n\
            The null reference originated in the process method.\n\
            SUGGESTION: Add null checks before calling methods on objects.";

        let result = parse_response("java.lang.NullPointerException", response);

        assert!(result.explanation.contains("Main.java:42"));
        assert!(result.explanation.contains("null reference"));
    }

    #[test]
    fn test_parse_response_real_rust_error() {
        let response = "SUMMARY: Ownership violation - value used after move.\n\
            EXPLANATION: In Rust, when you assign a value to another variable or pass it to a function, \
            ownership is transferred (moved). The original variable can no longer be used. \
            This error occurs at src/main.rs:10:5 where you tried to use 'x' after it was moved.\n\
            SUGGESTION: Consider using .clone() to create a copy, or borrow the value with & instead of moving it.";

        let result = parse_response("error[E0382]: borrow of moved value", response);

        assert!(result.summary.contains("Ownership violation"));
        assert!(result.explanation.contains("ownership is transferred"));
        assert!(result.suggestion.contains("clone()"));
    }

    #[test]
    fn test_parse_response_many_newlines() {
        let response = "SUMMARY: Test.\n\n\n\nEXPLANATION: Details.\n\n\nSUGGESTION: Fix.";
        let result = parse_response("error", response);

        assert_eq!(result.summary, "Test.");
        assert_eq!(result.explanation, "Details.");
        assert_eq!(result.suggestion, "Fix.");
    }

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

    #[test]
    fn test_interpret_exit_code_general_error() {
        assert_eq!(interpret_exit_code(1), "General error");
    }

    #[test]
    fn test_interpret_exit_code_command_not_found() {
        assert_eq!(interpret_exit_code(127), "Command not found");
    }

    #[test]
    fn test_interpret_exit_code_permission_denied() {
        assert!(interpret_exit_code(126).contains("permission"));
    }

    #[test]
    fn test_interpret_exit_code_ctrl_c() {
        assert!(interpret_exit_code(130).contains("Ctrl+C"));
    }

    #[test]
    fn test_interpret_exit_code_sigkill() {
        assert!(interpret_exit_code(137).contains("Kill"));
    }

    #[test]
    fn test_interpret_exit_code_sigsegv() {
        assert!(interpret_exit_code(139).contains("Segmentation fault"));
    }

    #[test]
    fn test_interpret_exit_code_unknown() {
        assert_eq!(interpret_exit_code(42), "Unknown error");
    }
}
