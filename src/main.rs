use anyhow::{bail, Context, Result};
use clap::{CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::{generate, Shell};
use colored::Colorize;
use crossterm::event::{self, Event, KeyCode, KeyModifiers};
use crossterm::terminal;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{send_logs_to_tracing, LogOptions};
use notify::{
    Config as NotifyConfig, Event as NotifyEvent, RecommendedWatcher, RecursiveMode, Watcher,
};
use regex::Regex;
use serde::Deserialize;
use serde::Serialize;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::env;
use std::fmt;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, BufReader, IsTerminal, Read, Seek, SeekFrom, Write};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

// Unix-specific imports for daemon mode
#[cfg(unix)]
use std::os::unix::net::{UnixListener, UnixStream};

// ============================================================================
// Stack Trace Intelligence (Feature 3)
// ============================================================================

/// Supported programming languages for stack trace parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Python,
    Rust,
    JavaScript,
    Go,
    Java,
    Cpp,
    Unknown,
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Language::Python => write!(f, "python"),
            Language::Rust => write!(f, "rust"),
            Language::JavaScript => write!(f, "javascript"),
            Language::Go => write!(f, "go"),
            Language::Java => write!(f, "java"),
            Language::Cpp => write!(f, "c/c++"),
            Language::Unknown => write!(f, "unknown"),
        }
    }
}

/// A single frame in a stack trace
#[derive(Debug, Clone, Serialize)]
pub struct StackFrame {
    /// Function or method name (if available)
    pub function: Option<String>,
    /// Source file path (if available)
    pub file: Option<PathBuf>,
    /// Line number in the source file (if available)
    pub line: Option<u32>,
    /// Column number in the source file (if available)
    pub column: Option<u32>,
    /// Whether this frame is user code (vs framework/stdlib)
    pub is_user_code: bool,
    /// Additional context for this frame (e.g., source snippet)
    pub context: Option<String>,
}

impl StackFrame {
    /// Create a new stack frame with all fields set to None/default
    pub fn new() -> Self {
        Self {
            function: None,
            file: None,
            line: None,
            column: None,
            is_user_code: true, // Assume user code by default
            context: None,
        }
    }

    /// Create a stack frame with the given function name
    pub fn with_function(mut self, function: impl Into<String>) -> Self {
        self.function = Some(function.into());
        self
    }

    /// Create a stack frame with the given file path
    pub fn with_file(mut self, file: impl Into<PathBuf>) -> Self {
        self.file = Some(file.into());
        self
    }

    /// Create a stack frame with the given line number
    pub fn with_line(mut self, line: u32) -> Self {
        self.line = Some(line);
        self
    }

    /// Create a stack frame with the given column number
    pub fn with_column(mut self, column: u32) -> Self {
        self.column = Some(column);
        self
    }

    /// Set whether this frame is user code
    pub fn with_is_user_code(mut self, is_user_code: bool) -> Self {
        self.is_user_code = is_user_code;
        self
    }

    /// Set the context for this frame
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

impl Default for StackFrame {
    fn default() -> Self {
        Self::new()
    }
}

/// A parsed stack trace from any supported language
#[derive(Debug, Clone, Serialize)]
pub struct StackTrace {
    /// The programming language of this stack trace
    pub language: Language,
    /// The type of error/exception (e.g., "KeyError", "NullPointerException")
    pub error_type: String,
    /// The error message
    pub error_message: String,
    /// Stack frames, ordered from innermost (most recent) to outermost
    pub frames: Vec<StackFrame>,
    /// The raw text of the stack trace
    pub raw_text: String,
}

impl StackTrace {
    /// Create a new stack trace
    pub fn new(language: Language, raw_text: impl Into<String>) -> Self {
        Self {
            language,
            error_type: String::new(),
            error_message: String::new(),
            frames: Vec::new(),
            raw_text: raw_text.into(),
        }
    }

    /// Set the error type
    pub fn with_error_type(mut self, error_type: impl Into<String>) -> Self {
        self.error_type = error_type.into();
        self
    }

    /// Set the error message
    pub fn with_error_message(mut self, error_message: impl Into<String>) -> Self {
        self.error_message = error_message.into();
        self
    }

    /// Add a frame to the stack trace
    pub fn add_frame(&mut self, frame: StackFrame) {
        self.frames.push(frame);
    }

    /// Get the most relevant frame (first user code frame, or first frame if none)
    pub fn root_cause_frame(&self) -> Option<&StackFrame> {
        self.frames
            .iter()
            .find(|f| f.is_user_code)
            .or_else(|| self.frames.first())
    }

    /// Get only user code frames
    pub fn user_frames(&self) -> Vec<&StackFrame> {
        self.frames.iter().filter(|f| f.is_user_code).collect()
    }
}

/// Trait for language-specific stack trace parsers
pub trait StackTraceParser: Send + Sync {
    /// The language this parser handles
    fn language(&self) -> Language;

    /// Check if this parser can parse the given input
    fn can_parse(&self, input: &str) -> bool;

    /// Parse the input into a StackTrace
    /// Returns None if parsing fails
    fn parse(&self, input: &str) -> Option<StackTrace>;
}

/// Registry for stack trace parsers with auto-detection
pub struct StackTraceParserRegistry {
    parsers: Vec<Box<dyn StackTraceParser>>,
}

impl StackTraceParserRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            parsers: Vec::new(),
        }
    }

    /// Create a registry with all built-in parsers
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        registry.register(Box::new(PythonStackTraceParser));
        registry.register(Box::new(RustStackTraceParser));
        registry.register(Box::new(JavaScriptStackTraceParser));
        registry.register(Box::new(GoStackTraceParser));
        registry.register(Box::new(JavaStackTraceParser));
        registry.register(Box::new(CppStackTraceParser));
        registry
    }

    /// Register a parser
    pub fn register(&mut self, parser: Box<dyn StackTraceParser>) {
        self.parsers.push(parser);
    }

    /// Get a parser by language
    pub fn get_parser(&self, language: Language) -> Option<&dyn StackTraceParser> {
        self.parsers
            .iter()
            .find(|p| p.language() == language)
            .map(|p| p.as_ref())
    }

    /// Auto-detect the language and parse the input
    pub fn parse(&self, input: &str) -> Option<StackTrace> {
        // Try each parser in order
        for parser in &self.parsers {
            if parser.can_parse(input) {
                if let Some(trace) = parser.parse(input) {
                    return Some(trace);
                }
            }
        }
        None
    }

    /// Detect the language of a stack trace without fully parsing
    pub fn detect_language(&self, input: &str) -> Language {
        for parser in &self.parsers {
            if parser.can_parse(input) {
                return parser.language();
            }
        }
        Language::Unknown
    }
}

impl Default for StackTraceParserRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

// ============================================================================
// Language-Specific Parsers (Phase 3.2)
// ============================================================================

/// Python stack trace parser
/// Parses Python tracebacks including:
/// - Standard "Traceback (most recent call last):" format
/// - File/line/function extraction
/// - Exception type and message
/// - Chained exceptions ("During handling of...")
struct PythonStackTraceParser;

impl PythonStackTraceParser {
    /// Parse a Python exception line like "KeyError: 'missing_key'" or "ValueError: invalid literal"
    fn parse_exception_line(line: &str) -> Option<(String, String)> {
        // Common Python exception patterns
        let exception_patterns = ["Error:", "Exception:", "Warning:", "Interrupt:"];

        for pattern in exception_patterns {
            if let Some(pos) = line.find(pattern) {
                // Find the exception type (word before pattern)
                let before = &line[..pos + pattern.len() - 1]; // Include up to "Error" etc
                let error_type = before
                    .split_whitespace()
                    .last()
                    .unwrap_or(before)
                    .to_string();
                let message = line[pos + pattern.len()..].trim().to_string();
                return Some((error_type, message));
            }
        }

        // Also handle bare exception names like "KeyError" or "ZeroDivisionError"
        let trimmed = line.trim();
        if trimmed
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false)
            && !trimmed.contains(' ')
            && (trimmed.ends_with("Error")
                || trimmed.ends_with("Exception")
                || trimmed.ends_with("Warning")
                || trimmed.ends_with("Interrupt"))
        {
            return Some((trimmed.to_string(), String::new()));
        }

        None
    }

    /// Parse a File line like `  File "path/to/file.py", line 42, in function_name`
    fn parse_file_line(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();
        if !trimmed.starts_with("File \"") {
            return None;
        }

        let mut frame = StackFrame::new();

        // Extract file path: File "path" ...
        if let Some(start) = trimmed.find('"') {
            if let Some(end) = trimmed[start + 1..].find('"') {
                let file_path = &trimmed[start + 1..start + 1 + end];
                frame.file = Some(PathBuf::from(file_path));

                // Check if it's user code
                frame.is_user_code = !Self::is_framework_path(file_path);
            }
        }

        // Extract line number: , line N, ...
        if let Some(line_pos) = trimmed.find(", line ") {
            let after_line = &trimmed[line_pos + 7..];
            let line_end = after_line
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(after_line.len());
            if let Ok(line_num) = after_line[..line_end].parse::<u32>() {
                frame.line = Some(line_num);
            }
        }

        // Extract function name: , in function_name
        if let Some(in_pos) = trimmed.find(", in ") {
            let func_name = trimmed[in_pos + 5..].trim();
            if !func_name.is_empty() {
                frame.function = Some(func_name.to_string());
            }
        }

        Some(frame)
    }

    /// Check if a path is a framework/stdlib path
    fn is_framework_path(path: &str) -> bool {
        let path_lower = path.to_lowercase();
        path_lower.contains("site-packages")
            || path_lower.contains("/usr/lib/python")
            || path_lower.contains("/lib/python")
            || path_lower.contains("\\lib\\python")
            || path_lower.contains("<frozen ")
            || path_lower.contains("<string>")
            || path_lower.starts_with("<")
    }
}

impl StackTraceParser for PythonStackTraceParser {
    fn language(&self) -> Language {
        Language::Python
    }

    fn can_parse(&self, input: &str) -> bool {
        input.contains("Traceback (most recent call last):")
            || input.contains("File \"") && (input.contains(", line ") || input.contains("Error:"))
    }

    fn parse(&self, input: &str) -> Option<StackTrace> {
        if !self.can_parse(input) {
            return None;
        }

        let mut trace = StackTrace::new(Language::Python, input);
        let mut lines = input.lines().peekable();
        let mut in_traceback = false;

        while let Some(line) = lines.next() {
            let trimmed = line.trim();

            // Start of traceback
            if trimmed == "Traceback (most recent call last):" {
                in_traceback = true;
                continue;
            }

            // Chained exception markers
            if trimmed.starts_with("During handling of the above exception")
                || trimmed.starts_with("The above exception was the direct cause")
            {
                // Continue parsing the chained exception
                in_traceback = true;
                continue;
            }

            if in_traceback {
                // File line
                if let Some(mut frame) = Self::parse_file_line(line) {
                    // Check for code context on next line
                    if let Some(next_line) = lines.peek() {
                        let next_trimmed = next_line.trim();
                        // Code line is indented but doesn't start with "File"
                        if !next_trimmed.starts_with("File")
                            && !next_trimmed.is_empty()
                            && Self::parse_exception_line(next_trimmed).is_none()
                        {
                            frame.context = Some(next_trimmed.to_string());
                            lines.next(); // Consume the context line
                        }
                    }
                    trace.add_frame(frame);
                    continue;
                }

                // Exception line (usually the last line of traceback)
                if let Some((error_type, message)) = Self::parse_exception_line(trimmed) {
                    trace.error_type = error_type;
                    trace.error_message = message;
                    in_traceback = false;
                    continue;
                }
            } else {
                // Not in traceback - check for standalone exception
                if let Some((error_type, message)) = Self::parse_exception_line(trimmed) {
                    if trace.error_type.is_empty() {
                        trace.error_type = error_type;
                        trace.error_message = message;
                    }
                }
            }
        }

        // If we have frames but no error type, try to extract from raw text
        if trace.error_type.is_empty() && !trace.frames.is_empty() {
            // Look for exception-like pattern at the end
            for line in input.lines().rev() {
                if let Some((error_type, message)) = Self::parse_exception_line(line.trim()) {
                    trace.error_type = error_type;
                    trace.error_message = message;
                    break;
                }
            }
        }

        Some(trace)
    }
}

/// Rust stack trace parser
/// Parses Rust panics and compiler errors including:
/// - "thread 'main' panicked at" format
/// - RUST_BACKTRACE=1 full backtraces
/// - Rust compiler error[E####] format
struct RustStackTraceParser;

impl RustStackTraceParser {
    /// Parse a panic line like `thread 'main' panicked at 'message', src/main.rs:10:5`
    fn parse_panic_line(line: &str) -> Option<(String, Option<StackFrame>)> {
        // Pattern: thread 'name' panicked at 'message', file:line:col
        if !line.contains("panicked at") {
            return None;
        }

        let mut message = String::new();
        let mut frame = None;

        // Extract panic message
        // New format (Rust 1.65+): thread 'main' panicked at file:line:col:
        // Old format: thread 'main' panicked at 'message', file:line:col
        if let Some(at_pos) = line.find("panicked at ") {
            let after_at = &line[at_pos + 12..];

            // Check for old format with quoted message
            if after_at.starts_with('\'') {
                if let Some(end_quote) = after_at[1..].find('\'') {
                    message = after_at[1..end_quote + 1].to_string();
                    // Parse location after the message
                    let after_msg = &after_at[end_quote + 2..];
                    if let Some(loc) = after_msg.strip_prefix(", ") {
                        frame = Self::parse_location(loc);
                    }
                }
            } else {
                // New format - location comes first, then message on next line
                frame = Self::parse_location(after_at.trim_end_matches(':'));
            }
        }

        Some((message, frame))
    }

    /// Parse a location like `src/main.rs:10:5`
    fn parse_location(loc: &str) -> Option<StackFrame> {
        let parts: Vec<&str> = loc.rsplitn(3, ':').collect();
        if parts.len() >= 2 {
            let mut frame = StackFrame::new();

            // Parts are reversed: [col, line, file] or [line, file]
            if parts.len() >= 2 {
                if let Ok(line) = parts[0].trim().parse::<u32>() {
                    frame.line = Some(line);
                    let file_parts: Vec<&str> = parts[1..].iter().rev().copied().collect();
                    frame.file = Some(PathBuf::from(file_parts.join(":")));
                } else if let Ok(line) = parts[1].trim().parse::<u32>() {
                    frame.line = Some(line);
                    if let Ok(col) = parts[0].trim().parse::<u32>() {
                        frame.column = Some(col);
                    }
                    if parts.len() > 2 {
                        frame.file = Some(PathBuf::from(parts[2]));
                    }
                }
            }

            // Check if user code
            if let Some(ref file) = frame.file {
                frame.is_user_code = !Self::is_framework_path(&file.to_string_lossy());
            }

            return Some(frame);
        }
        None
    }

    /// Parse a backtrace frame like `   0: rust_panic` or `   1: std::panicking::begin_panic`
    fn parse_backtrace_frame(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();

        // Pattern: N: function_name
        if let Some(colon_pos) = trimmed.find(':') {
            let num_part = trimmed[..colon_pos].trim();
            if num_part.chars().all(|c| c.is_ascii_digit()) {
                let func_part = trimmed[colon_pos + 1..].trim();
                let mut frame = StackFrame::new().with_function(func_part);
                frame.is_user_code = !Self::is_framework_path(func_part);
                return Some(frame);
            }
        }

        // Pattern: at path:line:col
        if let Some(loc) = trimmed.strip_prefix("at ") {
            return Self::parse_location(loc);
        }

        None
    }

    /// Parse compiler error format: error[E0382]: message
    fn parse_compiler_error(line: &str) -> Option<(String, String)> {
        // Pattern: error[E####]: message
        if let Some(bracket_pos) = line.find("[E") {
            if let Some(end_bracket) = line[bracket_pos..].find(']') {
                let error_code = &line[bracket_pos + 1..bracket_pos + end_bracket];
                let after = &line[bracket_pos + end_bracket + 1..];
                let message = after.strip_prefix(": ").unwrap_or(after).trim();
                return Some((format!("error[{}]", error_code), message.to_string()));
            }
        }
        None
    }

    /// Parse source location line: --> src/main.rs:10:5
    fn parse_source_location(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();
        if trimmed.starts_with("-->") {
            let loc = trimmed[3..].trim();
            return Self::parse_location(loc);
        }
        None
    }

    /// Check if a path/function is a framework/stdlib path
    fn is_framework_path(path: &str) -> bool {
        path.contains(".cargo")
            || path.contains("/rustc/")
            || path.starts_with("std::")
            || path.starts_with("core::")
            || path.starts_with("alloc::")
            || path.starts_with("panic_unwind")
            || path.starts_with("rust_panic")
            || path.starts_with("<alloc::")
            || path.starts_with("<core::")
    }
}

impl StackTraceParser for RustStackTraceParser {
    fn language(&self) -> Language {
        Language::Rust
    }

    fn can_parse(&self, input: &str) -> bool {
        input.contains("thread '") && input.contains("panicked at")
            || input.contains("stack backtrace:")
            || input.contains("error[E") && input.contains("-->")
    }

    fn parse(&self, input: &str) -> Option<StackTrace> {
        if !self.can_parse(input) {
            return None;
        }

        let mut trace = StackTrace::new(Language::Rust, input);
        let mut in_backtrace = false;

        for line in input.lines() {
            let trimmed = line.trim();

            // Parse panic line
            if let Some((message, frame)) = Self::parse_panic_line(trimmed) {
                if !message.is_empty() {
                    trace.error_message = message;
                }
                trace.error_type = "panic".to_string();
                if let Some(f) = frame {
                    trace.add_frame(f);
                }
                continue;
            }

            // Parse compiler error
            if let Some((error_type, message)) = Self::parse_compiler_error(trimmed) {
                trace.error_type = error_type;
                trace.error_message = message;
                continue;
            }

            // Source location for compiler errors
            if let Some(frame) = Self::parse_source_location(trimmed) {
                trace.add_frame(frame);
                continue;
            }

            // Backtrace start
            if trimmed == "stack backtrace:" {
                in_backtrace = true;
                continue;
            }

            // Parse backtrace frames
            if in_backtrace {
                if let Some(frame) = Self::parse_backtrace_frame(trimmed) {
                    trace.add_frame(frame);
                }
            }
        }

        Some(trace)
    }
}

/// JavaScript/Node.js stack trace parser
/// Parses JavaScript errors including:
/// - Node.js format: "Error: msg\n    at func (file:line:col)"
/// - Browser format: "Error: msg\n    at func (file:line:col)"
/// - Async stack traces
struct JavaScriptStackTraceParser;

impl JavaScriptStackTraceParser {
    /// Parse error type and message from first line like "TypeError: Cannot read property"
    fn parse_error_line(line: &str) -> Option<(String, String)> {
        let error_types = [
            "Error",
            "TypeError",
            "ReferenceError",
            "SyntaxError",
            "RangeError",
            "URIError",
            "EvalError",
            "AggregateError",
            "InternalError",
            "AssertionError",
        ];

        for error_type in error_types {
            if let Some(rest) = line.strip_prefix(error_type) {
                if rest.starts_with(':') {
                    return Some((error_type.to_string(), rest[1..].trim().to_string()));
                } else if rest.is_empty()
                    || rest
                        .chars()
                        .next()
                        .map(|c| c.is_whitespace())
                        .unwrap_or(false)
                {
                    return Some((error_type.to_string(), rest.trim().to_string()));
                }
            }
        }
        None
    }

    /// Parse a stack frame line like "    at functionName (file:line:col)"
    /// or "    at file:line:col"
    fn parse_frame_line(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();
        if !trimmed.starts_with("at ") {
            return None;
        }

        let content = &trimmed[3..].trim();
        let mut frame = StackFrame::new();

        // Pattern 1: "at functionName (file:line:col)"
        if let Some(paren_start) = content.find(" (") {
            if let Some(paren_end) = content.rfind(')') {
                let func_name = &content[..paren_start];
                frame.function = Some(func_name.to_string());

                let location = &content[paren_start + 2..paren_end];
                Self::parse_location(location, &mut frame);
            }
        }
        // Pattern 2: "at file:line:col" (anonymous)
        else if content.contains(':') {
            Self::parse_location(content, &mut frame);
        }
        // Pattern 3: "at functionName" (no location)
        else {
            frame.function = Some(content.to_string());
        }

        // Check if user code
        frame.is_user_code = !Self::is_framework_path(&frame);

        Some(frame)
    }

    /// Parse location string like "file:line:col" or "/path/to/file.js:10:5"
    fn parse_location(loc: &str, frame: &mut StackFrame) {
        // Handle file:line:col format
        let parts: Vec<&str> = loc.rsplitn(3, ':').collect();

        if parts.len() >= 2 {
            // Try to parse last two parts as numbers
            if let Ok(col) = parts[0].trim().parse::<u32>() {
                frame.column = Some(col);
                if let Ok(line) = parts[1].trim().parse::<u32>() {
                    frame.line = Some(line);
                    if parts.len() > 2 {
                        frame.file = Some(PathBuf::from(parts[2]));
                    }
                }
            } else if let Ok(line) = parts[0].trim().parse::<u32>() {
                frame.line = Some(line);
                if parts.len() > 1 {
                    let file_parts: Vec<&str> = parts[1..].iter().rev().copied().collect();
                    frame.file = Some(PathBuf::from(file_parts.join(":")));
                }
            }
        }
    }

    /// Check if a frame is framework/node_modules code
    fn is_framework_path(frame: &StackFrame) -> bool {
        let path = frame
            .file
            .as_ref()
            .map(|p| p.to_string_lossy().to_lowercase())
            .unwrap_or_default();

        let func = frame
            .function
            .as_ref()
            .map(|f| f.to_lowercase())
            .unwrap_or_default();

        path.contains("node_modules")
            || path.contains("internal/")
            || path.starts_with("node:")
            || path.starts_with("internal:")
            || func.starts_with("native ")
            || func.contains("<anonymous>")
            || func == "module.load"
            || func == "module._compile"
    }
}

impl StackTraceParser for JavaScriptStackTraceParser {
    fn language(&self) -> Language {
        Language::JavaScript
    }

    fn can_parse(&self, input: &str) -> bool {
        // Node.js style: "Error: message\n    at function (file:line:col)"
        // Browser style: "Error: message\n    at function (file:line:col)"
        (input.contains("Error:")
            || input.contains("TypeError:")
            || input.contains("ReferenceError:"))
            && input.contains("    at ")
    }

    fn parse(&self, input: &str) -> Option<StackTrace> {
        if !self.can_parse(input) {
            return None;
        }

        let mut trace = StackTrace::new(Language::JavaScript, input);

        for line in input.lines() {
            let trimmed = line.trim();

            // Parse error line
            if trace.error_type.is_empty() {
                if let Some((error_type, message)) = Self::parse_error_line(trimmed) {
                    trace.error_type = error_type;
                    trace.error_message = message;
                    continue;
                }
            }

            // Parse stack frame
            if let Some(frame) = Self::parse_frame_line(line) {
                trace.add_frame(frame);
            }
        }

        Some(trace)
    }
}

/// Go stack trace parser
/// Parses Go panics including:
/// - "panic: message" format
/// - goroutine sections
/// - file.go:line format
struct GoStackTraceParser;

impl GoStackTraceParser {
    /// Parse panic line like "panic: runtime error: index out of range"
    fn parse_panic_line(line: &str) -> Option<String> {
        if line.starts_with("panic:") {
            return Some(line[6..].trim().to_string());
        }
        None
    }

    /// Parse a stack frame line like "main.go:10 +0x1a"
    /// or "main.main()" on the function line
    fn parse_frame_line(line: &str, prev_function: &mut Option<String>) -> Option<StackFrame> {
        let trimmed = line.trim();

        // Function line like "main.main()" or "runtime.gopanic(...)"
        if trimmed.ends_with(')') && trimmed.contains('(') {
            let paren_pos = trimmed.find('(').unwrap();
            *prev_function = Some(trimmed[..paren_pos].to_string());
            return None;
        }

        // Location line like "/path/to/file.go:10 +0x1a"
        if trimmed.contains(".go:") {
            let mut frame = StackFrame::new();

            // Set the function from previous line
            if let Some(ref func) = prev_function {
                frame.function = Some(func.clone());
                frame.is_user_code = !Self::is_framework_function(func);
            }
            *prev_function = None;

            // Parse file:line
            if let Some(colon_pos) = trimmed.rfind(':') {
                let file_part = &trimmed[..colon_pos];
                let after_colon = &trimmed[colon_pos + 1..];

                // Line number is before any space
                let line_end = after_colon.find(' ').unwrap_or(after_colon.len());
                if let Ok(line_num) = after_colon[..line_end].parse::<u32>() {
                    frame.line = Some(line_num);
                    frame.file = Some(PathBuf::from(file_part));
                }
            }

            return Some(frame);
        }

        None
    }

    /// Check if a function is a runtime/framework function
    fn is_framework_function(func: &str) -> bool {
        func.starts_with("runtime.")
            || func.starts_with("syscall.")
            || func.starts_with("internal/")
            || func.starts_with("reflect.")
            || func.starts_with("testing.")
    }
}

impl StackTraceParser for GoStackTraceParser {
    fn language(&self) -> Language {
        Language::Go
    }

    fn can_parse(&self, input: &str) -> bool {
        input.contains("panic:") && input.contains("goroutine ")
            || input.contains(".go:") && input.contains("runtime.")
    }

    fn parse(&self, input: &str) -> Option<StackTrace> {
        if !self.can_parse(input) {
            return None;
        }

        let mut trace = StackTrace::new(Language::Go, input);
        let mut prev_function: Option<String> = None;

        for line in input.lines() {
            let trimmed = line.trim();

            // Parse panic line
            if trace.error_message.is_empty() {
                if let Some(message) = Self::parse_panic_line(trimmed) {
                    trace.error_type = "panic".to_string();
                    trace.error_message = message;
                    continue;
                }
            }

            // Skip goroutine header lines
            if trimmed.starts_with("goroutine ") {
                continue;
            }

            // Parse stack frames
            if let Some(frame) = Self::parse_frame_line(line, &mut prev_function) {
                trace.add_frame(frame);
            }
        }

        Some(trace)
    }
}

/// Java/JVM stack trace parser
/// Parses Java exceptions including:
/// - "Exception in thread" format
/// - "at package.Class.method(File.java:line)" frames
/// - "Caused by:" chained exceptions
struct JavaStackTraceParser;

impl JavaStackTraceParser {
    /// Parse exception line like "java.lang.NullPointerException: message"
    /// or "Exception in thread \"main\" java.lang.RuntimeException: msg"
    fn parse_exception_line(line: &str) -> Option<(String, String)> {
        let trimmed = line.trim();

        // Pattern: Exception in thread "name" ExceptionType: message
        if trimmed.starts_with("Exception in thread") {
            if let Some(quote_end) = trimmed[21..].find('"') {
                let after_thread = &trimmed[21 + quote_end + 1..].trim();
                return Self::parse_exception_type(after_thread);
            }
        }

        // Pattern: Caused by: ExceptionType: message
        if trimmed.starts_with("Caused by:") {
            let after = &trimmed[10..].trim();
            return Self::parse_exception_type(after);
        }

        // Pattern: ExceptionType: message (at start of trace)
        if trimmed.contains("Exception") || trimmed.contains("Error") {
            return Self::parse_exception_type(trimmed);
        }

        None
    }

    /// Parse exception type and message from "java.lang.NullPointerException: msg"
    fn parse_exception_type(text: &str) -> Option<(String, String)> {
        if let Some(colon_pos) = text.find(':') {
            let error_type = text[..colon_pos].trim().to_string();
            let message = text[colon_pos + 1..].trim().to_string();
            return Some((error_type, message));
        }
        // No message, just exception type
        let error_type = text.split_whitespace().next()?.to_string();
        if error_type.contains('.')
            || error_type.ends_with("Exception")
            || error_type.ends_with("Error")
        {
            return Some((error_type, String::new()));
        }
        None
    }

    /// Parse stack frame like "at com.example.Main.process(Main.java:42)"
    fn parse_frame_line(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();

        // Must start with "at "
        if !trimmed.starts_with("at ") {
            return None;
        }

        let content = &trimmed[3..];
        let mut frame = StackFrame::new();

        // Pattern: package.Class.method(File.java:line)
        if let Some(paren_start) = content.find('(') {
            if let Some(paren_end) = content.find(')') {
                let method_full = &content[..paren_start];
                frame.function = Some(method_full.to_string());
                frame.is_user_code = !Self::is_framework_class(method_full);

                // Parse (File.java:line)
                let location = &content[paren_start + 1..paren_end];
                if let Some(colon_pos) = location.rfind(':') {
                    let file = &location[..colon_pos];
                    frame.file = Some(PathBuf::from(file));

                    if let Ok(line_num) = location[colon_pos + 1..].parse::<u32>() {
                        frame.line = Some(line_num);
                    }
                } else if location != "Native Method" && location != "Unknown Source" {
                    frame.file = Some(PathBuf::from(location));
                }
            }
        }

        Some(frame)
    }

    /// Check if a class is a framework/JDK class
    fn is_framework_class(method: &str) -> bool {
        method.starts_with("java.")
            || method.starts_with("javax.")
            || method.starts_with("sun.")
            || method.starts_with("jdk.")
            || method.starts_with("org.junit.")
            || method.starts_with("org.springframework.")
            || method.starts_with("org.apache.")
            || method.starts_with("com.google.")
    }
}

impl StackTraceParser for JavaStackTraceParser {
    fn language(&self) -> Language {
        Language::Java
    }

    fn can_parse(&self, input: &str) -> bool {
        // Java style: "Exception in thread" or "at package.Class.method(File.java:line)"
        input.contains("Exception in thread")
            || input.contains("at ") && input.contains(".java:")
            || input.contains("Caused by:")
    }

    fn parse(&self, input: &str) -> Option<StackTrace> {
        if !self.can_parse(input) {
            return None;
        }

        let mut trace = StackTrace::new(Language::Java, input);

        for line in input.lines() {
            // Parse exception line
            if trace.error_type.is_empty() {
                if let Some((error_type, message)) = Self::parse_exception_line(line) {
                    trace.error_type = error_type;
                    trace.error_message = message;
                    continue;
                }
            }

            // Parse "Caused by:" to update the root cause
            if line.trim().starts_with("Caused by:") {
                if let Some((error_type, message)) = Self::parse_exception_line(line) {
                    // Update to the root cause
                    trace.error_type = error_type;
                    trace.error_message = message;
                }
                continue;
            }

            // Parse stack frame
            if let Some(frame) = Self::parse_frame_line(line) {
                trace.add_frame(frame);
            }
        }

        Some(trace)
    }
}

/// C/C++ stack trace parser
/// Parses C/C++ errors including:
/// - gdb/lldb backtrace format
/// - AddressSanitizer output
/// - Segmentation faults
struct CppStackTraceParser;

impl CppStackTraceParser {
    /// Parse AddressSanitizer error like "AddressSanitizer: heap-buffer-overflow"
    fn parse_asan_line(line: &str) -> Option<(String, String)> {
        if line.contains("AddressSanitizer:") {
            if let Some(pos) = line.find("AddressSanitizer:") {
                let error_type = "AddressSanitizer".to_string();
                let message = line[pos + 17..].trim().to_string();
                return Some((error_type, message));
            }
        }
        None
    }

    /// Parse a gdb/lldb frame like "#0  0x00007fff main (argc=1) at main.cpp:10"
    fn parse_gdb_frame(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();

        // Must start with #N
        if !trimmed.starts_with('#') {
            return None;
        }

        // Extract frame number
        let after_hash = &trimmed[1..];
        let num_end = after_hash.find(|c: char| !c.is_ascii_digit()).unwrap_or(0);
        if num_end == 0 {
            return None;
        }

        let rest = after_hash[num_end..].trim();
        let mut frame = StackFrame::new();

        // Skip address like "0x00007fff"
        let content = if rest.starts_with("0x") {
            if let Some(space) = rest.find(' ') {
                rest[space..].trim()
            } else {
                rest
            }
        } else {
            rest
        };

        // Parse function name
        // Pattern: "function (args) at file:line" or "in function at file:line"
        let (func_part, location_part) = if let Some(at_pos) = content.find(" at ") {
            (&content[..at_pos], Some(&content[at_pos + 4..]))
        } else if let Some(from_pos) = content.find(" from ") {
            (&content[..from_pos], None)
        } else {
            (content, None)
        };

        // Extract function name
        let func_name = if let Some(in_pos) = func_part.find(" in ") {
            &func_part[in_pos + 4..]
        } else if func_part.starts_with("in ") {
            &func_part[3..]
        } else {
            func_part
        };

        // Remove arguments from function name
        let func_clean = if let Some(paren) = func_name.find('(') {
            &func_name[..paren]
        } else {
            func_name
        };

        if !func_clean.is_empty() {
            frame.function = Some(func_clean.trim().to_string());
            frame.is_user_code = !Self::is_framework_function(func_clean);
        }

        // Parse location
        if let Some(loc) = location_part {
            if let Some(colon) = loc.rfind(':') {
                let file = &loc[..colon];
                frame.file = Some(PathBuf::from(file.trim()));
                if let Ok(line_num) = loc[colon + 1..].trim().parse::<u32>() {
                    frame.line = Some(line_num);
                }
            } else {
                frame.file = Some(PathBuf::from(loc.trim()));
            }
        }

        Some(frame)
    }

    /// Parse AddressSanitizer frame like "#0 0x5555 in function file.cpp:10"
    fn parse_asan_frame(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();

        if !trimmed.starts_with('#') {
            return None;
        }

        let mut frame = StackFrame::new();

        // Pattern: #N 0xADDR in function_name file:line
        if let Some(in_pos) = trimmed.find(" in ") {
            let after_in = &trimmed[in_pos + 4..];

            // Split into function and location
            if let Some(space) = after_in.rfind(' ') {
                let func = &after_in[..space];
                let loc = &after_in[space + 1..];

                frame.function = Some(func.trim().to_string());
                frame.is_user_code = !Self::is_framework_function(func);

                // Parse location
                if let Some(colon) = loc.rfind(':') {
                    frame.file = Some(PathBuf::from(&loc[..colon]));
                    if let Ok(line_num) = loc[colon + 1..].parse::<u32>() {
                        frame.line = Some(line_num);
                    }
                }
            } else {
                frame.function = Some(after_in.trim().to_string());
            }
        }

        Some(frame)
    }

    /// Check if this is a signal/segfault line
    fn parse_signal_line(line: &str) -> Option<String> {
        if line.contains("Segmentation fault") {
            return Some("Segmentation fault (SIGSEGV)".to_string());
        }
        if line.contains("SIGSEGV") {
            return Some("Segmentation fault (SIGSEGV)".to_string());
        }
        if line.contains("SIGABRT") {
            return Some("Abort signal (SIGABRT)".to_string());
        }
        if line.contains("SIGBUS") {
            return Some("Bus error (SIGBUS)".to_string());
        }
        if line.contains("SIGFPE") {
            return Some("Floating point exception (SIGFPE)".to_string());
        }
        None
    }

    /// Check if a function is a framework/system function
    fn is_framework_function(func: &str) -> bool {
        func.starts_with("__")
            || func.starts_with("_start")
            || func.starts_with("libc")
            || func.starts_with("std::")
            || func.starts_with("__libc")
            || func.starts_with("pthread_")
            || func.contains("clone")
    }
}

impl StackTraceParser for CppStackTraceParser {
    fn language(&self) -> Language {
        Language::Cpp
    }

    fn can_parse(&self, input: &str) -> bool {
        // gdb/lldb backtrace or AddressSanitizer
        input.contains("#0 ") && (input.contains(" in ") || input.contains(" at "))
            || input.contains("AddressSanitizer:")
            || input.contains("Segmentation fault")
    }

    fn parse(&self, input: &str) -> Option<StackTrace> {
        if !self.can_parse(input) {
            return None;
        }

        let mut trace = StackTrace::new(Language::Cpp, input);

        for line in input.lines() {
            // Parse AddressSanitizer error
            if trace.error_type.is_empty() {
                if let Some((error_type, message)) = Self::parse_asan_line(line) {
                    trace.error_type = error_type;
                    trace.error_message = message;
                    continue;
                }
            }

            // Parse signal
            if trace.error_message.is_empty() {
                if let Some(signal) = Self::parse_signal_line(line) {
                    trace.error_type = "signal".to_string();
                    trace.error_message = signal;
                    continue;
                }
            }

            // Parse frames - try ASan format first, then gdb format
            if line.trim().starts_with('#') {
                if let Some(frame) = Self::parse_asan_frame(line) {
                    if frame.function.is_some() || frame.file.is_some() {
                        trace.add_frame(frame);
                        continue;
                    }
                }
                if let Some(frame) = Self::parse_gdb_frame(line) {
                    trace.add_frame(frame);
                }
            }
        }

        Some(trace)
    }
}

// ============================================================================
// Source Context Injection (Phase 3.4)
// ============================================================================

/// Configuration for source context extraction
#[derive(Debug, Clone)]
pub struct SourceContextConfig {
    /// Number of lines to show before and after the error line
    pub context_lines: usize,
    /// Root path for resolving relative file paths
    pub context_root: Option<PathBuf>,
    /// Maximum total characters of context to include
    pub max_context_chars: usize,
}

impl Default for SourceContextConfig {
    fn default() -> Self {
        Self {
            context_lines: 5,
            context_root: None,
            max_context_chars: 4096, // ~1024 tokens
        }
    }
}

// Stack trace prompt building functions - part of the planned enhanced prompt API
#[allow(dead_code)]
/// Extract source context for a stack frame
fn extract_frame_context(frame: &StackFrame, config: &SourceContextConfig) -> Option<String> {
    let file_path = frame.file.as_ref()?;
    let line_num = frame.line?;

    // Resolve the file path
    let resolved_path = resolve_source_path(file_path, &config.context_root)?;

    // Read the source file
    let contents = std::fs::read_to_string(&resolved_path).ok()?;
    let lines: Vec<&str> = contents.lines().collect();

    // Calculate line range (1-indexed to 0-indexed)
    let line_idx = line_num.saturating_sub(1) as usize;
    let start = line_idx.saturating_sub(config.context_lines);
    let end = (line_idx + config.context_lines + 1).min(lines.len());

    if start >= lines.len() {
        return None;
    }

    // Build context with line numbers
    let mut context = String::new();
    for (i, line) in lines[start..end].iter().enumerate() {
        let actual_line_num = start + i + 1;
        let marker = if actual_line_num == line_num as usize {
            ">"
        } else {
            " "
        };
        context.push_str(&format!("{} {:4} | {}\n", marker, actual_line_num, line));
    }

    Some(context)
}

#[allow(dead_code)]
/// Resolve a source file path, optionally prepending context_root
fn resolve_source_path(path: &Path, context_root: &Option<PathBuf>) -> Option<PathBuf> {
    // If absolute path, check if it exists
    if path.is_absolute() {
        if path.exists() {
            return Some(path.to_path_buf());
        }
        // Try stripping common prefixes for Docker/container paths
        // e.g., /app/src/main.py -> src/main.py
        if let Ok(stripped) = path.strip_prefix("/app") {
            let resolved = if let Some(root) = context_root {
                root.join(stripped)
            } else {
                PathBuf::from(".").join(stripped)
            };
            if resolved.exists() {
                return Some(resolved);
            }
        }
        return None;
    }

    // Relative path - prepend context_root or CWD
    let resolved = if let Some(root) = context_root {
        root.join(path)
    } else {
        PathBuf::from(".").join(path)
    };

    if resolved.exists() {
        Some(resolved)
    } else {
        None
    }
}

#[allow(dead_code)]
/// Extract context for all user code frames in a stack trace
fn extract_stack_trace_context(trace: &StackTrace, config: &SourceContextConfig) -> String {
    let mut total_context = String::new();
    let mut total_chars = 0;

    // Prioritize user code frames
    let frames: Vec<&StackFrame> = trace
        .frames
        .iter()
        .filter(|f| f.is_user_code && f.file.is_some() && f.line.is_some())
        .collect();

    for frame in frames {
        if total_chars >= config.max_context_chars {
            break;
        }

        if let Some(context) = extract_frame_context(frame, config) {
            let file_str = frame
                .file
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            let func_str = frame.function.as_deref().unwrap_or("<unknown>");

            let header = format!("\n--- {} in {} ---\n", func_str, file_str);
            total_context.push_str(&header);
            total_context.push_str(&context);

            total_chars += header.len() + context.len();
        }
    }

    total_context
}

// ============================================================================
// Phase 3.5: Enhanced Prompt Construction
// ============================================================================

#[allow(dead_code)]
/// Maximum number of frames to include in prompt
const MAX_PROMPT_FRAMES: usize = 5;

#[allow(dead_code)]
/// Format a stack frame for prompt inclusion
fn format_frame_for_prompt(frame: &StackFrame) -> String {
    let mut parts = Vec::new();

    if let Some(ref func) = frame.function {
        parts.push(func.clone());
    }

    if let Some(ref file) = frame.file {
        let file_str = file.display().to_string();
        if let Some(line) = frame.line {
            if let Some(col) = frame.column {
                parts.push(format!("at {}:{}:{}", file_str, line, col));
            } else {
                parts.push(format!("at {}:{}", file_str, line));
            }
        } else {
            parts.push(format!("at {}", file_str));
        }
    }

    if parts.is_empty() {
        "<unknown frame>".to_string()
    } else {
        parts.join(" ")
    }
}

#[allow(dead_code)]
/// Select the most relevant frames for the prompt
fn select_frames_for_prompt(trace: &StackTrace, max_frames: usize) -> Vec<&StackFrame> {
    let mut selected: Vec<&StackFrame> = Vec::new();

    // Always include user code frames first (most relevant)
    let user_frames: Vec<&StackFrame> = trace.frames.iter().filter(|f| f.is_user_code).collect();

    // Then framework/stdlib frames
    let other_frames: Vec<&StackFrame> = trace.frames.iter().filter(|f| !f.is_user_code).collect();

    // Add user frames first
    for frame in user_frames.iter().take(max_frames) {
        selected.push(frame);
    }

    // Fill remaining slots with other frames
    let remaining = max_frames.saturating_sub(selected.len());
    for frame in other_frames.iter().take(remaining) {
        selected.push(frame);
    }

    selected
}

#[allow(dead_code)]
/// Get language-specific hints for common error types
fn get_error_hints(trace: &StackTrace) -> Option<String> {
    match trace.language {
        Language::Python => get_python_hints(&trace.error_type, &trace.error_message),
        Language::Rust => get_rust_hints(&trace.error_type, &trace.error_message),
        Language::JavaScript => get_javascript_hints(&trace.error_type, &trace.error_message),
        Language::Go => get_go_hints(&trace.error_type, &trace.error_message),
        Language::Java => get_java_hints(&trace.error_type, &trace.error_message),
        Language::Cpp => get_cpp_hints(&trace.error_type, &trace.error_message),
        Language::Unknown => None,
    }
}

#[allow(dead_code)]
/// Python-specific error hints
fn get_python_hints(error_type: &str, _error_message: &str) -> Option<String> {
    match error_type {
        "KeyError" => Some(
            "Common causes: accessing dict with missing key, use .get() for safe access"
                .to_string(),
        ),
        "AttributeError" => {
            Some("Common causes: None value, typo in attribute name, wrong object type".to_string())
        }
        "TypeError" => Some(
            "Common causes: wrong argument type, None where object expected, operator mismatch"
                .to_string(),
        ),
        "IndexError" => {
            Some("Common causes: list index out of range, empty list access".to_string())
        }
        "ValueError" => {
            Some("Common causes: invalid conversion, wrong format, unexpected input".to_string())
        }
        "ImportError" | "ModuleNotFoundError" => Some(
            "Common causes: missing package, typo in module name, virtual environment issue"
                .to_string(),
        ),
        "FileNotFoundError" => {
            Some("Common causes: wrong path, file doesn't exist, permission issues".to_string())
        }
        "ZeroDivisionError" => {
            Some("Common causes: division by zero, check divisor before operation".to_string())
        }
        _ => None,
    }
}

#[allow(dead_code)]
/// Rust-specific error hints
fn get_rust_hints(error_type: &str, error_message: &str) -> Option<String> {
    if error_type.contains("E0382")
        || error_message.contains("borrow")
        || error_message.contains("moved")
    {
        return Some("Ownership issue: value was moved or borrowed. Consider using .clone(), references, or restructuring code".to_string());
    }
    if error_type.contains("E0502") || error_message.contains("mutable") {
        return Some(
            "Borrow checker issue: cannot have mutable and immutable borrows simultaneously"
                .to_string(),
        );
    }
    if error_type == "panic" {
        if error_message.contains("unwrap") || error_message.contains("None") {
            return Some(
                "Called unwrap() on None. Use match, if let, or unwrap_or() for safe handling"
                    .to_string(),
            );
        }
        if error_message.contains("index out of bounds") {
            return Some(
                "Array/vector index out of bounds. Check length before indexing or use .get()"
                    .to_string(),
            );
        }
    }
    None
}

#[allow(dead_code)]
/// JavaScript-specific error hints
fn get_javascript_hints(error_type: &str, error_message: &str) -> Option<String> {
    match error_type {
        "TypeError" => {
            if error_message.contains("undefined") || error_message.contains("null") {
                return Some("Accessing property on null/undefined. Use optional chaining (?.) or null checks".to_string());
            }
            Some("Type mismatch: wrong type passed or returned".to_string())
        }
        "ReferenceError" => {
            Some("Variable not defined. Check spelling, scope, and declaration order".to_string())
        }
        "SyntaxError" => {
            Some("Syntax error: check brackets, quotes, commas, and semicolons".to_string())
        }
        _ => {
            if error_message.contains("Promise") || error_message.contains("async") {
                return Some(
                    "Async error: ensure await is used, check Promise chain, add .catch() handler"
                        .to_string(),
                );
            }
            None
        }
    }
}

#[allow(dead_code)]
/// Go-specific error hints
fn get_go_hints(error_type: &str, error_message: &str) -> Option<String> {
    if error_type == "panic" {
        if error_message.contains("nil pointer") {
            return Some(
                "Nil pointer dereference. Check pointer is not nil before use".to_string(),
            );
        }
        if error_message.contains("index out of range") {
            return Some(
                "Slice/array index out of bounds. Check length before indexing".to_string(),
            );
        }
    }
    None
}

#[allow(dead_code)]
/// Java-specific error hints
fn get_java_hints(error_type: &str, _error_message: &str) -> Option<String> {
    if error_type.contains("NullPointerException") {
        return Some("Null pointer: object is null. Add null checks or use Optional".to_string());
    }
    if error_type.contains("ArrayIndexOutOfBoundsException") {
        return Some("Array index out of bounds. Check array length before access".to_string());
    }
    if error_type.contains("ClassCastException") {
        return Some("Invalid cast. Check object type with instanceof before casting".to_string());
    }
    if error_type.contains("ConcurrentModificationException") {
        return Some(
            "Collection modified during iteration. Use Iterator.remove() or copy collection"
                .to_string(),
        );
    }
    None
}

#[allow(dead_code)]
/// C/C++-specific error hints
fn get_cpp_hints(error_type: &str, error_message: &str) -> Option<String> {
    if error_type == "signal" || error_message.contains("Segmentation fault") {
        return Some(
            "Segfault: invalid memory access. Check null pointers, array bounds, freed memory"
                .to_string(),
        );
    }
    if error_type == "AddressSanitizer" {
        if error_message.contains("heap-buffer-overflow") {
            return Some(
                "Heap buffer overflow: writing past allocated memory. Check array bounds"
                    .to_string(),
            );
        }
        if error_message.contains("use-after-free") {
            return Some(
                "Use after free: accessing freed memory. Check pointer lifecycle".to_string(),
            );
        }
        if error_message.contains("stack-buffer-overflow") {
            return Some(
                "Stack buffer overflow: local array overrun. Check buffer sizes".to_string(),
            );
        }
    }
    None
}

#[allow(dead_code)]
/// Build an enhanced prompt with stack trace information
fn build_stack_trace_prompt(
    error_input: &str,
    trace: &StackTrace,
    source_context: Option<&str>,
) -> String {
    let mut prompt = String::new();

    // Error type and message
    prompt.push_str(&format!("ERROR TYPE: {}\n", trace.error_type));
    if !trace.error_message.is_empty() {
        prompt.push_str(&format!("ERROR MESSAGE: {}\n", trace.error_message));
    }
    prompt.push('\n');

    // Selected stack frames
    let selected_frames = select_frames_for_prompt(trace, MAX_PROMPT_FRAMES);
    if !selected_frames.is_empty() {
        prompt.push_str("STACK TRACE (most relevant frames):\n");
        for (i, frame) in selected_frames.iter().enumerate() {
            let marker = if frame.is_user_code { ">" } else { " " };
            prompt.push_str(&format!(
                "  {}[{}] {}\n",
                marker,
                i,
                format_frame_for_prompt(frame)
            ));
        }
        prompt.push('\n');
    }

    // Source context
    if let Some(context) = source_context {
        if !context.is_empty() {
            prompt.push_str("SOURCE CONTEXT:\n");
            prompt.push_str(context);
            prompt.push('\n');
        }
    }

    // Language-specific hints
    if let Some(hints) = get_error_hints(trace) {
        prompt.push_str(&format!("HINT: {}\n\n", hints));
    }

    // Original raw input for context
    if error_input != trace.raw_text && !error_input.is_empty() {
        prompt.push_str("ORIGINAL INPUT:\n");
        prompt.push_str(error_input);
        prompt.push('\n');
    }

    prompt
}

// ============================================================================
// End Stack Trace Intelligence
// ============================================================================

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

    /// Path to GGUF model file (overrides embedded model)
    #[arg(long, short = 'm', value_name = "PATH")]
    model: Option<PathBuf>,

    /// Model family for prompt template (auto-detected if not specified)
    #[arg(long, short = 't', value_enum, value_name = "FAMILY")]
    template: Option<ModelFamily>,

    /// List available model variants and exit
    #[arg(long)]
    list_models: bool,

    /// Generate shell completions
    #[arg(long, value_enum, value_name = "SHELL")]
    completions: Option<Shell>,

    /// Enable streaming output (display tokens as they're generated)
    #[arg(long, short = 's')]
    stream: bool,

    /// Generate shell hook integration script for the specified shell
    #[arg(long, value_enum, value_name = "SHELL")]
    hook: Option<Shell>,

    /// Exit code from the failed command (used by shell hooks)
    #[arg(long, value_name = "CODE")]
    exit_code: Option<i32>,

    /// The command that failed (used by shell hooks)
    #[arg(long, value_name = "CMD")]
    last_command: Option<String>,

    /// Enable source context injection (read source files referenced in stack traces)
    #[arg(long, short = 'c')]
    context: bool,

    /// Number of lines to show around error location (default: 5)
    #[arg(long, default_value = "5", value_name = "N")]
    context_lines: usize,

    /// Root path for resolving relative file paths in stack traces
    #[arg(long, value_name = "PATH")]
    context_root: Option<PathBuf>,

    /// Show parsed stack trace frames (requires stack trace in input)
    #[arg(long)]
    show_frames: bool,

    /// Run a command and explain its error output if it fails
    /// Example: why --capture npm run build
    #[arg(long)]
    capture: bool,

    /// Also capture and analyze stdout in capture mode (default: stderr only)
    #[arg(long)]
    capture_all: bool,

    /// Ask for confirmation before explaining errors (interactive mode)
    #[arg(long)]
    confirm: bool,

    /// Always explain without prompting (opposite of --confirm)
    #[arg(long)]
    auto: bool,

    /// Output default hook configuration to stdout
    #[arg(long)]
    hook_config: bool,

    /// Install hook integration into shell config file
    #[arg(long, value_enum, value_name = "SHELL")]
    hook_install: Option<Shell>,

    /// Uninstall hook integration from shell config file
    #[arg(long, value_enum, value_name = "SHELL")]
    hook_uninstall: Option<Shell>,

    // ========================================================================
    // Watch Mode (Feature 2)
    // ========================================================================
    /// Watch a file or command for errors and explain them as they appear
    /// Examples:
    ///   why --watch /var/log/app.log
    ///   why --watch "npm run dev"
    #[arg(long, short = 'w', value_name = "TARGET")]
    watch: Option<String>,

    /// Debounce time in milliseconds for watch mode (default: 500)
    #[arg(long, default_value = "500", value_name = "MS")]
    debounce: u64,

    /// Disable duplicate error suppression in watch mode
    #[arg(long)]
    no_dedup: bool,

    /// Custom regex pattern for error detection in watch mode
    #[arg(long, value_name = "REGEX")]
    pattern: Option<String>,

    /// Clear screen between errors in watch mode
    #[arg(long)]
    clear: bool,

    /// Quiet mode - only show explanations, no status messages
    #[arg(long, short = 'q')]
    quiet: bool,

    // ========================================================================
    // Daemon Mode (Feature 5)
    // ========================================================================
    /// Daemon management subcommand
    #[command(subcommand)]
    daemon: Option<DaemonCommand>,

    /// Prefer daemon connection for inference (falls back to direct if unavailable)
    #[arg(long, short = 'D')]
    use_daemon: bool,

    /// Require daemon connection (fail if daemon unavailable)
    #[arg(long)]
    daemon_required: bool,

    /// Don't auto-start daemon when using --daemon
    #[arg(long)]
    no_auto_start: bool,
}

/// Daemon management subcommand
#[derive(Subcommand, Debug, Clone)]
enum DaemonCommand {
    /// Start the daemon in the background
    Start {
        /// Run in foreground instead of daemonizing
        #[arg(long, short = 'f')]
        foreground: bool,

        /// Idle timeout in minutes (default: 30)
        #[arg(long, default_value = "30", value_name = "MINUTES")]
        idle_timeout: u64,
    },
    /// Stop the running daemon
    Stop {
        /// Force stop with SIGKILL if graceful shutdown fails
        #[arg(long)]
        force: bool,
    },
    /// Restart the daemon
    Restart {
        /// Run in foreground instead of daemonizing
        #[arg(long, short = 'f')]
        foreground: bool,
    },
    /// Show daemon status and statistics
    Status,
    /// Install system service (systemd/launchd)
    InstallService,
    /// Uninstall system service
    UninstallService,
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

/// JSON representation of a stack frame for structured output
#[derive(Debug, Serialize)]
struct StackFrameJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    function: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    column: Option<u32>,
    is_user_code: bool,
}

impl From<&StackFrame> for StackFrameJson {
    fn from(frame: &StackFrame) -> Self {
        Self {
            function: frame.function.clone(),
            file: frame.file.as_ref().map(|p| p.display().to_string()),
            line: frame.line,
            column: frame.column,
            is_user_code: frame.is_user_code,
        }
    }
}

/// JSON representation of a stack trace for structured output
#[derive(Debug, Serialize)]
struct StackTraceJson {
    language: Language,
    error_type: String,
    error_message: String,
    frames: Vec<StackFrameJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    root_cause_frame: Option<StackFrameJson>,
}

impl From<&StackTrace> for StackTraceJson {
    fn from(trace: &StackTrace) -> Self {
        Self {
            language: trace.language,
            error_type: trace.error_type.clone(),
            error_message: trace.error_message.clone(),
            frames: trace.frames.iter().map(StackFrameJson::from).collect(),
            root_cause_frame: trace.root_cause_frame().map(StackFrameJson::from),
        }
    }
}

// ============================================================================
// Configuration System (Feature 4, Phase 4.5)
// ============================================================================

/// Configuration for hook behavior
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct HookConfig {
    /// Always explain without prompting
    pub auto_explain: bool,
    /// Exit codes to skip (e.g., [0, 130] for success and Ctrl+C)
    pub skip_exit_codes: Vec<i32>,
    /// Minimum number of stderr lines to trigger explanation
    pub min_stderr_lines: usize,
    /// Command patterns to ignore
    pub ignore_commands: IgnoreCommandsConfig,
}

impl Default for HookConfig {
    fn default() -> Self {
        Self {
            auto_explain: false,
            skip_exit_codes: vec![0, 130], // Success and Ctrl+C
            min_stderr_lines: 1,
            ignore_commands: IgnoreCommandsConfig::default(),
        }
    }
}

/// Command patterns to ignore
#[derive(Debug, Deserialize, Clone)]
#[serde(default)]
pub struct IgnoreCommandsConfig {
    /// Regex patterns for commands to ignore
    pub patterns: Vec<String>,
}

impl Default for IgnoreCommandsConfig {
    fn default() -> Self {
        Self {
            patterns: vec![
                "^cd ".to_string(),
                "^ls".to_string(),
                "^echo ".to_string(),
                "^pwd$".to_string(),
                "^clear$".to_string(),
            ],
        }
    }
}

/// Root configuration structure
#[derive(Debug, Deserialize, Clone, Default)]
#[serde(default)]
pub struct Config {
    pub hook: HookConfig,
}

impl Config {
    /// Load config from file, returning default config if file doesn't exist
    pub fn load() -> Self {
        Self::load_from_path(Self::config_path())
    }

    /// Load config from a specific path
    pub fn load_from_path(path: Option<PathBuf>) -> Self {
        let Some(path) = path else {
            return Self::default();
        };

        if !path.exists() {
            return Self::default();
        }

        match std::fs::read_to_string(&path) {
            Ok(contents) => {
                match toml::from_str(&contents) {
                    Ok(config) => config,
                    Err(_e) => {
                        // Invalid config file, use defaults
                        Self::default()
                    }
                }
            }
            Err(_) => Self::default(),
        }
    }

    /// Get the config file path (~/.config/why/config.toml)
    pub fn config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("why").join("config.toml"))
    }

    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        // WHY_HOOK_AUTO=1 enables auto-explain
        if env::var("WHY_HOOK_AUTO").map(|v| v == "1").unwrap_or(false) {
            self.hook.auto_explain = true;
        }
    }

    /// Check if hook is disabled via environment variable
    pub fn is_hook_disabled() -> bool {
        env::var("WHY_HOOK_DISABLE")
            .map(|v| v == "1")
            .unwrap_or(false)
    }

    /// Check if a command matches any ignore patterns
    pub fn should_ignore_command(&self, command: &str) -> bool {
        for pattern in &self.hook.ignore_commands.patterns {
            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(command) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if an exit code should be skipped
    pub fn should_skip_exit_code(&self, code: i32) -> bool {
        self.hook.skip_exit_codes.contains(&code)
    }
}

/// Generate default config as TOML string
fn generate_default_config() -> String {
    r#"# Why - Error explanation tool configuration
# Place this file at ~/.config/why/config.toml

[hook]
# Always explain errors without prompting (default: false)
auto_explain = false

# Exit codes to skip (0 = success, 130 = Ctrl+C)
skip_exit_codes = [0, 130]

# Minimum stderr lines to trigger explanation
min_stderr_lines = 1

[hook.ignore_commands]
# Regex patterns for commands to ignore (won't be explained)
patterns = [
    "^cd ",      # cd command
    "^ls",       # ls command
    "^echo ",    # echo command
    "^pwd$",     # pwd command
    "^clear$",   # clear command
]

# Environment variable overrides:
# WHY_HOOK_AUTO=1    - Force auto-explain (overrides config)
# WHY_HOOK_DISABLE=1 - Temporarily disable hook explanations
"#
    .to_string()
}

/// Print the default config to stdout
fn print_hook_config() {
    print!("{}", generate_default_config());
}

/// Marker comment for detecting existing hook installations
const HOOK_MARKER_START: &str = "# >>> why shell hook >>>";
const HOOK_MARKER_END: &str = "# <<< why shell hook <<<";

/// Get the shell config file path for a given shell
fn get_shell_config_path(shell: Shell) -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    match shell {
        Shell::Bash => Some(home.join(".bashrc")),
        Shell::Zsh => Some(home.join(".zshrc")),
        Shell::Fish => dirs::config_dir().map(|p| p.join("fish").join("conf.d").join("why.fish")),
        Shell::PowerShell => dirs::config_dir().map(|p| {
            // Windows: ~/Documents/PowerShell/Microsoft.PowerShell_profile.ps1
            // Unix: ~/.config/powershell/Microsoft.PowerShell_profile.ps1
            if cfg!(windows) {
                dirs::document_dir()
                    .unwrap_or(p.clone())
                    .join("PowerShell")
                    .join("Microsoft.PowerShell_profile.ps1")
            } else {
                p.join("powershell")
                    .join("Microsoft.PowerShell_profile.ps1")
            }
        }),
        _ => None,
    }
}

/// Generate the hook code wrapped with markers
fn generate_hook_with_markers(shell: Shell) -> String {
    let mut output = String::new();
    output.push_str(HOOK_MARKER_START);
    output.push('\n');

    // Generate the hook content based on shell type
    let hook_content = match shell {
        Shell::Bash => {
            r#"__why_prompt_command() {
    local exit_code=$?
    if [[ $exit_code -ne 0 && $exit_code -ne 130 ]]; then
        why --exit-code "$exit_code" --last-command "$BASH_COMMAND" 2>/dev/null
    fi
}
PROMPT_COMMAND="__why_prompt_command${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
"#
        }
        Shell::Zsh => {
            r#"__why_precmd() {
    local exit_code=$?
    if [[ $exit_code -ne 0 && $exit_code -ne 130 ]]; then
        why --exit-code "$exit_code" --last-command "$__why_last_cmd" 2>/dev/null
    fi
}
__why_preexec() {
    __why_last_cmd="$1"
}
autoload -Uz add-zsh-hook
add-zsh-hook precmd __why_precmd
add-zsh-hook preexec __why_preexec
"#
        }
        Shell::Fish => {
            r#"function __why_postexec --on-event fish_postexec
    set -l exit_code $status
    if test $exit_code -ne 0 -a $exit_code -ne 130
        why --exit-code $exit_code --last-command "$argv" 2>/dev/null
    end
end
"#
        }
        Shell::PowerShell => {
            r#"function global:__why_prompt {
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0 -and $exitCode -ne 130) {
        why --exit-code $exitCode --last-command $MyInvocation.MyCommand 2>$null
    }
}
# Note: PowerShell hook integration is limited
"#
        }
        _ => "",
    };
    output.push_str(hook_content);
    output.push_str(HOOK_MARKER_END);
    output.push('\n');
    output
}

/// Check if hooks are already installed in a config file
fn hooks_already_installed(config_path: &Path) -> bool {
    if let Ok(contents) = std::fs::read_to_string(config_path) {
        contents.contains(HOOK_MARKER_START)
    } else {
        false
    }
}

/// Install hook integration into shell config file
fn install_hook(shell: Shell) -> Result<()> {
    let config_path = get_shell_config_path(shell)
        .ok_or_else(|| anyhow::anyhow!("Could not determine config path for {}", shell))?;

    // Check if hooks are already installed
    if hooks_already_installed(&config_path) {
        println!(
            "{} Why hooks are already installed in {}",
            "✓".green(),
            config_path.display()
        );
        return Ok(());
    }

    // Create parent directories if needed (especially for Fish)
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    // Read existing content or start fresh
    let mut content = std::fs::read_to_string(&config_path).unwrap_or_default();

    // Add a newline separator if file has content
    if !content.is_empty() && !content.ends_with('\n') {
        content.push('\n');
    }
    content.push('\n');

    // Append hook code
    content.push_str(&generate_hook_with_markers(shell));

    // Write back
    std::fs::write(&config_path, content)
        .with_context(|| format!("Failed to write to: {}", config_path.display()))?;

    println!(
        "{} {}",
        "✓".green(),
        "Why shell hooks installed successfully!".green().bold()
    );
    println!();
    println!(
        "  {} {}",
        "Config file:".blue().bold(),
        config_path.display()
    );
    println!();
    println!("  {} To activate, run:", "Next steps:".yellow().bold());
    match shell {
        Shell::Bash => println!("    source ~/.bashrc"),
        Shell::Zsh => println!("    source ~/.zshrc"),
        Shell::Fish => println!("    source {}", config_path.display()),
        Shell::PowerShell => println!("    . $PROFILE"),
        _ => {}
    }
    println!();
    println!("  Or open a new terminal session.");
    println!();

    Ok(())
}

/// Uninstall hook integration from shell config file
fn uninstall_hook(shell: Shell) -> Result<()> {
    let config_path = get_shell_config_path(shell)
        .ok_or_else(|| anyhow::anyhow!("Could not determine config path for {}", shell))?;

    if !config_path.exists() {
        println!(
            "{} Config file does not exist: {}",
            "?".yellow(),
            config_path.display()
        );
        return Ok(());
    }

    // Check if hooks are installed
    if !hooks_already_installed(&config_path) {
        println!(
            "{} Why hooks are not installed in {}",
            "?".yellow(),
            config_path.display()
        );
        return Ok(());
    }

    // Read content
    let content = std::fs::read_to_string(&config_path)
        .with_context(|| format!("Failed to read: {}", config_path.display()))?;

    // Remove the hook block
    let mut new_content = String::new();
    let mut in_hook_block = false;

    for line in content.lines() {
        if line.trim() == HOOK_MARKER_START {
            in_hook_block = true;
            continue;
        }
        if line.trim() == HOOK_MARKER_END {
            in_hook_block = false;
            continue;
        }
        if !in_hook_block {
            new_content.push_str(line);
            new_content.push('\n');
        }
    }

    // Clean up excessive blank lines
    while new_content.ends_with("\n\n\n") {
        new_content.pop();
    }

    // Write back
    std::fs::write(&config_path, new_content)
        .with_context(|| format!("Failed to write to: {}", config_path.display()))?;

    println!(
        "{} {}",
        "✓".green(),
        "Why shell hooks uninstalled successfully!".green().bold()
    );
    println!();
    println!(
        "  {} {}",
        "Config file:".blue().bold(),
        config_path.display()
    );
    println!();
    println!("  Restart your terminal or source the config file to apply changes.");
    println!();

    Ok(())
}

/// Print parsed stack trace frames as a formatted table
fn print_frames(trace: &StackTrace) {
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

    // Print header
    println!(
        "  {} {:^4} {:40} {:30} {:>6}",
        "".normal(),
        "#".dimmed(),
        "Function".dimmed(),
        "File".dimmed(),
        "Line".dimmed()
    );
    println!("  {}", "─".repeat(85).dimmed());

    // Print frames
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

    // Print root cause frame summary
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
fn format_file_line(file: &Path, line: Option<u32>, column: Option<u32>) -> String {
    let mut result = file.display().to_string().cyan().to_string();
    if let Some(l) = line {
        result.push_str(&format!(":{}", l.to_string().yellow()));
        if let Some(c) = column {
            result.push_str(&format!(":{}", c.to_string().yellow()));
        }
    }
    result
}

/// Check if a string contains common error patterns that suggest auto-explanation
fn contains_error_patterns(text: &str) -> bool {
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

/// Prompt user for confirmation and return whether they want to explain
fn prompt_confirm(command: &str, exit_code: i32, stderr: &str) -> bool {
    // If stderr contains obvious error patterns, suggest yes
    let suggested = if contains_error_patterns(stderr) {
        "Y/n"
    } else {
        "y/N"
    };
    let default_yes = contains_error_patterns(stderr);

    eprintln!();
    eprintln!(
        "{} {} (exit {})",
        "Command failed:".yellow().bold(),
        command,
        exit_code
    );
    eprint!(
        "{} Explain this error? [{}]: ",
        "?".cyan().bold(),
        suggested
    );
    io::stderr().flush().ok();

    // Read user input
    let mut input = String::new();
    if io::stdin().read_line(&mut input).is_err() {
        // If we can't read stdin, default to the suggested value
        return default_yes;
    }

    let input = input.trim().to_lowercase();
    match input.as_str() {
        "" => default_yes, // Empty input = default
        "y" | "yes" => true,
        "n" | "no" => false,
        _ => default_yes, // Unknown input = default
    }
}

/// Result of running a command in capture mode
#[derive(Debug)]
struct CaptureResult {
    /// Command that was run (for display)
    command: String,
    /// Exit code from the command
    exit_code: i32,
    /// Captured stdout (if capture_all is enabled)
    stdout: String,
    /// Captured stderr
    stderr: String,
}

/// Run a command and capture its output
/// Passes through stdout/stderr in real-time while also buffering
fn run_capture_command(command: &[String], capture_all: bool) -> Result<CaptureResult> {
    if command.is_empty() {
        bail!("No command specified. Use: why --capture -- <command>");
    }

    let cmd_name = &command[0];
    let cmd_args = &command[1..];
    let command_str = command.join(" ");

    // Spawn the command with piped outputs
    let mut child = Command::new(cmd_name)
        .args(cmd_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("Failed to run command: {}", command_str))?;

    // Capture buffers
    let stdout_buffer = Arc::new(Mutex::new(Vec::new()));
    let stderr_buffer = Arc::new(Mutex::new(Vec::new()));

    // Take ownership of child stdout/stderr
    let child_stdout = child.stdout.take();
    let child_stderr = child.stderr.take();

    // Spawn thread to handle stdout
    let stdout_handle = if let Some(mut stdout) = child_stdout {
        let buffer = Arc::clone(&stdout_buffer);
        let capture = capture_all;
        Some(thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match stdout.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        // Pass through to terminal
                        let _ = io::stdout().write_all(&buf[..n]);
                        let _ = io::stdout().flush();
                        // Buffer for later analysis if capture_all
                        if capture {
                            buffer.lock().unwrap().extend_from_slice(&buf[..n]);
                        }
                    }
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
        }))
    } else {
        None
    };

    // Spawn thread to handle stderr
    let stderr_handle = if let Some(mut stderr) = child_stderr {
        let buffer = Arc::clone(&stderr_buffer);
        Some(thread::spawn(move || {
            let mut buf = [0u8; 4096];
            loop {
                match stderr.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        // Pass through to terminal
                        let _ = io::stderr().write_all(&buf[..n]);
                        let _ = io::stderr().flush();
                        // Always buffer stderr for analysis
                        buffer.lock().unwrap().extend_from_slice(&buf[..n]);
                    }
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(_) => break,
                }
            }
        }))
    } else {
        None
    };

    // Wait for command to complete
    let status = child.wait()?;

    // Wait for output threads to finish
    if let Some(handle) = stdout_handle {
        let _ = handle.join();
    }
    if let Some(handle) = stderr_handle {
        let _ = handle.join();
    }

    // Extract captured output
    let stdout = String::from_utf8_lossy(&stdout_buffer.lock().unwrap()).to_string();
    let stderr = String::from_utf8_lossy(&stderr_buffer.lock().unwrap()).to_string();

    let exit_code = status.code().unwrap_or(-1);

    Ok(CaptureResult {
        command: command_str,
        exit_code,
        stdout,
        stderr,
    })
}

// ============================================================================
// Watch Mode (Feature 2)
// ============================================================================

/// Configuration for watch mode
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Debounce time for file changes
    pub debounce_ms: u64,
    /// Whether to suppress duplicate errors
    pub dedup: bool,
    /// TTL for duplicate suppression (5 minutes)
    pub dedup_ttl: Duration,
    /// Custom error detection pattern
    pub pattern: Option<Regex>,
    /// Whether to clear screen between errors
    pub clear: bool,
    /// Quiet mode - no status messages
    pub quiet: bool,
    /// Maximum lines to aggregate for an error
    pub max_aggregation_lines: usize,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            debounce_ms: 500,
            dedup: true,
            dedup_ttl: Duration::from_secs(300), // 5 minutes
            pattern: None,
            clear: false,
            quiet: false,
            max_aggregation_lines: 50,
        }
    }
}

/// Represents a detected error in watch mode
#[derive(Debug, Clone)]
pub struct DetectedError {
    /// The error content (potentially multi-line)
    pub content: String,
    /// When the error was detected
    pub timestamp: Instant,
    /// Hash of the error content (for deduplication)
    pub content_hash: u64,
}

impl DetectedError {
    /// Create a new detected error
    pub fn new(content: String) -> Self {
        let content_hash = Self::compute_hash(&content);
        Self {
            content,
            timestamp: Instant::now(),
            content_hash,
        }
    }

    /// Compute a hash of error content, ignoring timestamps and line numbers
    fn compute_hash(content: &str) -> u64 {
        // Strip common timestamp patterns and line numbers for dedup
        let normalized = Self::normalize_for_hash(content);
        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        hasher.finish()
    }

    /// Normalize content for hashing by stripping timestamps and line numbers
    fn normalize_for_hash(content: &str) -> String {
        // Common timestamp patterns: 2024-01-01, 12:34:56, [timestamp], etc.
        let timestamp_re =
            Regex::new(r"(?:\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|\[\d{10,}\]|\d{2}/\d{2}/\d{4})")
                .unwrap();

        // Line number patterns: :42:, line 42, at line 42
        let line_re = Regex::new(r"(?::\d+:|line \d+|at line \d+)").unwrap();

        let mut normalized = timestamp_re.replace_all(content, "").to_string();
        normalized = line_re.replace_all(&normalized, "").to_string();

        // Collapse whitespace
        normalized.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

/// Error deduplication tracker
pub struct ErrorDeduplicator {
    /// Map of content hash to last seen time
    seen: HashMap<u64, Instant>,
    /// TTL for entries
    ttl: Duration,
}

impl ErrorDeduplicator {
    pub fn new(ttl: Duration) -> Self {
        Self {
            seen: HashMap::new(),
            ttl,
        }
    }

    /// Check if an error is a duplicate (seen within TTL)
    pub fn is_duplicate(&mut self, error: &DetectedError) -> bool {
        // First, clean up expired entries
        self.cleanup();

        if let Some(&last_seen) = self.seen.get(&error.content_hash) {
            if last_seen.elapsed() < self.ttl {
                return true;
            }
        }

        // Not a duplicate, record it
        self.seen.insert(error.content_hash, Instant::now());
        false
    }

    /// Remove expired entries
    fn cleanup(&mut self) {
        self.seen
            .retain(|_, last_seen| last_seen.elapsed() < self.ttl);
    }
}

/// Error detector for watch mode
pub struct ErrorDetector {
    /// Custom pattern (if any)
    pattern: Option<Regex>,
    /// Lines being aggregated for current error
    aggregation_buffer: Vec<String>,
    /// Maximum lines to aggregate
    max_lines: usize,
    /// Whether we're currently in an error context
    in_error: bool,
    /// Blank line count (for boundary detection)
    blank_count: usize,
}

impl ErrorDetector {
    pub fn new(pattern: Option<Regex>, max_lines: usize) -> Self {
        Self {
            pattern,
            aggregation_buffer: Vec::new(),
            max_lines,
            in_error: false,
            blank_count: 0,
        }
    }

    /// Check if a line looks like an error
    pub fn is_error_line(&self, line: &str) -> bool {
        // If we have a custom pattern, use it
        if let Some(ref pattern) = self.pattern {
            return pattern.is_match(line);
        }

        // Default error detection heuristics
        let lower = line.to_lowercase();

        // Keywords at start of line
        let starts_with_error = lower.starts_with("error")
            || lower.starts_with("e:")
            || lower.starts_with("err:")
            || lower.starts_with("fatal")
            || lower.starts_with("panic")
            || lower.starts_with("exception")
            || lower.starts_with("traceback");

        // Keywords anywhere
        let contains_error = lower.contains("error:")
            || lower.contains("error[e")  // Rust error codes
            || lower.contains("exception:")
            || lower.contains("failed:")
            || lower.contains(": error:")
            || lower.contains("panic:")
            || lower.contains("segmentation fault")
            || lower.contains("sigsegv")
            || lower.contains("sigabrt")
            || lower.contains("undefined reference")
            || lower.contains("cannot find")
            || lower.contains("not found")
            || lower.contains("no such file");

        // Stack trace indicators
        let is_stack_trace = line.trim().starts_with("at ")
            || line.contains("File \"")  // Python
            || line.contains("at /")     // Many languages
            || lower.contains("traceback (most recent call last)");

        starts_with_error || contains_error || is_stack_trace
    }

    /// Process a line and return detected errors
    pub fn process_line(&mut self, line: &str) -> Option<DetectedError> {
        let trimmed = line.trim();
        let is_blank = trimmed.is_empty();
        let is_error = self.is_error_line(line);

        // Track blank lines for boundary detection
        if is_blank {
            self.blank_count += 1;
        } else {
            self.blank_count = 0;
        }

        if is_error {
            // Start or continue error aggregation
            self.in_error = true;
            self.aggregation_buffer.push(line.to_string());
            None
        } else if self.in_error {
            // We're in an error context
            if is_blank && self.blank_count >= 2 {
                // Two consecutive blank lines = end of error
                return self.flush_error();
            } else if self.aggregation_buffer.len() >= self.max_lines {
                // Hit max aggregation limit
                return self.flush_error();
            } else if !is_blank {
                // Continuation of error (stack trace lines, etc.)
                self.aggregation_buffer.push(line.to_string());
            }
            None
        } else {
            None
        }
    }

    /// Flush current aggregation buffer and return detected error
    pub fn flush_error(&mut self) -> Option<DetectedError> {
        if self.aggregation_buffer.is_empty() {
            return None;
        }

        let content = self.aggregation_buffer.join("\n");
        self.aggregation_buffer.clear();
        self.in_error = false;
        self.blank_count = 0;

        Some(DetectedError::new(content))
    }
}

/// File watcher for watch mode
#[derive(Debug)]
pub struct FileWatcher {
    /// Path to watch
    path: PathBuf,
    /// Current read position
    position: u64,
    /// Last known file size (for truncation detection)
    last_size: u64,
}

impl FileWatcher {
    /// Create a new file watcher
    pub fn new(path: PathBuf) -> Result<Self> {
        // Validate path exists
        if !path.exists() {
            bail!("File does not exist: {}", path.display());
        }

        if !path.is_file() {
            bail!("Not a file: {}", path.display());
        }

        // Get initial file size
        let metadata = std::fs::metadata(&path)?;
        let size = metadata.len();

        Ok(Self {
            path,
            position: size, // Start at end of file (tail behavior)
            last_size: size,
        })
    }

    /// Read new content from the file
    pub fn read_new_content(&mut self) -> Result<Option<String>> {
        let metadata = std::fs::metadata(&self.path)?;
        let current_size = metadata.len();

        // Check for truncation (log rotation)
        if current_size < self.last_size {
            // File was truncated, reset to beginning
            self.position = 0;
        }
        self.last_size = current_size;

        // No new content
        if self.position >= current_size {
            return Ok(None);
        }

        // Read new content
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.position))?;

        let mut content = String::new();
        file.read_to_string(&mut content)?;

        self.position = current_size;

        if content.is_empty() {
            Ok(None)
        } else {
            Ok(Some(content))
        }
    }

    /// Get the path being watched
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Command watcher for watch mode
pub struct CommandWatcher {
    /// Child process
    child: Child,
    /// Command string (for display)
    command: String,
    /// Reader for stdout
    stdout_reader: Option<BufReader<std::process::ChildStdout>>,
    /// Reader for stderr
    stderr_reader: Option<BufReader<std::process::ChildStderr>>,
}

impl CommandWatcher {
    /// Create a new command watcher
    pub fn new(command: &str) -> Result<Self> {
        // Parse command - simple split on whitespace for now
        // For complex commands, users should wrap in shell
        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            bail!("Empty command");
        }

        let mut child = Command::new(parts[0])
            .args(&parts[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("Failed to start command: {}", command))?;

        let stdout_reader = child.stdout.take().map(BufReader::new);
        let stderr_reader = child.stderr.take().map(BufReader::new);

        Ok(Self {
            child,
            command: command.to_string(),
            stdout_reader,
            stderr_reader,
        })
    }

    /// Check if the child process is still running
    pub fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Get the exit code (if finished)
    pub fn exit_code(&mut self) -> Option<i32> {
        match self.child.try_wait() {
            Ok(Some(status)) => status.code(),
            _ => None,
        }
    }

    /// Kill the child process
    pub fn kill(&mut self) -> Result<()> {
        self.child.kill()?;
        Ok(())
    }

    /// Get the command string
    pub fn command(&self) -> &str {
        &self.command
    }
}

/// Watch mode session state
pub struct WatchSession {
    /// Configuration
    config: WatchConfig,
    /// Error detector
    detector: ErrorDetector,
    /// Error deduplicator
    deduplicator: ErrorDeduplicator,
    /// Error counter
    error_count: usize,
    /// Explained error count
    explained_count: usize,
    /// Paused state
    paused: bool,
    /// Running flag
    running: Arc<AtomicBool>,
}

impl WatchSession {
    /// Create a new watch session
    pub fn new(config: WatchConfig) -> Self {
        let pattern = config.pattern.clone();
        let max_lines = config.max_aggregation_lines;
        let ttl = config.dedup_ttl;
        let dedup = config.dedup;

        Self {
            config,
            detector: ErrorDetector::new(pattern, max_lines),
            deduplicator: ErrorDeduplicator::new(if dedup { ttl } else { Duration::ZERO }),
            error_count: 0,
            explained_count: 0,
            paused: false,
            running: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Get running flag for shutdown signaling
    pub fn running_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.running)
    }

    /// Stop the session
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }

    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Toggle pause state
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    /// Check if paused
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Toggle deduplication
    pub fn toggle_dedup(&mut self) {
        self.config.dedup = !self.config.dedup;
    }

    /// Process a line of input
    pub fn process_line(&mut self, line: &str) -> Option<DetectedError> {
        if self.paused {
            return None;
        }

        if let Some(error) = self.detector.process_line(line) {
            self.error_count += 1;

            // Check deduplication
            if self.config.dedup && self.deduplicator.is_duplicate(&error) {
                return None;
            }

            Some(error)
        } else {
            None
        }
    }

    /// Force flush any pending error
    pub fn flush(&mut self) -> Option<DetectedError> {
        if let Some(error) = self.detector.flush_error() {
            self.error_count += 1;
            if self.config.dedup && self.deduplicator.is_duplicate(&error) {
                return None;
            }
            Some(error)
        } else {
            None
        }
    }

    /// Increment explained count
    pub fn mark_explained(&mut self) {
        self.explained_count += 1;
    }

    /// Get status string
    pub fn status(&self) -> String {
        format!(
            "[{}/{}] errors detected/explained",
            self.error_count, self.explained_count
        )
    }

    /// Get config
    pub fn config(&self) -> &WatchConfig {
        &self.config
    }
}

/// Print watch mode startup banner
fn print_watch_banner(target: &str, config: &WatchConfig) {
    println!();
    println!("{} {}", "▸".cyan(), "Watch Mode".cyan().bold());
    println!("  {} {}", "Watching:".blue().bold(), target.bright_white());
    println!("  {} {}ms", "Debounce:".blue().bold(), config.debounce_ms);
    println!(
        "  {} {}",
        "Dedup:".blue().bold(),
        if config.dedup {
            "enabled (5 min TTL)"
        } else {
            "disabled"
        }
    );
    if let Some(ref pattern) = config.pattern {
        println!("  {} {}", "Pattern:".blue().bold(), pattern.as_str());
    }
    println!();
    println!(
        "  {}",
        "Press 'q' to quit, 'p' to pause, 'd' to toggle dedup".dimmed()
    );
    println!();
    println!("{}", "─".repeat(60).dimmed());
    println!();
}

/// Print waiting indicator
fn print_waiting() {
    print!("\r{} ", "Waiting for errors...".dimmed());
    io::stdout().flush().ok();
}

/// Clear the waiting line
fn clear_waiting_line() {
    print!("\r{}\r", " ".repeat(40));
    io::stdout().flush().ok();
}

/// Print error separator with timestamp
fn print_error_separator(count: usize) {
    let now = chrono_lite_now();
    println!();
    println!(
        "{} {} {}",
        "─".repeat(10).dimmed(),
        format!("[{}] Error #{}", now, count).yellow(),
        "─".repeat(10).dimmed()
    );
    println!();
}

/// Simple timestamp without chrono dependency
fn chrono_lite_now() -> String {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let seconds = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, seconds)
}

/// Run watch mode for a file
fn run_file_watch(
    path: PathBuf,
    config: WatchConfig,
    cli: &Cli,
    model_info: &ModelPathInfo,
) -> Result<()> {
    let mut file_watcher = FileWatcher::new(path.clone())?;
    let mut session = WatchSession::new(config.clone());

    if !config.quiet {
        print_watch_banner(&path.display().to_string(), &config);
    }

    // Set up file system watcher
    let (tx, rx) = mpsc::channel();
    let debounce_duration = Duration::from_millis(config.debounce_ms);

    let mut watcher = RecommendedWatcher::new(
        move |res: std::result::Result<NotifyEvent, notify::Error>| {
            if let Ok(event) = res {
                let _ = tx.send(event);
            }
        },
        NotifyConfig::default().with_poll_interval(Duration::from_millis(100)),
    )?;

    watcher.watch(&path, RecursiveMode::NonRecursive)?;

    // Set up keyboard handling
    let _running = session.running_flag();
    let is_tty = io::stdin().is_terminal();

    if is_tty {
        terminal::enable_raw_mode().ok();
    }

    // Store last debounce time
    let mut last_event_time = Instant::now();
    let mut pending_read = false;

    if !config.quiet && is_tty {
        print_waiting();
    }

    // Main loop
    while session.is_running() {
        // Check for keyboard input
        if is_tty && event::poll(Duration::from_millis(50))? {
            if let Event::Key(key_event) = event::read()? {
                match key_event.code {
                    KeyCode::Char('q') => {
                        session.stop();
                        break;
                    }
                    KeyCode::Char('c') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                        session.stop();
                        break;
                    }
                    KeyCode::Char('p') => {
                        session.toggle_pause();
                        if !config.quiet {
                            clear_waiting_line();
                            if session.is_paused() {
                                println!("{}", "Paused. Press 'p' to resume.".yellow());
                            } else {
                                println!("{}", "Resumed.".green());
                            }
                            print_waiting();
                        }
                    }
                    KeyCode::Char('d') => {
                        session.toggle_dedup();
                        if !config.quiet {
                            clear_waiting_line();
                            println!(
                                "Dedup {}",
                                if session.config().dedup {
                                    "enabled".green()
                                } else {
                                    "disabled".red()
                                }
                            );
                            print_waiting();
                        }
                    }
                    KeyCode::Char('c') => {
                        if config.clear {
                            print!("\x1B[2J\x1B[1;1H");
                            io::stdout().flush().ok();
                        }
                    }
                    _ => {}
                }
            }
        }

        // Check for file events
        match rx.try_recv() {
            Ok(_event) => {
                // Debounce: only mark pending if enough time has passed
                if last_event_time.elapsed() >= debounce_duration {
                    pending_read = true;
                    last_event_time = Instant::now();
                }
            }
            Err(mpsc::TryRecvError::Empty) => {
                // If we have a pending read and debounce time has passed, do the read
                if pending_read && last_event_time.elapsed() >= debounce_duration {
                    pending_read = false;

                    if let Some(content) = file_watcher.read_new_content()? {
                        if !config.quiet {
                            clear_waiting_line();
                        }

                        // Process each line
                        for line in content.lines() {
                            if let Some(error) = session.process_line(line) {
                                process_detected_error(
                                    &error,
                                    &mut session,
                                    cli,
                                    model_info,
                                    &config,
                                )?;
                            }
                        }

                        // Flush any remaining error
                        if let Some(error) = session.flush() {
                            process_detected_error(&error, &mut session, cli, model_info, &config)?;
                        }

                        if !config.quiet && is_tty {
                            print_waiting();
                        }
                    }
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                break;
            }
        }

        // Small sleep to avoid busy loop
        thread::sleep(Duration::from_millis(10));
    }

    // Cleanup
    if is_tty {
        terminal::disable_raw_mode().ok();
    }

    if !config.quiet {
        clear_waiting_line();
        println!();
        println!("{} {}", "▸".green(), "Watch mode ended".green().bold());
        println!("  {}", session.status());
        println!();
    }

    Ok(())
}

/// Run watch mode for a command
fn run_command_watch(
    command: &str,
    config: WatchConfig,
    cli: &Cli,
    model_info: &ModelPathInfo,
) -> Result<()> {
    let mut cmd_watcher = CommandWatcher::new(command)?;
    let mut session = WatchSession::new(config.clone());

    if !config.quiet {
        print_watch_banner(command, &config);
    }

    // Set up keyboard handling
    let _running = session.running_flag();
    let is_tty = io::stdin().is_terminal();

    if is_tty {
        terminal::enable_raw_mode().ok();
    }

    // Channels for stdout/stderr lines
    let (line_tx, line_rx) = mpsc::channel::<String>();

    // Spawn thread for stderr
    let stderr_tx = line_tx.clone();
    let stderr_reader = cmd_watcher.stderr_reader.take();
    let stderr_handle = stderr_reader.map(|reader| {
        thread::spawn(move || {
            for line in reader.lines() {
                if let Ok(line) = line {
                    // Echo to terminal
                    eprintln!("{}", line);
                    // Send for processing
                    let _ = stderr_tx.send(line);
                }
            }
        })
    });

    // Spawn thread for stdout (just pass through, optionally process)
    let stdout_tx = line_tx;
    let stdout_reader = cmd_watcher.stdout_reader.take();
    let stdout_handle = stdout_reader.map(|reader| {
        thread::spawn(move || {
            for line in reader.lines() {
                if let Ok(line) = line {
                    // Echo to terminal
                    println!("{}", line);
                    // Send for processing
                    let _ = stdout_tx.send(line);
                }
            }
        })
    });

    // Main loop
    while session.is_running() {
        // Check if command is still running
        if !cmd_watcher.is_running() {
            // Process any remaining lines
            while let Ok(line) = line_rx.try_recv() {
                if let Some(error) = session.process_line(&line) {
                    process_detected_error(&error, &mut session, cli, model_info, &config)?;
                }
            }
            if let Some(error) = session.flush() {
                process_detected_error(&error, &mut session, cli, model_info, &config)?;
            }

            // Check exit code
            if let Some(exit_code) = cmd_watcher.exit_code() {
                if exit_code != 0 && !config.quiet {
                    println!();
                    println!(
                        "{} {} (exit {})",
                        "Command exited:".yellow().bold(),
                        interpret_exit_code(exit_code),
                        exit_code
                    );
                }
            }
            break;
        }

        // Check for keyboard input
        if is_tty && event::poll(Duration::from_millis(50))? {
            if let Event::Key(key_event) = event::read()? {
                match key_event.code {
                    KeyCode::Char('q') => {
                        cmd_watcher.kill().ok();
                        session.stop();
                        break;
                    }
                    KeyCode::Char('c') if key_event.modifiers.contains(KeyModifiers::CONTROL) => {
                        cmd_watcher.kill().ok();
                        session.stop();
                        break;
                    }
                    KeyCode::Char('p') => {
                        session.toggle_pause();
                        if !config.quiet {
                            if session.is_paused() {
                                println!("{}", "Paused (output continues, errors not explained). Press 'p' to resume.".yellow());
                            } else {
                                println!("{}", "Resumed.".green());
                            }
                        }
                    }
                    KeyCode::Char('d') => {
                        session.toggle_dedup();
                        if !config.quiet {
                            println!(
                                "Dedup {}",
                                if session.config().dedup {
                                    "enabled".green()
                                } else {
                                    "disabled".red()
                                }
                            );
                        }
                    }
                    _ => {}
                }
            }
        }

        // Process incoming lines
        while let Ok(line) = line_rx.try_recv() {
            if let Some(error) = session.process_line(&line) {
                process_detected_error(&error, &mut session, cli, model_info, &config)?;
            }
        }

        // Small sleep to avoid busy loop
        thread::sleep(Duration::from_millis(10));
    }

    // Wait for threads
    if let Some(handle) = stderr_handle {
        let _ = handle.join();
    }
    if let Some(handle) = stdout_handle {
        let _ = handle.join();
    }

    // Cleanup
    if is_tty {
        terminal::disable_raw_mode().ok();
    }

    if !config.quiet {
        println!();
        println!("{} {}", "▸".green(), "Watch mode ended".green().bold());
        println!("  {}", session.status());
        println!();
    }

    Ok(())
}

/// Process a detected error in watch mode
fn process_detected_error(
    error: &DetectedError,
    session: &mut WatchSession,
    cli: &Cli,
    model_info: &ModelPathInfo,
    config: &WatchConfig,
) -> Result<()> {
    if config.clear {
        print!("\x1B[2J\x1B[1;1H");
        io::stdout().flush().ok();
    }

    if !config.quiet {
        print_error_separator(session.error_count);
    }

    // Run inference on the error
    let model_path = &model_info.path;
    let (model_family, _) = if let Some(family) = cli.template {
        (family, "override".to_string())
    } else if let Some(family) = model_info.embedded_family {
        (family, "embedded".to_string())
    } else {
        let detected = detect_model_family(model_path);
        (detected, "auto".to_string())
    };

    let prompt = build_prompt(&error.content, model_family);

    // Run inference with streaming if enabled
    let callback: Option<TokenCallback> = if cli.stream && !cli.json {
        Some(Box::new(|token: &str| {
            print!("{}", token);
            io::stdout().flush().ok();
            Ok(true)
        }))
    } else {
        None
    };

    let params = SamplingParams::default();
    match run_inference_with_callback(model_path, &prompt, &params, callback) {
        Ok((response, _stats)) => {
            if !cli.stream || cli.json {
                let result = parse_response(&error.content, &response);
                if cli.json {
                    let payload = serde_json::json!({
                        "input": error.content,
                        "error": result.error,
                        "summary": result.summary,
                        "explanation": result.explanation,
                        "suggestion": result.suggestion
                    });
                    println!("{}", serde_json::to_string_pretty(&payload)?);
                } else {
                    print_colored(&result);
                }
            } else {
                // Streaming mode - output already printed
                println!();
            }
            session.mark_explained();
        }
        Err(e) => {
            eprintln!(
                "{} Failed to explain error: {}",
                "Warning:".yellow().bold(),
                e
            );
        }
    }

    Ok(())
}

/// Determine if watch target is a file or command
fn is_file_target(target: &str) -> bool {
    let path = Path::new(target);
    // If it contains path separators or exists as a file, treat as file
    path.exists() || target.contains('/') || target.contains('\\')
}

/// Run watch mode
fn run_watch_mode(target: &str, cli: &Cli, model_info: &ModelPathInfo) -> Result<()> {
    let config = WatchConfig {
        debounce_ms: cli.debounce,
        dedup: !cli.no_dedup,
        dedup_ttl: Duration::from_secs(300),
        pattern: cli.pattern.as_ref().map(|p| Regex::new(p)).transpose()?,
        clear: cli.clear,
        quiet: cli.quiet,
        max_aggregation_lines: 50,
    };

    if is_file_target(target) {
        run_file_watch(PathBuf::from(target), config, cli, model_info)
    } else {
        run_command_watch(target, config, cli, model_info)
    }
}

// ============================================================================
// Daemon Mode (Feature 5)
// ============================================================================

/// Default connection timeout for daemon client
const DAEMON_CONNECTION_TIMEOUT_MS: u64 = 1000;

/// Default grace period for daemon shutdown
const DAEMON_SHUTDOWN_GRACE_MS: u64 = 5000;

/// Get the daemon socket path
/// Uses XDG_RUNTIME_DIR if available, falls back to /tmp
#[cfg(unix)]
pub fn get_socket_path() -> PathBuf {
    // Try XDG_RUNTIME_DIR first (preferred on Linux)
    if let Ok(runtime_dir) = env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(runtime_dir).join("why.sock");
    }

    // Fall back to /tmp with UID for uniqueness
    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/tmp/why-{}.sock", uid))
}

/// Get the daemon PID file path
#[cfg(unix)]
pub fn get_pid_path() -> PathBuf {
    // Try XDG_RUNTIME_DIR first
    if let Ok(runtime_dir) = env::var("XDG_RUNTIME_DIR") {
        return PathBuf::from(runtime_dir).join("why.pid");
    }

    let uid = unsafe { libc::getuid() };
    PathBuf::from(format!("/tmp/why-{}.pid", uid))
}

/// Get the daemon log file path
#[cfg(unix)]
pub fn get_log_path() -> Option<PathBuf> {
    dirs::cache_dir().map(|p| p.join("why").join("daemon.log"))
}

/// Daemon request protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonRequest {
    /// Action to perform
    pub action: DaemonAction,
    /// Input for explain action
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    /// Options for the request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<DaemonRequestOptions>,
}

/// Daemon request options
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DaemonRequestOptions {
    /// Enable streaming response
    #[serde(default)]
    pub stream: bool,
    /// Output as JSON
    #[serde(default)]
    pub json: bool,
    /// Enable source context
    #[serde(default)]
    pub context: bool,
    /// Context lines
    #[serde(default)]
    pub context_lines: Option<usize>,
    /// Context root path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_root: Option<String>,
}

/// Daemon action types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DaemonAction {
    /// Explain an error
    Explain,
    /// Ping the daemon (health check)
    Ping,
    /// Request daemon shutdown
    Shutdown,
    /// Get daemon statistics
    Stats,
}

/// Daemon response protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonResponse {
    /// Response type
    #[serde(rename = "type")]
    pub response_type: DaemonResponseType,
    /// Token content (for streaming)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Complete explanation (for final response)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<ErrorExplanationResponse>,
    /// Error message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Stats (for stats response)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<DaemonStats>,
}

impl DaemonResponse {
    /// Create a token response
    pub fn token(content: &str) -> Self {
        Self {
            response_type: DaemonResponseType::Token,
            content: Some(content.to_string()),
            explanation: None,
            error: None,
            stats: None,
        }
    }

    /// Create a complete response with explanation
    pub fn complete(explanation: ErrorExplanationResponse) -> Self {
        Self {
            response_type: DaemonResponseType::Complete,
            content: None,
            explanation: Some(explanation),
            error: None,
            stats: None,
        }
    }

    /// Create an error response
    pub fn error(message: &str) -> Self {
        Self {
            response_type: DaemonResponseType::Error,
            content: None,
            explanation: None,
            error: Some(message.to_string()),
            stats: None,
        }
    }

    /// Create a pong response
    pub fn pong() -> Self {
        Self {
            response_type: DaemonResponseType::Pong,
            content: None,
            explanation: None,
            error: None,
            stats: None,
        }
    }

    /// Create a stats response
    pub fn stats(stats: DaemonStats) -> Self {
        Self {
            response_type: DaemonResponseType::Stats,
            content: None,
            explanation: None,
            error: None,
            stats: Some(stats),
        }
    }

    /// Create a shutdown acknowledgment
    pub fn shutdown_ack() -> Self {
        Self {
            response_type: DaemonResponseType::ShutdownAck,
            content: None,
            explanation: None,
            error: None,
            stats: None,
        }
    }
}

/// Daemon response types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DaemonResponseType {
    /// Streaming token
    Token,
    /// Complete response
    Complete,
    /// Error response
    Error,
    /// Pong (ping response)
    Pong,
    /// Stats response
    Stats,
    /// Shutdown acknowledgment
    #[serde(rename = "shutdown_ack")]
    ShutdownAck,
}

/// Error explanation for daemon response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorExplanationResponse {
    pub error: String,
    pub summary: String,
    pub explanation: String,
    pub suggestion: String,
}

impl From<&ErrorExplanation> for ErrorExplanationResponse {
    fn from(exp: &ErrorExplanation) -> Self {
        Self {
            error: exp.error.clone(),
            summary: exp.summary.clone(),
            explanation: exp.explanation.clone(),
            suggestion: exp.suggestion.clone(),
        }
    }
}

/// Daemon statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DaemonStats {
    /// Daemon uptime in seconds
    pub uptime_seconds: u64,
    /// Total requests served
    pub requests_served: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Current memory usage (RSS) in MB
    pub memory_mb: f64,
    /// Model family
    pub model_family: String,
    /// Whether model is loaded
    pub model_loaded: bool,
}

/// Check if daemon is running by testing socket connectivity
#[cfg(unix)]
pub fn is_daemon_running() -> bool {
    let socket_path = get_socket_path();
    if !socket_path.exists() {
        return false;
    }

    // Try to connect with short timeout
    match UnixStream::connect(&socket_path) {
        Ok(stream) => {
            // Set short timeout
            stream
                .set_read_timeout(Some(Duration::from_millis(100)))
                .ok();
            stream
                .set_write_timeout(Some(Duration::from_millis(100)))
                .ok();

            // Try to send a ping
            use std::io::Write;
            let request = DaemonRequest {
                action: DaemonAction::Ping,
                input: None,
                options: None,
            };
            if let Ok(json) = serde_json::to_string(&request) {
                let mut writer = std::io::BufWriter::new(&stream);
                if writeln!(writer, "{}", json).is_ok() && writer.flush().is_ok() {
                    // Try to read response
                    let mut reader = std::io::BufReader::new(&stream);
                    let mut line = String::new();
                    if std::io::BufRead::read_line(&mut reader, &mut line).is_ok() {
                        if let Ok(response) = serde_json::from_str::<DaemonResponse>(&line) {
                            return response.response_type == DaemonResponseType::Pong;
                        }
                    }
                }
            }
            false
        }
        Err(_) => false,
    }
}

/// Non-Unix stub for is_daemon_running
#[cfg(not(unix))]
pub fn is_daemon_running() -> bool {
    false
}

/// Read PID from PID file
#[cfg(unix)]
pub fn read_daemon_pid() -> Option<u32> {
    let pid_path = get_pid_path();
    std::fs::read_to_string(pid_path)
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Check if a process with given PID is running
#[cfg(unix)]
pub fn is_process_running(pid: u32) -> bool {
    // Send signal 0 to check if process exists
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

/// Send a request to the daemon and get responses
#[cfg(unix)]
pub fn send_daemon_request(request: &DaemonRequest) -> Result<Vec<DaemonResponse>> {
    let socket_path = get_socket_path();
    let stream = UnixStream::connect(&socket_path)
        .with_context(|| format!("Failed to connect to daemon at {}", socket_path.display()))?;

    stream.set_read_timeout(Some(Duration::from_secs(60))).ok();
    stream
        .set_write_timeout(Some(Duration::from_millis(DAEMON_CONNECTION_TIMEOUT_MS)))
        .ok();

    // Send request
    let json = serde_json::to_string(request)?;
    {
        use std::io::Write;
        let mut writer = std::io::BufWriter::new(&stream);
        writeln!(writer, "{}", json)?;
        writer.flush()?;
    }

    // Read responses (may be multiple for streaming)
    let mut responses = Vec::new();
    let reader = std::io::BufReader::new(&stream);
    for line in std::io::BufRead::lines(reader) {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let response: DaemonResponse = serde_json::from_str(&line)
            .with_context(|| format!("Invalid daemon response: {}", line))?;

        let is_final = matches!(
            response.response_type,
            DaemonResponseType::Complete
                | DaemonResponseType::Error
                | DaemonResponseType::Pong
                | DaemonResponseType::Stats
                | DaemonResponseType::ShutdownAck
        );

        responses.push(response);

        if is_final {
            break;
        }
    }

    Ok(responses)
}

/// Non-Unix stub
#[cfg(not(unix))]
pub fn send_daemon_request(_request: &DaemonRequest) -> Result<Vec<DaemonResponse>> {
    bail!("Daemon mode is not supported on this platform")
}

/// Handle daemon subcommand
#[cfg(unix)]
fn handle_daemon_command(cmd: &DaemonCommand, cli: &Cli) -> Result<()> {
    match cmd {
        DaemonCommand::Start {
            foreground,
            idle_timeout,
        } => daemon_start(*foreground, *idle_timeout, cli),
        DaemonCommand::Stop { force } => daemon_stop(*force),
        DaemonCommand::Restart { foreground } => daemon_restart(*foreground, cli),
        DaemonCommand::Status => daemon_status(),
        DaemonCommand::InstallService => daemon_install_service(),
        DaemonCommand::UninstallService => daemon_uninstall_service(),
    }
}

/// Non-Unix stub for daemon command handler
#[cfg(not(unix))]
fn handle_daemon_command(_cmd: &DaemonCommand, _cli: &Cli) -> Result<()> {
    bail!("Daemon mode is not supported on this platform")
}

/// Start the daemon
#[cfg(unix)]
fn daemon_start(foreground: bool, idle_timeout: u64, cli: &Cli) -> Result<()> {
    // Check if daemon is already running
    if is_daemon_running() {
        println!("{} Daemon is already running", "✓".green());
        return Ok(());
    }

    // Clean up stale socket if exists
    let socket_path = get_socket_path();
    if socket_path.exists() {
        std::fs::remove_file(&socket_path).ok();
    }

    if foreground {
        // Run in foreground
        run_daemon_foreground(idle_timeout, cli)
    } else {
        // Fork and daemonize
        daemon_fork(idle_timeout)
    }
}

/// Fork and run daemon in background
#[cfg(unix)]
fn daemon_fork(idle_timeout: u64) -> Result<()> {
    use std::process::Command;

    // Get current executable path
    let exe = env::current_exe()?;

    // Spawn child process
    let mut child = Command::new(exe)
        .args([
            "daemon",
            "start",
            "--foreground",
            "--idle-timeout",
            &idle_timeout.to_string(),
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context("Failed to spawn daemon process")?;

    // Wait briefly for socket to become available
    let socket_path = get_socket_path();
    let start = Instant::now();
    let timeout = Duration::from_secs(10);

    while start.elapsed() < timeout {
        if socket_path.exists() && is_daemon_running() {
            println!(
                "{} {}",
                "✓".green(),
                "Daemon started successfully".green().bold()
            );
            println!("  {} {}", "Socket:".blue().bold(), socket_path.display());
            if let Some(pid) = read_daemon_pid() {
                println!("  {} {}", "PID:".blue().bold(), pid);
            }
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }

    // Daemon didn't start in time
    let _ = child.kill();
    bail!("Daemon failed to start within timeout")
}

/// Run daemon in foreground
#[cfg(unix)]
fn run_daemon_foreground(idle_timeout: u64, cli: &Cli) -> Result<()> {
    // Get model path
    let model_info = get_model_path(cli.model.as_ref())?;

    // Create socket
    let socket_path = get_socket_path();

    // Ensure parent directory exists
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Remove stale socket
    if socket_path.exists() {
        std::fs::remove_file(&socket_path)?;
    }

    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("Failed to create socket at {}", socket_path.display()))?;

    // Set socket permissions (owner only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&socket_path, std::fs::Permissions::from_mode(0o600))?;
    }

    // Write PID file
    let pid_path = get_pid_path();
    std::fs::write(&pid_path, std::process::id().to_string())?;

    println!("{} {}", "▸".cyan(), "Why Daemon".cyan().bold());
    println!("  {} {}", "Socket:".blue().bold(), socket_path.display());
    println!("  {} {}", "PID:".blue().bold(), std::process::id());
    println!(
        "  {} {} minutes",
        "Idle timeout:".blue().bold(),
        idle_timeout
    );
    println!();
    println!("Loading model...");

    // Load model
    let start = Instant::now();
    let backend = LlamaBackend::init()?;
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

    let model_params = LlamaModelParams::default();
    let model = LlamaModel::load_from_file(&backend, &model_info.path, &model_params)
        .context("Failed to load model")?;

    let load_time = start.elapsed();
    println!("Model loaded in {:.2}s", load_time.as_secs_f64());

    // Determine model family
    let model_family = if let Some(family) = cli.template {
        family
    } else if let Some(family) = model_info.embedded_family {
        family
    } else {
        detect_model_family(&model_info.path)
    };

    println!("  {} {:?}", "Model family:".blue().bold(), model_family);
    println!();
    println!("Daemon ready. Waiting for connections...");
    println!();

    // Stats tracking
    let daemon_start = Instant::now();
    let mut requests_served: u64 = 0;
    let mut total_response_time_ms: f64 = 0.0;
    let mut last_activity = Instant::now();

    // Set listener to non-blocking for idle timeout
    listener.set_nonblocking(true)?;

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    // Handle SIGTERM/SIGINT
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .ok();

    // Main loop
    while running.load(Ordering::SeqCst) {
        // Check idle timeout
        let idle_duration = Duration::from_secs(idle_timeout * 60);
        if last_activity.elapsed() > idle_duration {
            println!("Idle timeout reached. Shutting down...");
            break;
        }

        // Try to accept connection
        match listener.accept() {
            Ok((stream, _)) => {
                last_activity = Instant::now();
                let request_start = Instant::now();

                // Handle connection
                if let Err(e) = handle_daemon_connection(
                    stream,
                    &model,
                    &backend,
                    model_family,
                    &running,
                    daemon_start,
                    requests_served,
                    total_response_time_ms,
                ) {
                    eprintln!("Error handling connection: {}", e);
                }

                requests_served += 1;
                total_response_time_ms += request_start.elapsed().as_secs_f64() * 1000.0;
            }
            Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                // No connection available, sleep briefly
                thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                eprintln!("Error accepting connection: {}", e);
            }
        }
    }

    // Cleanup
    println!("Shutting down daemon...");
    std::fs::remove_file(&socket_path).ok();
    std::fs::remove_file(&pid_path).ok();
    println!("Daemon stopped.");

    Ok(())
}

/// Handle a single daemon connection
#[cfg(unix)]
#[allow(clippy::too_many_arguments)]
fn handle_daemon_connection(
    stream: UnixStream,
    model: &LlamaModel,
    backend: &LlamaBackend,
    model_family: ModelFamily,
    running: &Arc<AtomicBool>,
    daemon_start: Instant,
    requests_served: u64,
    total_response_time_ms: f64,
) -> Result<()> {
    use std::io::{BufRead, Write};

    stream.set_read_timeout(Some(Duration::from_secs(60)))?;
    stream.set_write_timeout(Some(Duration::from_secs(60)))?;

    let reader = std::io::BufReader::new(&stream);
    let mut writer = std::io::BufWriter::new(&stream);

    for line in reader.lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }

        let request: DaemonRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let response = DaemonResponse::error(&format!("Invalid request: {}", e));
                writeln!(writer, "{}", serde_json::to_string(&response)?)?;
                writer.flush()?;
                continue;
            }
        };

        match request.action {
            DaemonAction::Ping => {
                let response = DaemonResponse::pong();
                writeln!(writer, "{}", serde_json::to_string(&response)?)?;
                writer.flush()?;
            }
            DaemonAction::Shutdown => {
                let response = DaemonResponse::shutdown_ack();
                writeln!(writer, "{}", serde_json::to_string(&response)?)?;
                writer.flush()?;
                running.store(false, Ordering::SeqCst);
                return Ok(());
            }
            DaemonAction::Stats => {
                let uptime = daemon_start.elapsed().as_secs();
                let avg_time = if requests_served > 0 {
                    total_response_time_ms / requests_served as f64
                } else {
                    0.0
                };

                let stats = DaemonStats {
                    uptime_seconds: uptime,
                    requests_served,
                    avg_response_time_ms: avg_time,
                    memory_mb: 0.0, // Would need platform-specific code
                    model_family: format!("{:?}", model_family),
                    model_loaded: true,
                };
                let response = DaemonResponse::stats(stats);
                writeln!(writer, "{}", serde_json::to_string(&response)?)?;
                writer.flush()?;
            }
            DaemonAction::Explain => {
                let input = match request.input {
                    Some(i) => i,
                    None => {
                        let response = DaemonResponse::error("Missing input for explain action");
                        writeln!(writer, "{}", serde_json::to_string(&response)?)?;
                        writer.flush()?;
                        continue;
                    }
                };

                // Build prompt
                let prompt = build_prompt(&input, model_family);

                // Create context
                let ctx_params = LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(2048))
                    .with_n_batch(512);

                let mut ctx = model
                    .new_context(backend, ctx_params)
                    .context("Failed to create context")?;

                // Tokenize
                let tokens = model
                    .str_to_token(&prompt, AddBos::Always)
                    .context("Failed to tokenize")?;

                let mut batch = LlamaBatch::new(ctx.n_ctx() as usize, 1);
                for (i, token) in tokens.iter().enumerate() {
                    let is_last = i == tokens.len() - 1;
                    batch.add(*token, i as i32, &[0], is_last)?;
                }

                ctx.decode(&mut batch)?;

                // Generate response
                let mut sampler =
                    LlamaSampler::chain_simple([LlamaSampler::temp(0.7), LlamaSampler::dist(42)]);

                let stream_enabled = request.options.as_ref().map(|o| o.stream).unwrap_or(false);

                let mut response_text = String::new();
                let max_tokens = 512;

                for _ in 0..max_tokens {
                    let token = sampler.sample(&ctx, -1);
                    if model.is_eog_token(token) {
                        break;
                    }

                    let token_str = model.token_to_str(token, Special::Tokenize)?;
                    response_text.push_str(&token_str);

                    // Stream token if enabled
                    if stream_enabled {
                        let token_response = DaemonResponse::token(&token_str);
                        writeln!(writer, "{}", serde_json::to_string(&token_response)?)?;
                        writer.flush()?;
                    }

                    batch.clear();
                    batch.add(token, tokens.len() as i32, &[0], true)?;
                    ctx.decode(&mut batch)?;
                }

                // Parse and send final response
                let result = parse_response(&input, &response_text);
                let explanation = ErrorExplanationResponse::from(&result);
                let response = DaemonResponse::complete(explanation);
                writeln!(writer, "{}", serde_json::to_string(&response)?)?;
                writer.flush()?;
            }
        }

        // Only handle one request per connection for simplicity
        break;
    }

    Ok(())
}

/// Stop the daemon
#[cfg(unix)]
fn daemon_stop(force: bool) -> Result<()> {
    if !is_daemon_running() {
        println!("{} Daemon is not running", "?".yellow());
        return Ok(());
    }

    // Try graceful shutdown first
    let request = DaemonRequest {
        action: DaemonAction::Shutdown,
        input: None,
        options: None,
    };

    match send_daemon_request(&request) {
        Ok(responses) => {
            if responses
                .iter()
                .any(|r| r.response_type == DaemonResponseType::ShutdownAck)
            {
                println!(
                    "{} {}",
                    "✓".green(),
                    "Daemon stopped gracefully".green().bold()
                );

                // Wait for socket to disappear
                let socket_path = get_socket_path();
                let start = Instant::now();
                while socket_path.exists() && start.elapsed() < Duration::from_secs(5) {
                    thread::sleep(Duration::from_millis(100));
                }

                return Ok(());
            }
        }
        Err(e) => {
            eprintln!("Warning: Failed to send shutdown command: {}", e);
        }
    }

    // Graceful shutdown failed, try SIGTERM
    if let Some(pid) = read_daemon_pid() {
        eprintln!("Sending SIGTERM to PID {}...", pid);
        unsafe {
            libc::kill(pid as i32, libc::SIGTERM);
        }

        // Wait for process to exit
        let start = Instant::now();
        let timeout = Duration::from_millis(DAEMON_SHUTDOWN_GRACE_MS);
        while is_process_running(pid) && start.elapsed() < timeout {
            thread::sleep(Duration::from_millis(100));
        }

        if !is_process_running(pid) {
            println!("{} {}", "✓".green(), "Daemon stopped".green().bold());

            // Clean up files
            std::fs::remove_file(get_socket_path()).ok();
            std::fs::remove_file(get_pid_path()).ok();
            return Ok(());
        }

        // SIGTERM failed, try SIGKILL if force
        if force {
            eprintln!("Sending SIGKILL to PID {}...", pid);
            unsafe {
                libc::kill(pid as i32, libc::SIGKILL);
            }
            thread::sleep(Duration::from_millis(500));

            // Clean up files
            std::fs::remove_file(get_socket_path()).ok();
            std::fs::remove_file(get_pid_path()).ok();

            println!(
                "{} {}",
                "✓".yellow(),
                "Daemon killed forcefully".yellow().bold()
            );
            return Ok(());
        }
    }

    bail!("Failed to stop daemon")
}

/// Restart the daemon
#[cfg(unix)]
fn daemon_restart(foreground: bool, cli: &Cli) -> Result<()> {
    // Stop if running
    if is_daemon_running() {
        daemon_stop(false)?;
        // Wait a bit for cleanup
        thread::sleep(Duration::from_millis(500));
    }

    // Start
    daemon_start(foreground, 30, cli)
}

/// Show daemon status
#[cfg(unix)]
fn daemon_status() -> Result<()> {
    let socket_path = get_socket_path();
    let pid_path = get_pid_path();

    println!();
    println!("{} {}", "▸".cyan(), "Daemon Status".cyan().bold());

    // Check socket
    println!("  {} {}", "Socket:".blue().bold(), socket_path.display());
    println!(
        "    Exists: {}",
        if socket_path.exists() {
            "yes".green()
        } else {
            "no".red()
        }
    );

    // Check PID file
    println!("  {} {}", "PID file:".blue().bold(), pid_path.display());
    if let Some(pid) = read_daemon_pid() {
        println!("    PID: {}", pid);
        println!(
            "    Process running: {}",
            if is_process_running(pid) {
                "yes".green()
            } else {
                "no".red()
            }
        );
    } else {
        println!("    {}", "No PID file".dimmed());
    }

    // Check connectivity
    println!(
        "  {} {}",
        "Status:".blue().bold(),
        if is_daemon_running() {
            "Running".green().bold()
        } else {
            "Not running".red().bold()
        }
    );

    // Get stats if running
    if is_daemon_running() {
        let request = DaemonRequest {
            action: DaemonAction::Stats,
            input: None,
            options: None,
        };

        if let Ok(responses) = send_daemon_request(&request) {
            for response in responses {
                if let Some(stats) = response.stats {
                    println!();
                    println!(
                        "  {} {}",
                        "Uptime:".blue().bold(),
                        format_duration(stats.uptime_seconds)
                    );
                    println!(
                        "  {} {}",
                        "Requests served:".blue().bold(),
                        stats.requests_served
                    );
                    if stats.requests_served > 0 {
                        println!(
                            "  {} {:.1}ms",
                            "Avg response time:".blue().bold(),
                            stats.avg_response_time_ms
                        );
                    }
                    println!("  {} {}", "Model family:".blue().bold(), stats.model_family);
                }
            }
        }
    }

    println!();
    Ok(())
}

/// Format duration in human-readable form
fn format_duration(seconds: u64) -> String {
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    }
}

/// Install system service
#[cfg(unix)]
fn daemon_install_service() -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        install_launchd_service()
    }
    #[cfg(target_os = "linux")]
    {
        install_systemd_service()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        bail!("Service installation not supported on this platform")
    }
}

/// Install launchd service (macOS)
#[cfg(target_os = "macos")]
fn install_launchd_service() -> Result<()> {
    let plist_path = dirs::home_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?
        .join("Library")
        .join("LaunchAgents")
        .join("com.why.daemon.plist");

    // Get current executable path
    let exe = env::current_exe()?.display().to_string();

    let plist_content = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.why.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{}</string>
        <string>daemon</string>
        <string>start</string>
        <string>--foreground</string>
    </array>
    <key>RunAtLoad</key>
    <false/>
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
"#,
        exe
    );

    // Ensure directory exists
    if let Some(parent) = plist_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&plist_path, plist_content)?;

    println!(
        "{} {}",
        "✓".green(),
        "Launchd service installed".green().bold()
    );
    println!("  {} {}", "Plist:".blue().bold(), plist_path.display());
    println!();
    println!("  To load the service:");
    println!("    launchctl load {}", plist_path.display());
    println!();
    println!("  To start the service:");
    println!("    launchctl start com.why.daemon");
    println!();

    Ok(())
}

/// Install systemd service (Linux)
#[cfg(target_os = "linux")]
fn install_systemd_service() -> Result<()> {
    let service_path = dirs::config_dir()
        .ok_or_else(|| anyhow::anyhow!("Could not find config directory"))?
        .join("systemd")
        .join("user")
        .join("why.service");

    // Get current executable path
    let exe = env::current_exe()?.display().to_string();

    let service_content = format!(
        r#"[Unit]
Description=Why Error Explainer Daemon
After=network.target

[Service]
Type=simple
ExecStart={} daemon start --foreground
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"#,
        exe
    );

    // Ensure directory exists
    if let Some(parent) = service_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    std::fs::write(&service_path, service_content)?;

    println!(
        "{} {}",
        "✓".green(),
        "Systemd service installed".green().bold()
    );
    println!(
        "  {} {}",
        "Service file:".blue().bold(),
        service_path.display()
    );
    println!();
    println!("  To enable and start:");
    println!("    systemctl --user daemon-reload");
    println!("    systemctl --user enable why");
    println!("    systemctl --user start why");
    println!();

    Ok(())
}

/// Uninstall system service
#[cfg(unix)]
fn daemon_uninstall_service() -> Result<()> {
    #[cfg(target_os = "macos")]
    {
        let plist_path = dirs::home_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find home directory"))?
            .join("Library")
            .join("LaunchAgents")
            .join("com.why.daemon.plist");

        if plist_path.exists() {
            // Try to unload first
            std::process::Command::new("launchctl")
                .args(["unload", &plist_path.display().to_string()])
                .output()
                .ok();

            std::fs::remove_file(&plist_path)?;
            println!(
                "{} {}",
                "✓".green(),
                "Launchd service uninstalled".green().bold()
            );
        } else {
            println!("{} Service file not found", "?".yellow());
        }
    }
    #[cfg(target_os = "linux")]
    {
        let service_path = dirs::config_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not find config directory"))?
            .join("systemd")
            .join("user")
            .join("why.service");

        if service_path.exists() {
            // Try to stop and disable first
            std::process::Command::new("systemctl")
                .args(["--user", "stop", "why"])
                .output()
                .ok();
            std::process::Command::new("systemctl")
                .args(["--user", "disable", "why"])
                .output()
                .ok();

            std::fs::remove_file(&service_path)?;
            println!(
                "{} {}",
                "✓".green(),
                "Systemd service uninstalled".green().bold()
            );
        } else {
            println!("{} Service file not found", "?".yellow());
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        bail!("Service uninstallation not supported on this platform")
    }

    Ok(())
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

/// Embedded model info: offset, size, and optional model family
struct EmbeddedModelInfo {
    offset: u64,
    size: u64,
    family: Option<ModelFamily>,
}

fn find_embedded_model() -> Result<EmbeddedModelInfo> {
    let exe_path = env::current_exe().context("failed to get executable path")?;
    let mut file = File::open(&exe_path).context("failed to open self")?;
    let file_len = file.metadata()?.len();

    // Try new 25-byte trailer format first (with family byte)
    // Note: This correctly distinguishes from old 24-byte format because:
    // - New format: magic is at End(-25), so bytes [0..8] contain "WHYMODEL"
    // - Old format: magic is at End(-24), so when reading 25 bytes from End(-25),
    //   the magic would be at bytes [1..9], not [0..8], causing the check to fail
    //   and fall through to the old format handler below.
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

/// Result of getting model path, includes embedded family if available
struct ModelPathInfo {
    path: PathBuf,
    embedded_family: Option<ModelFamily>,
}

fn get_model_path(cli_model: Option<&PathBuf>) -> Result<ModelPathInfo> {
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
    // Include both old and new model filenames for compatibility
    let candidates = [
        PathBuf::from("model.gguf"),
        PathBuf::from("qwen2.5-coder-0.5b-instruct-q8_0.gguf"),
        PathBuf::from("qwen2.5-coder-0.5b.gguf"), // Old filename for compatibility
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

/// ChatML template for Qwen and SmolLM models
const TEMPLATE_CHATML: &str = include_str!("prompts/chatml.txt");

/// Gemma template using <start_of_turn> format
const TEMPLATE_GEMMA: &str = include_str!("prompts/gemma.txt");

/// Detect model family from filename/path
fn detect_model_family(model_path: &Path) -> ModelFamily {
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

fn build_prompt(error: &str, family: ModelFamily) -> String {
    let template = match family {
        ModelFamily::Gemma => TEMPLATE_GEMMA,
        ModelFamily::Qwen | ModelFamily::Smollm => TEMPLATE_CHATML,
    };
    template.replace("{error}", error.trim())
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

/// Run inference with optional streaming callback
/// If callback is provided, tokens are streamed to it as they're generated
fn run_inference_with_callback(
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

    let max_gen_tokens = 512;
    let mut n_cur = batch.n_tokens();
    let start_n = n_cur;
    let mut output = String::new();

    let generation_start = Instant::now();
    // Buffer for incomplete UTF-8 sequences
    let mut utf8_buffer: Vec<u8> = Vec::new();

    while can_generate_more(start_n, n_cur, max_gen_tokens) {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        let bytes = model.token_to_bytes(token, Special::Tokenize)?;

        // Buffer bytes and decode only complete UTF-8 sequences
        utf8_buffer.extend_from_slice(&bytes);

        // Try to decode as much valid UTF-8 as possible
        let mut token_str = String::new();
        match std::str::from_utf8(&utf8_buffer) {
            Ok(s) => {
                token_str = s.to_string();
                utf8_buffer.clear();
            }
            Err(e) => {
                // Decode the valid prefix
                let valid_up_to = e.valid_up_to();
                if valid_up_to > 0 {
                    token_str =
                        String::from_utf8(utf8_buffer[..valid_up_to].to_vec()).unwrap_or_default();
                    utf8_buffer = utf8_buffer[valid_up_to..].to_vec();
                }
                // If error_len is None, we need more bytes - keep buffering
            }
        }

        if !token_str.is_empty() {
            output.push_str(&token_str);

            // Invoke callback if provided
            if let Some(ref mut cb) = callback {
                match cb(&token_str) {
                    Ok(true) => {}           // Continue
                    Ok(false) => break,      // Stop requested
                    Err(e) => return Err(e), // Error
                }
            }
        }

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        ctx.decode(&mut batch)?;
        n_cur += 1;
    }

    // Flush any remaining bytes in the UTF-8 buffer (lossy)
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

/// Backward-compatible run_inference without callback
#[allow(dead_code)]
fn run_inference(
    model_path: &PathBuf,
    prompt: &str,
    params: &SamplingParams,
) -> Result<(String, InferenceStats)> {
    run_inference_with_callback(model_path, prompt, params, None)
}

fn print_completions(shell: Shell) {
    let mut cmd = Cli::command();
    generate(shell, &mut cmd, "why", &mut io::stdout());
}

/// Generate shell hook script for automatic error explanation
fn print_hook(shell: Shell) {
    match shell {
        Shell::Bash => print_bash_hook(),
        Shell::Zsh => print_zsh_hook(),
        Shell::Fish => print_fish_hook(),
        _ => {
            eprintln!(
                "{} Shell hooks are only supported for bash, zsh, and fish.",
                "Error:".red().bold()
            );
            std::process::exit(1);
        }
    }
}

fn print_bash_hook() {
    println!(
        r#"# why - shell hook for bash
# Add this to your ~/.bashrc:
#   eval "$(why --hook bash)"

__why_prompt_command() {{
    local exit_code=$?
    if [[ $exit_code -ne 0 && $exit_code -ne 130 ]]; then
        # 130 = Ctrl+C, don't explain
        why --exit-code $exit_code --last-command "$(fc -ln -1 2>/dev/null | sed 's/^[[:space:]]*//')"
    fi
    return $exit_code
}}

# Preserve existing PROMPT_COMMAND
if [[ -z "$PROMPT_COMMAND" ]]; then
    PROMPT_COMMAND="__why_prompt_command"
else
    PROMPT_COMMAND="__why_prompt_command; $PROMPT_COMMAND"
fi
"#
    );
}

fn print_zsh_hook() {
    println!(
        r#"# why - shell hook for zsh
# Add this to your ~/.zshrc:
#   eval "$(why --hook zsh)"

__why_precmd() {{
    local exit_code=$?
    if [[ $exit_code -ne 0 && $exit_code -ne 130 ]]; then
        # 130 = Ctrl+C, don't explain
        why --exit-code $exit_code --last-command "${{history[$HISTCMD]}}"
    fi
    return $exit_code
}}

# Add to precmd hooks
autoload -Uz add-zsh-hook
add-zsh-hook precmd __why_precmd
"#
    );
}

fn print_fish_hook() {
    println!(
        r#"# why - shell hook for fish
# Add this to your ~/.config/fish/config.fish:
#   why --hook fish | source

function __why_prompt --on-event fish_prompt
    set -l exit_code $status
    if test $exit_code -ne 0 -a $exit_code -ne 130
        # 130 = Ctrl+C, don't explain
        why --exit-code $exit_code --last-command "$history[1]"
    end
end
"#
    );
}

/// Interpret exit code to human-readable meaning
fn interpret_exit_code(code: i32) -> &'static str {
    match code {
        1 => "General error",
        2 => "Misuse of shell command",
        126 => "Command invoked cannot execute (permission problem or not executable)",
        127 => "Command not found",
        128 => "Invalid argument to exit",
        130 => "Script terminated by Ctrl+C",
        _ if code > 128 && code < 256 => {
            // Signal-based exit codes
            let signal = code - 128;
            match signal {
                1 => "Hangup (SIGHUP)",
                2 => "Interrupt (SIGINT)",
                3 => "Quit (SIGQUIT)",
                4 => "Illegal instruction (SIGILL)",
                6 => "Abort (SIGABRT)",
                8 => "Floating point exception (SIGFPE)",
                9 => "Kill (SIGKILL)",
                11 => "Segmentation fault (SIGSEGV)",
                13 => "Broken pipe (SIGPIPE)",
                14 => "Alarm (SIGALRM)",
                15 => "Termination (SIGTERM)",
                _ => "Killed by signal",
            }
        }
        _ => "Unknown error",
    }
}

#[allow(clippy::print_literal)]
fn print_model_list() {
    println!("{}", "Available Model Variants".bold());
    println!();
    println!(
        "These models can be built with {} or used with {}:",
        "nix build .#<variant>".cyan(),
        "--model".cyan()
    );
    println!();
    println!(
        "  {:<20} {:<12} {}",
        "Variant".blue().bold(),
        "Size".blue().bold(),
        "Description".blue().bold()
    );
    println!(
        "  {:<20} {:<12} {}",
        "───────────────────", "──────────", "─────────────────────────────────────"
    );
    println!(
        "  {:<20} {:<12} {} {}",
        "why-qwen2_5-coder",
        "~530MB",
        "Qwen2.5-Coder 0.5B - best quality",
        "(default)".dimmed()
    );
    println!(
        "  {:<20} {:<12} {}",
        "why-qwen3", "~639MB", "Qwen3 0.6B - newest Qwen"
    );
    println!(
        "  {:<20} {:<12} {}",
        "why-gemma3", "~292MB", "Gemma 3 270M - Google"
    );
    println!(
        "  {:<20} {:<12} {}",
        "why-smollm2", "~145MB", "SmolLM2 135M - smallest/fastest"
    );
    println!();
    println!("{}", "Template Families".bold());
    println!();
    println!(
        "  Use {} to override auto-detection:",
        "--template <family>".cyan()
    );
    println!();
    println!(
        "  {:<12} {}",
        "qwen".green(),
        "ChatML format (Qwen, SmolLM)"
    );
    println!(
        "  {:<12} {}",
        "gemma".green(),
        "Gemma format (<start_of_turn>)"
    );
    println!("  {:<12} {}", "smollm".green(), "ChatML format (alias)");
    println!();
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

    // Handle --hook
    if let Some(shell) = cli.hook {
        print_hook(shell);
        return Ok(());
    }

    // Handle --list-models
    if cli.list_models {
        print_model_list();
        return Ok(());
    }

    // Handle --hook-config
    if cli.hook_config {
        print_hook_config();
        return Ok(());
    }

    // Handle --hook-install
    if let Some(shell) = cli.hook_install {
        return install_hook(shell);
    }

    // Handle --hook-uninstall
    if let Some(shell) = cli.hook_uninstall {
        return uninstall_hook(shell);
    }

    // Handle --watch mode
    if let Some(ref target) = cli.watch {
        let model_info = get_model_path(cli.model.as_ref())?;
        return run_watch_mode(target, &cli, &model_info);
    }

    // Handle daemon subcommand
    if let Some(ref daemon_cmd) = cli.daemon {
        return handle_daemon_command(daemon_cmd, &cli);
    }

    // Load configuration (with env overrides)
    let mut config = Config::load();
    config.apply_env_overrides();

    // Check if hook is disabled via environment variable
    if Config::is_hook_disabled() && (cli.capture || cli.exit_code.is_some()) {
        // Hook mode is disabled, just pass through
        return Ok(());
    }

    // Handle --capture mode
    if cli.capture {
        let result = run_capture_command(&cli.error, cli.capture_all)?;

        // Check if this exit code should be skipped (from config)
        if config.should_skip_exit_code(result.exit_code) {
            return Ok(());
        }

        // Check if command matches ignore patterns
        if config.should_ignore_command(&result.command) {
            return Ok(());
        }

        // Build input from captured output
        let captured_output = if cli.capture_all && !result.stdout.is_empty() {
            format!("{}\n{}", result.stdout, result.stderr)
        } else {
            result.stderr.clone()
        };

        // If no output captured, just report the exit code
        if captured_output.trim().is_empty() {
            let interpretation = interpret_exit_code(result.exit_code);
            println!();
            println!(
                "{} {} (exit {})",
                "Command failed:".red().bold(),
                interpretation,
                result.exit_code
            );
            return Ok(());
        }

        // Handle confirmation mode
        // Priority: --auto CLI flag > config auto_explain > --confirm CLI flag
        let auto_explain = cli.auto || config.hook.auto_explain;
        if cli.confirm && !auto_explain {
            // Check if stdin is a terminal for interactive prompting
            if std::io::stdin().is_terminal() {
                if !prompt_confirm(&result.command, result.exit_code, &captured_output) {
                    return Ok(());
                }
            } else {
                // Non-interactive: check for error patterns to decide
                if !contains_error_patterns(&captured_output) {
                    // No obvious errors and non-interactive, skip
                    return Ok(());
                }
            }
        }

        // Check min_stderr_lines from config
        if captured_output.lines().count() < config.hook.min_stderr_lines {
            return Ok(());
        }

        // Build enhanced input with command context
        let input = format!(
            "Command: {}\nExit code: {} ({})\n\nOutput:\n{}",
            result.command,
            result.exit_code,
            interpret_exit_code(result.exit_code),
            captured_output.trim()
        );

        // Now run the normal explanation flow with this input
        // Parse stack trace from captured output
        let registry = StackTraceParserRegistry::with_builtins();
        let parsed_stack_trace = registry.parse(&captured_output);

        // If --show-frames is requested, display parsed frames
        if cli.show_frames {
            if let Some(ref trace) = parsed_stack_trace {
                print_frames(trace);
            }
        }

        let model_info = get_model_path(cli.model.as_ref())?;
        let model_path = &model_info.path;

        let (model_family, _family_source) = if let Some(family) = cli.template {
            (family, "override".to_string())
        } else if let Some(family) = model_info.embedded_family {
            (family, "embedded".to_string())
        } else {
            let detected = detect_model_family(model_path);
            let filename = model_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            (detected, format!("auto-detected from '{}'", filename))
        };

        let prompt = build_prompt(&input, model_family);

        // Run inference
        let callback: Option<TokenCallback> = if cli.stream && !cli.json {
            Some(Box::new(|token: &str| {
                print!("{}", token);
                io::stdout().flush().ok();
                Ok(true)
            }))
        } else {
            None
        };

        let (response, stats) =
            run_inference_with_callback(model_path, &prompt, &SamplingParams::default(), callback)?;

        if cli.stream && !cli.json {
            println!();
            println!();
        }

        let parsed = parse_response(&input, &response);
        let has_content = !parsed.summary.is_empty()
            || !parsed.explanation.is_empty()
            || !parsed.suggestion.is_empty();

        if cli.json {
            let mut payload = serde_json::json!({
                "command": result.command,
                "exit_code": result.exit_code,
                "captured_output": captured_output.trim(),
                "summary": parsed.summary,
                "explanation": parsed.explanation,
                "suggestion": parsed.suggestion
            });
            if let Some(ref trace) = parsed_stack_trace {
                payload["stack_trace"] = serde_json::to_value(StackTraceJson::from(trace))?;
            }
            if cli.stats {
                payload["stats"] = serde_json::to_value(&stats)?;
            }
            println!("{}", serde_json::to_string_pretty(&payload)?);
        } else {
            println!();
            println!(
                "{} {} {}",
                "▸".red(),
                "Command failed:".red().bold(),
                result.command.bright_white()
            );
            println!(
                "  {} {} (exit {})",
                "Status:".blue().bold(),
                interpret_exit_code(result.exit_code),
                result.exit_code
            );
            println!();

            if has_content {
                // Print file:line highlighting if we have a root cause frame
                if let Some(ref trace) = parsed_stack_trace {
                    if let Some(root_frame) = trace.root_cause_frame() {
                        if let Some(ref file) = root_frame.file {
                            println!("{} {}", "▸".cyan(), "Location".cyan().bold());
                            println!(
                                "  {}",
                                format_file_line(file, root_frame.line, root_frame.column)
                            );
                            println!();
                        }
                    }
                }
                print_colored(&parsed);
            }
            if cli.stats {
                print_stats(&stats);
            }
        }

        return Ok(());
    }

    // Build input - enhanced for hook mode
    let input = if let (Some(exit_code), Some(ref command)) = (cli.exit_code, &cli.last_command) {
        // Hook mode: build enhanced prompt with command context
        let interpretation = interpret_exit_code(exit_code);
        format!(
            "Command: {}\nExit code: {} ({})\n\nExplain why this command failed.",
            command.trim(),
            exit_code,
            interpretation
        )
    } else if cli.exit_code.is_some() || cli.last_command.is_some() {
        // Partial hook mode - try to use what we have
        if let Some(exit_code) = cli.exit_code {
            let interpretation = interpret_exit_code(exit_code);
            format!(
                "Exit code: {} ({})\n\nExplain what this exit code means.",
                exit_code, interpretation
            )
        } else if let Some(ref command) = cli.last_command {
            format!(
                "Command failed: {}\n\nExplain why this might have failed.",
                command.trim()
            )
        } else {
            get_input(&cli)?
        }
    } else {
        get_input(&cli)?
    };

    // Parse stack trace from input (if present)
    let registry = StackTraceParserRegistry::with_builtins();
    let parsed_stack_trace = registry.parse(&input);

    // If --show-frames is requested, display parsed frames immediately (even if no inference needed)
    if cli.show_frames {
        if let Some(ref trace) = parsed_stack_trace {
            print_frames(trace);
        } else {
            println!();
            println!(
                "{} {}",
                "?".yellow(),
                "No stack trace detected in input".yellow().bold()
            );
            println!();
            println!(
                "  {}",
                "The input does not appear to contain a recognized stack trace format.".dimmed()
            );
            println!();
        }
    }

    let model_info = get_model_path(cli.model.as_ref())?;
    let model_path = &model_info.path;

    // Determine model family: CLI override > embedded family > auto-detect from path
    let (model_family, family_source) = if let Some(family) = cli.template {
        (family, "override".to_string())
    } else if let Some(family) = model_info.embedded_family {
        (family, "embedded".to_string())
    } else {
        let detected = detect_model_family(model_path);
        let filename = model_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        (detected, format!("auto-detected from '{}'", filename))
    };
    let prompt = build_prompt(&input, model_family);

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
        eprintln!(
            "{} {} ({})",
            "Family:".blue().bold(),
            model_family,
            family_source
        );
        match std::fs::metadata(model_path) {
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
        // Create streaming callback if streaming mode is enabled
        let callback: Option<TokenCallback> = if cli.stream && !cli.json {
            Some(Box::new(|token: &str| {
                print!("{}", token);
                io::stdout().flush().ok();
                Ok(true)
            }))
        } else {
            None
        };

        (response, stats) = run_inference_with_callback(model_path, &prompt, &params, callback)?;

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

    // Add newline after streaming output
    if cli.stream && !cli.json {
        println!();
        println!();
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
            // Include stack trace data if parsed
            if let Some(ref trace) = parsed_stack_trace {
                payload["stack_trace"] = serde_json::to_value(StackTraceJson::from(trace))?;
            }
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
            // Include stack trace data if parsed
            if let Some(ref trace) = parsed_stack_trace {
                payload["stack_trace"] = serde_json::to_value(StackTraceJson::from(trace))?;
            }
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
        // Include stack trace data if parsed
        if let Some(ref trace) = parsed_stack_trace {
            payload["stack_trace"] = serde_json::to_value(StackTraceJson::from(trace))?;
        }
        if cli.stats {
            payload["stats"] = serde_json::to_value(&stats)?;
        }
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        // Print file:line highlighting if we have a root cause frame
        if let Some(ref trace) = parsed_stack_trace {
            if let Some(root_frame) = trace.root_cause_frame() {
                if let Some(ref file) = root_frame.file {
                    println!();
                    println!("{} {}", "▸".cyan(), "Location".cyan().bold());
                    println!(
                        "  {}",
                        format_file_line(file, root_frame.line, root_frame.column)
                    );
                }
            }
        }
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
        let prompt = build_prompt("segmentation fault", ModelFamily::Qwen);
        assert!(prompt.contains("segmentation fault"));
        assert!(prompt.contains("<|im_start|>"));
        assert!(prompt.contains("SUMMARY:"));
    }

    #[test]
    fn test_build_prompt_trims_whitespace() {
        let prompt = build_prompt("  error with spaces  ", ModelFamily::Qwen);
        assert!(prompt.contains("error with spaces"));
        assert!(!prompt.contains("  error")); // leading spaces trimmed
    }

    #[test]
    fn test_build_prompt_gemma_template() {
        let prompt = build_prompt("test error", ModelFamily::Gemma);
        assert!(prompt.contains("test error"));
        assert!(prompt.contains("<start_of_turn>"));
        assert!(prompt.contains("<end_of_turn>"));
        assert!(!prompt.contains("<|im_start|>")); // No ChatML tokens
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
        assert_eq!(detect_model_family(&path), ModelFamily::Qwen); // Default
    }

    #[test]
    fn test_model_family_display() {
        assert_eq!(format!("{}", ModelFamily::Qwen), "qwen (ChatML)");
        assert_eq!(format!("{}", ModelFamily::Gemma), "gemma (Gemma format)");
        assert_eq!(format!("{}", ModelFamily::Smollm), "smollm (ChatML)");
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
        let prompt = build_prompt(&long_error, ModelFamily::Qwen);
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

        let prompt = build_prompt(multiline_error, ModelFamily::Qwen);

        assert!(prompt.contains("error[E0382]"));
        assert!(prompt.contains("src/main.rs:10:5"));
        assert!(prompt.contains("value borrowed here after move"));
    }

    #[test]
    fn test_build_prompt_preserves_newlines() {
        let input = "line1\nline2\nline3";
        let prompt = build_prompt(input, ModelFamily::Qwen);

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
        let prompt = build_prompt(&multiline, ModelFamily::Qwen);

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
        let prompt = build_prompt(input, ModelFamily::Qwen);

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

    #[test]
    fn test_cli_parses_model_flag() {
        let cli = Cli::parse_from(["why", "--model", "/path/to/model.gguf", "error"]);
        assert_eq!(cli.model, Some(PathBuf::from("/path/to/model.gguf")));
    }

    #[test]
    fn test_cli_parses_short_model_flag() {
        let cli = Cli::parse_from(["why", "-m", "/path/to/model.gguf", "error"]);
        assert_eq!(cli.model, Some(PathBuf::from("/path/to/model.gguf")));
    }

    #[test]
    fn test_cli_parses_template_flag() {
        let cli = Cli::parse_from(["why", "--template", "gemma", "error"]);
        assert_eq!(cli.template, Some(ModelFamily::Gemma));
    }

    #[test]
    fn test_cli_parses_short_template_flag() {
        let cli = Cli::parse_from(["why", "-t", "qwen", "error"]);
        assert_eq!(cli.template, Some(ModelFamily::Qwen));
    }

    #[test]
    fn test_cli_parses_template_smollm() {
        let cli = Cli::parse_from(["why", "--template", "smollm", "error"]);
        assert_eq!(cli.template, Some(ModelFamily::Smollm));
    }

    #[test]
    fn test_cli_parses_list_models_flag() {
        let cli = Cli::parse_from(["why", "--list-models"]);
        assert!(cli.list_models);
    }

    #[test]
    fn test_cli_model_and_template_together() {
        let cli = Cli::parse_from([
            "why",
            "--model",
            "/path/to/gemma.gguf",
            "--template",
            "gemma",
            "error",
        ]);
        assert_eq!(cli.model, Some(PathBuf::from("/path/to/gemma.gguf")));
        assert_eq!(cli.template, Some(ModelFamily::Gemma));
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

    // Phase 1.1: Core Streaming Infrastructure tests
    #[test]
    fn test_token_callback_invoked() {
        // Verify the callback type compiles and can be created
        let mut tokens_received: Vec<String> = Vec::new();
        let callback: TokenCallback = Box::new(|token: &str| {
            tokens_received.push(token.to_string());
            Ok(true)
        });
        // Callback is created successfully - full integration test would require model
        drop(callback);
        assert!(tokens_received.is_empty()); // No tokens yet, just testing type
    }

    #[test]
    fn test_callback_return_values() {
        // Test callback can return different values
        let mut continue_cb: TokenCallback = Box::new(|_| Ok(true));
        let mut stop_cb: TokenCallback = Box::new(|_| Ok(false));
        let mut error_cb: TokenCallback = Box::new(|_| Err(anyhow::anyhow!("test error")));

        assert!(continue_cb("test").unwrap());
        assert!(!stop_cb("test").unwrap());
        assert!(error_cb("test").is_err());
    }

    #[test]
    fn test_cli_parses_stream_flag() {
        let cli = Cli::parse_from(["why", "--stream", "error"]);
        assert!(cli.stream);
    }

    #[test]
    fn test_cli_parses_short_stream_flag() {
        let cli = Cli::parse_from(["why", "-s", "error"]);
        assert!(cli.stream);
    }

    #[test]
    fn test_cli_stream_and_json_together() {
        let cli = Cli::parse_from(["why", "-s", "-j", "error"]);
        assert!(cli.stream);
        assert!(cli.json);
    }

    // Phase 4.1: Shell Integration Hooks tests
    #[test]
    fn test_cli_parses_hook_bash() {
        let cli = Cli::parse_from(["why", "--hook", "bash"]);
        assert_eq!(cli.hook, Some(Shell::Bash));
    }

    #[test]
    fn test_cli_parses_hook_zsh() {
        let cli = Cli::parse_from(["why", "--hook", "zsh"]);
        assert_eq!(cli.hook, Some(Shell::Zsh));
    }

    #[test]
    fn test_cli_parses_hook_fish() {
        let cli = Cli::parse_from(["why", "--hook", "fish"]);
        assert_eq!(cli.hook, Some(Shell::Fish));
    }

    #[test]
    fn test_cli_parses_exit_code() {
        let cli = Cli::parse_from(["why", "--exit-code", "127"]);
        assert_eq!(cli.exit_code, Some(127));
    }

    #[test]
    fn test_cli_parses_last_command() {
        let cli = Cli::parse_from(["why", "--last-command", "npm run build"]);
        assert_eq!(cli.last_command, Some("npm run build".to_string()));
    }

    #[test]
    fn test_cli_parses_hook_mode_full() {
        let cli = Cli::parse_from(["why", "--exit-code", "1", "--last-command", "cargo build"]);
        assert_eq!(cli.exit_code, Some(1));
        assert_eq!(cli.last_command, Some("cargo build".to_string()));
    }

    // Phase 4.2: Exit code interpretation tests
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
        // 128 + 9 = 137 (SIGKILL)
        assert!(interpret_exit_code(137).contains("Kill"));
    }

    #[test]
    fn test_interpret_exit_code_sigsegv() {
        // 128 + 11 = 139 (SIGSEGV)
        assert!(interpret_exit_code(139).contains("Segmentation fault"));
    }

    #[test]
    fn test_interpret_exit_code_unknown() {
        assert_eq!(interpret_exit_code(42), "Unknown error");
    }

    // ========================================================================
    // Phase 3.1: Stack Trace Parser Framework tests
    // ========================================================================

    #[test]
    fn test_stack_trace_struct_creation() {
        let trace = StackTrace::new(Language::Python, "Traceback...");
        assert_eq!(trace.language, Language::Python);
        assert!(trace.error_type.is_empty());
        assert!(trace.error_message.is_empty());
        assert!(trace.frames.is_empty());
        assert_eq!(trace.raw_text, "Traceback...");
    }

    #[test]
    fn test_stack_trace_with_error_info() {
        let trace = StackTrace::new(Language::Python, "test")
            .with_error_type("KeyError")
            .with_error_message("'missing_key'");

        assert_eq!(trace.error_type, "KeyError");
        assert_eq!(trace.error_message, "'missing_key'");
    }

    #[test]
    fn test_stack_frame_optional_fields() {
        // Test with all fields None
        let frame = StackFrame::new();
        assert!(frame.function.is_none());
        assert!(frame.file.is_none());
        assert!(frame.line.is_none());
        assert!(frame.column.is_none());
        assert!(frame.is_user_code); // default true
        assert!(frame.context.is_none());
    }

    #[test]
    fn test_stack_frame_builder_pattern() {
        let frame = StackFrame::new()
            .with_function("main")
            .with_file("/src/main.py")
            .with_line(42)
            .with_column(10)
            .with_is_user_code(true)
            .with_context("x = dict['key']");

        assert_eq!(frame.function, Some("main".to_string()));
        assert_eq!(frame.file, Some(PathBuf::from("/src/main.py")));
        assert_eq!(frame.line, Some(42));
        assert_eq!(frame.column, Some(10));
        assert!(frame.is_user_code);
        assert_eq!(frame.context, Some("x = dict['key']".to_string()));
    }

    #[test]
    fn test_stack_frame_default() {
        let frame: StackFrame = Default::default();
        assert!(frame.function.is_none());
        assert!(frame.is_user_code);
    }

    #[test]
    fn test_stack_trace_add_frame() {
        let mut trace = StackTrace::new(Language::Rust, "panic");
        trace.add_frame(StackFrame::new().with_function("main"));
        trace.add_frame(StackFrame::new().with_function("process"));

        assert_eq!(trace.frames.len(), 2);
        assert_eq!(trace.frames[0].function, Some("main".to_string()));
        assert_eq!(trace.frames[1].function, Some("process".to_string()));
    }

    #[test]
    fn test_stack_trace_root_cause_frame_user_code() {
        let mut trace = StackTrace::new(Language::Python, "test");
        trace.add_frame(
            StackFrame::new()
                .with_function("stdlib_func")
                .with_is_user_code(false),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("user_func")
                .with_is_user_code(true),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("framework")
                .with_is_user_code(false),
        );

        let root = trace.root_cause_frame().unwrap();
        assert_eq!(root.function, Some("user_func".to_string()));
    }

    #[test]
    fn test_stack_trace_root_cause_frame_no_user_code() {
        let mut trace = StackTrace::new(Language::Python, "test");
        trace.add_frame(
            StackFrame::new()
                .with_function("stdlib")
                .with_is_user_code(false),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("framework")
                .with_is_user_code(false),
        );

        // Falls back to first frame
        let root = trace.root_cause_frame().unwrap();
        assert_eq!(root.function, Some("stdlib".to_string()));
    }

    #[test]
    fn test_stack_trace_user_frames() {
        let mut trace = StackTrace::new(Language::Python, "test");
        trace.add_frame(
            StackFrame::new()
                .with_function("stdlib")
                .with_is_user_code(false),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("user1")
                .with_is_user_code(true),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("user2")
                .with_is_user_code(true),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("framework")
                .with_is_user_code(false),
        );

        let user_frames = trace.user_frames();
        assert_eq!(user_frames.len(), 2);
        assert_eq!(user_frames[0].function, Some("user1".to_string()));
        assert_eq!(user_frames[1].function, Some("user2".to_string()));
    }

    #[test]
    fn test_parser_trait_implementation() {
        // Verify PythonStackTraceParser implements the trait
        let parser = PythonStackTraceParser;
        assert_eq!(parser.language(), Language::Python);
    }

    #[test]
    fn test_parser_registry_lookup() {
        let registry = StackTraceParserRegistry::with_builtins();

        let python = registry.get_parser(Language::Python);
        assert!(python.is_some());
        assert_eq!(python.unwrap().language(), Language::Python);

        let rust = registry.get_parser(Language::Rust);
        assert!(rust.is_some());
        assert_eq!(rust.unwrap().language(), Language::Rust);
    }

    #[test]
    fn test_parser_registry_empty() {
        let registry = StackTraceParserRegistry::new();
        assert!(registry.get_parser(Language::Python).is_none());
    }

    #[test]
    fn test_auto_detection_python() {
        let registry = StackTraceParserRegistry::with_builtins();
        let input = "Traceback (most recent call last):\n  File \"test.py\", line 1, in <module>\nKeyError: 'x'";

        assert_eq!(registry.detect_language(input), Language::Python);
    }

    #[test]
    fn test_auto_detection_rust() {
        let registry = StackTraceParserRegistry::with_builtins();
        let input = "thread 'main' panicked at 'index out of bounds', src/main.rs:10:5";

        assert_eq!(registry.detect_language(input), Language::Rust);
    }

    #[test]
    fn test_auto_detection_rust_compiler() {
        let registry = StackTraceParserRegistry::with_builtins();
        let input = "error[E0382]: borrow of moved value\n --> src/main.rs:10:5";

        assert_eq!(registry.detect_language(input), Language::Rust);
    }

    #[test]
    fn test_auto_detection_javascript() {
        let registry = StackTraceParserRegistry::with_builtins();
        let input =
            "TypeError: Cannot read property 'x' of undefined\n    at func (/app/index.js:10:5)";

        assert_eq!(registry.detect_language(input), Language::JavaScript);
    }

    #[test]
    fn test_auto_detection_go() {
        let registry = StackTraceParserRegistry::with_builtins();
        let input = "panic: runtime error\n\ngoroutine 1 [running]:\nmain.go:10";

        assert_eq!(registry.detect_language(input), Language::Go);
    }

    #[test]
    fn test_auto_detection_java() {
        let registry = StackTraceParserRegistry::with_builtins();
        let input = "Exception in thread \"main\" java.lang.NullPointerException\n\tat Main.main(Main.java:10)";

        assert_eq!(registry.detect_language(input), Language::Java);
    }

    #[test]
    fn test_auto_detection_cpp() {
        let registry = StackTraceParserRegistry::with_builtins();
        let input = "#0  0x00007fff main () in /app/main\n#1  0x00007fff at main.cpp:10";

        assert_eq!(registry.detect_language(input), Language::Cpp);
    }

    #[test]
    fn test_auto_detection_unknown() {
        let registry = StackTraceParserRegistry::with_builtins();
        let input = "Some random error message without stack trace patterns";

        assert_eq!(registry.detect_language(input), Language::Unknown);
    }

    #[test]
    fn test_registry_parse_returns_stack_trace() {
        let registry = StackTraceParserRegistry::with_builtins();
        let input = "Traceback (most recent call last):\n  File \"test.py\", line 1\nKeyError";

        let result = registry.parse(input);
        assert!(result.is_some());
        let trace = result.unwrap();
        assert_eq!(trace.language, Language::Python);
        assert_eq!(trace.raw_text, input);
    }

    #[test]
    fn test_registry_parse_returns_none_for_unknown() {
        let registry = StackTraceParserRegistry::with_builtins();
        let input = "Just a regular error message";

        let result = registry.parse(input);
        assert!(result.is_none());
    }

    #[test]
    fn test_language_display() {
        assert_eq!(format!("{}", Language::Python), "python");
        assert_eq!(format!("{}", Language::Rust), "rust");
        assert_eq!(format!("{}", Language::JavaScript), "javascript");
        assert_eq!(format!("{}", Language::Go), "go");
        assert_eq!(format!("{}", Language::Java), "java");
        assert_eq!(format!("{}", Language::Cpp), "c/c++");
        assert_eq!(format!("{}", Language::Unknown), "unknown");
    }

    #[test]
    fn test_language_serialization() {
        // Verify Language serializes to lowercase
        let lang = Language::Python;
        let json = serde_json::to_string(&lang).unwrap();
        assert_eq!(json, "\"python\"");

        let lang = Language::JavaScript;
        let json = serde_json::to_string(&lang).unwrap();
        assert_eq!(json, "\"javascript\"");
    }

    #[test]
    fn test_stack_frame_serialization() {
        let frame = StackFrame::new()
            .with_function("test_func")
            .with_file("/app/main.py")
            .with_line(42);

        let json = serde_json::to_string(&frame).unwrap();
        assert!(json.contains("\"function\":\"test_func\""));
        assert!(json.contains("\"line\":42"));
    }

    #[test]
    fn test_stack_trace_serialization() {
        let mut trace = StackTrace::new(Language::Python, "test")
            .with_error_type("KeyError")
            .with_error_message("missing");
        trace.add_frame(StackFrame::new().with_function("main"));

        let json = serde_json::to_string(&trace).unwrap();
        assert!(json.contains("\"language\":\"python\""));
        assert!(json.contains("\"error_type\":\"KeyError\""));
        assert!(json.contains("\"frames\":["));
    }

    // ========================================================================
    // Phase 3.2: Language-Specific Parser Tests
    // ========================================================================

    // Python Parser Tests
    #[test]
    fn test_python_traceback_basic() {
        let input = r#"Traceback (most recent call last):
  File "test.py", line 10, in main
    result = process()
  File "test.py", line 5, in process
    return data['missing']
KeyError: 'missing'"#;

        let parser = PythonStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::Python);
        assert_eq!(trace.error_type, "KeyError");
        assert_eq!(trace.error_message, "'missing'");
        assert_eq!(trace.frames.len(), 2);
        assert_eq!(trace.frames[0].function, Some("main".to_string()));
        assert_eq!(trace.frames[0].line, Some(10));
    }

    #[test]
    fn test_python_traceback_chained() {
        let input = r#"Traceback (most recent call last):
  File "test.py", line 10, in main
    process()
ValueError: first error

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "test.py", line 15, in handler
    raise RuntimeError("second")
RuntimeError: second"#;

        let parser = PythonStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::Python);
        // Should capture the last/outer exception
        assert!(trace.error_type.contains("Error"));
    }

    #[test]
    fn test_python_extract_file_line() {
        let input = r#"Traceback (most recent call last):
  File "/path/to/app/main.py", line 42, in run_app
    do_something()
TypeError: unsupported operand"#;

        let parser = PythonStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.frames.len(), 1);
        let frame = &trace.frames[0];
        assert_eq!(frame.file, Some(PathBuf::from("/path/to/app/main.py")));
        assert_eq!(frame.line, Some(42));
        assert_eq!(frame.function, Some("run_app".to_string()));
        assert!(frame.is_user_code);
    }

    #[test]
    fn test_python_framework_detection() {
        let input = r#"Traceback (most recent call last):
  File "/usr/lib/python3.9/site-packages/flask/app.py", line 100, in handle
    response = self.dispatch()
  File "/home/user/app/views.py", line 10, in index
    return data['key']
KeyError: 'key'"#;

        let parser = PythonStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.frames.len(), 2);
        // First frame is framework code (site-packages)
        assert!(!trace.frames[0].is_user_code);
        // Second frame is user code
        assert!(trace.frames[1].is_user_code);
    }

    // Rust Parser Tests
    #[test]
    fn test_rust_panic_basic() {
        let input = "thread 'main' panicked at 'index out of bounds: the len is 3 but the index is 10', src/main.rs:42:5";

        let parser = RustStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::Rust);
        assert_eq!(trace.error_type, "panic");
        assert!(trace.error_message.contains("index out of bounds"));
    }

    #[test]
    fn test_rust_compiler_error() {
        let input = r#"error[E0382]: borrow of moved value: `x`
 --> src/main.rs:10:5
  |
8 |     let y = x;
  |             - value moved here
10|     println!("{}", x);
  |                    ^ value borrowed here after move"#;

        let parser = RustStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::Rust);
        assert!(trace.error_type.contains("E0382"));
        assert!(trace.error_message.contains("borrow of moved value"));
        assert!(!trace.frames.is_empty());
    }

    #[test]
    fn test_rust_backtrace_full() {
        let input = r#"thread 'main' panicked at 'called Option::unwrap() on a None value', src/main.rs:10:5
stack backtrace:
   0: rust_begin_unwind
   1: core::panicking::panic_fmt
   2: core::panicking::panic
   3: my_app::main
   4: std::rt::lang_start"#;

        let parser = RustStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::Rust);
        assert!(trace.frames.len() >= 2);
        // std:: functions should be marked as non-user code
        let std_func = trace.frames.iter().find(|f| {
            f.function
                .as_ref()
                .map(|s| s.starts_with("std::"))
                .unwrap_or(false)
        });
        assert!(std_func.map(|f| !f.is_user_code).unwrap_or(true));
        // core:: functions should be marked as non-user code
        let core_func = trace.frames.iter().find(|f| {
            f.function
                .as_ref()
                .map(|s| s.starts_with("core::"))
                .unwrap_or(false)
        });
        assert!(core_func.map(|f| !f.is_user_code).unwrap_or(true));
    }

    // JavaScript Parser Tests
    #[test]
    fn test_node_error_basic() {
        let input = r#"TypeError: Cannot read property 'name' of undefined
    at getUser (/app/src/user.js:42:15)
    at processRequest (/app/src/handler.js:10:5)
    at Layer.handle (/app/node_modules/express/lib/router/layer.js:95:5)"#;

        let parser = JavaScriptStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::JavaScript);
        assert_eq!(trace.error_type, "TypeError");
        assert!(trace.error_message.contains("Cannot read property"));
        assert!(trace.frames.len() >= 2);
        // First frame should be user code
        assert!(trace.frames[0].is_user_code);
        // Last frame with node_modules should be framework
        assert!(!trace.frames[2].is_user_code);
    }

    #[test]
    fn test_node_async_stack() {
        let input = r#"ReferenceError: x is not defined
    at async processData (/app/index.js:25:10)
    at async main (/app/index.js:10:5)"#;

        let parser = JavaScriptStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::JavaScript);
        assert_eq!(trace.error_type, "ReferenceError");
        assert!(trace.frames.len() >= 1);
    }

    // Go Parser Tests
    #[test]
    fn test_go_panic_goroutine() {
        let input = r#"panic: runtime error: index out of range [10] with length 3

goroutine 1 [running]:
main.process()
	/home/user/app/main.go:42 +0x1a
main.main()
	/home/user/app/main.go:10 +0x2b"#;

        let parser = GoStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::Go);
        assert_eq!(trace.error_type, "panic");
        assert!(trace.error_message.contains("index out of range"));
        assert!(trace.frames.len() >= 1);
    }

    // Java Parser Tests
    #[test]
    fn test_java_exception_basic() {
        let input = r#"Exception in thread "main" java.lang.NullPointerException: Cannot invoke method on null
	at com.example.App.process(App.java:42)
	at com.example.App.main(App.java:10)"#;

        let parser = JavaStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::Java);
        assert_eq!(trace.error_type, "java.lang.NullPointerException");
        assert!(trace.error_message.contains("Cannot invoke"));
        assert_eq!(trace.frames.len(), 2);
        assert_eq!(trace.frames[0].line, Some(42));
    }

    #[test]
    fn test_java_caused_by_chain() {
        let input = r#"Exception in thread "main" java.lang.RuntimeException: Wrapper
	at com.example.App.main(App.java:10)
Caused by: java.lang.IllegalArgumentException: Invalid input
	at com.example.App.validate(App.java:25)"#;

        let parser = JavaStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::Java);
        // Should capture the root cause
        assert!(trace.error_type.contains("IllegalArgumentException"));
    }

    // C/C++ Parser Tests
    #[test]
    fn test_cpp_gdb_backtrace() {
        let input = r#"#0  0x00007f1234567890 in main (argc=1, argv=0x7fff12345678) at main.cpp:42
#1  0x00007f1234567891 in __libc_start_main () from /lib/libc.so.6"#;

        let parser = CppStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::Cpp);
        assert!(trace.frames.len() >= 1);
        assert_eq!(trace.frames[0].line, Some(42));
    }

    #[test]
    fn test_cpp_asan_output() {
        let input = r#"AddressSanitizer: heap-buffer-overflow on address 0x602000000014
#0 0x555555557a8b in main /home/user/app/main.cpp:10
#1 0x7ffff7a2d830 in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x21830)"#;

        let parser = CppStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::Cpp);
        assert_eq!(trace.error_type, "AddressSanitizer");
        assert!(trace.error_message.contains("heap-buffer-overflow"));
    }

    #[test]
    fn test_cpp_segfault() {
        let input = "Segmentation fault (core dumped)";

        let parser = CppStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert_eq!(trace.language, Language::Cpp);
        assert!(trace.error_message.contains("Segmentation fault"));
    }

    // ========================================================================
    // Phase 3.3: Frame Classification Tests
    // ========================================================================

    #[test]
    fn test_is_user_code_python_site_packages() {
        let input = r#"Traceback (most recent call last):
  File "/usr/lib/python3.9/site-packages/flask/app.py", line 100, in handle
    do_stuff()
ValueError: test"#;

        let parser = PythonStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert!(!trace.frames[0].is_user_code);
    }

    #[test]
    fn test_is_user_code_python_stdlib() {
        let input = r#"Traceback (most recent call last):
  File "/usr/lib/python3.9/json/decoder.py", line 25, in decode
    obj = json.loads(s)
JSONDecodeError: test"#;

        let parser = PythonStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert!(!trace.frames[0].is_user_code);
    }

    #[test]
    fn test_is_user_code_node_modules() {
        let input = r#"Error: test
    at Layer.handle (/app/node_modules/express/lib/router/layer.js:95:5)"#;

        let parser = JavaScriptStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert!(!trace.frames[0].is_user_code);
    }

    #[test]
    fn test_is_user_code_rust_std() {
        let input = r#"thread 'main' panicked at 'test', src/main.rs:10:5
stack backtrace:
   0: std::panicking::begin_panic
   1: my_crate::main"#;

        let parser = RustStackTraceParser;
        let trace = parser.parse(input).unwrap();

        // std:: function should be non-user code
        let std_frame = trace.frames.iter().find(|f| {
            f.function
                .as_ref()
                .map(|s| s.starts_with("std::"))
                .unwrap_or(false)
        });
        assert!(std_frame.map(|f| !f.is_user_code).unwrap_or(true));

        // my_crate:: function should be user code
        let user_frame = trace.frames.iter().find(|f| {
            f.function
                .as_ref()
                .map(|s| s.starts_with("my_crate::"))
                .unwrap_or(false)
        });
        assert!(user_frame.map(|f| f.is_user_code).unwrap_or(true));
    }

    #[test]
    fn test_is_user_code_java_jdk() {
        let input = r#"Exception in thread "main" java.lang.RuntimeException: test
	at java.util.HashMap.put(HashMap.java:100)
	at com.myapp.Main.process(Main.java:25)"#;

        let parser = JavaStackTraceParser;
        let trace = parser.parse(input).unwrap();

        // java.util is JDK code
        assert!(!trace.frames[0].is_user_code);
        // com.myapp is user code
        assert!(trace.frames[1].is_user_code);
    }

    #[test]
    fn test_is_user_code_local_file() {
        let input = r#"Traceback (most recent call last):
  File "./src/main.py", line 10, in main
    process()
ValueError: test"#;

        let parser = PythonStackTraceParser;
        let trace = parser.parse(input).unwrap();

        assert!(trace.frames[0].is_user_code);
    }

    #[test]
    fn test_relevance_scoring_via_root_cause() {
        // The root_cause_frame method prefers user code frames
        let mut trace = StackTrace::new(Language::Python, "test");
        trace.add_frame(
            StackFrame::new()
                .with_function("framework_internal")
                .with_is_user_code(false),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("user_function")
                .with_is_user_code(true),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("stdlib_helper")
                .with_is_user_code(false),
        );

        let root = trace.root_cause_frame().unwrap();
        assert_eq!(root.function, Some("user_function".to_string()));
        assert!(root.is_user_code);
    }

    #[test]
    fn test_user_frames_filtering() {
        let mut trace = StackTrace::new(Language::Python, "test");
        trace.add_frame(
            StackFrame::new()
                .with_function("flask_handler")
                .with_is_user_code(false),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("my_view")
                .with_is_user_code(true),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("my_model")
                .with_is_user_code(true),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("sqlalchemy_query")
                .with_is_user_code(false),
        );

        let user_frames = trace.user_frames();
        assert_eq!(user_frames.len(), 2);
        assert!(user_frames.iter().all(|f| f.is_user_code));
    }

    // ========================================================================
    // Phase 3.4: Source Context Injection Tests
    // ========================================================================

    #[test]
    fn test_context_flag_parsing() {
        let cli = Cli::parse_from(["why", "--context", "error"]);
        assert!(cli.context);
    }

    #[test]
    fn test_context_short_flag_parsing() {
        let cli = Cli::parse_from(["why", "-c", "error"]);
        assert!(cli.context);
    }

    #[test]
    fn test_context_lines_parsing() {
        let cli = Cli::parse_from(["why", "--context", "--context-lines", "10", "error"]);
        assert!(cli.context);
        assert_eq!(cli.context_lines, 10);
    }

    #[test]
    fn test_context_root_parsing() {
        let cli = Cli::parse_from(["why", "--context", "--context-root", "/app", "error"]);
        assert!(cli.context);
        assert_eq!(cli.context_root, Some(PathBuf::from("/app")));
    }

    #[test]
    fn test_source_context_config_default() {
        let config = SourceContextConfig::default();
        assert_eq!(config.context_lines, 5);
        assert!(config.context_root.is_none());
        assert_eq!(config.max_context_chars, 4096);
    }

    #[test]
    fn test_resolve_source_path_relative() {
        // Create a temp file for testing
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_resolve_source.rs");
        std::fs::write(&test_file, "fn main() {}").unwrap();

        let path = PathBuf::from(test_file.file_name().unwrap());
        let context_root = Some(temp_dir.clone());

        let resolved = resolve_source_path(&path, &context_root);
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap(), test_file);

        // Cleanup
        std::fs::remove_file(&test_file).ok();
    }

    #[test]
    fn test_resolve_source_path_absolute() {
        // Create a temp file for testing
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_resolve_absolute.rs");
        std::fs::write(&test_file, "fn main() {}").unwrap();

        let resolved = resolve_source_path(&test_file, &None);
        assert!(resolved.is_some());
        assert_eq!(resolved.unwrap(), test_file);

        // Cleanup
        std::fs::remove_file(&test_file).ok();
    }

    #[test]
    fn test_resolve_source_path_nonexistent() {
        let path = PathBuf::from("/nonexistent/path/to/file.rs");
        let resolved = resolve_source_path(&path, &None);
        assert!(resolved.is_none());
    }

    #[test]
    fn test_extract_frame_context_basic() {
        // Create a temp file with numbered lines
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_context_basic.py");
        std::fs::write(
            &test_file,
            "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10\n",
        )
        .unwrap();

        let frame = StackFrame::new().with_file(test_file.clone()).with_line(5);

        let config = SourceContextConfig {
            context_lines: 2,
            context_root: None,
            max_context_chars: 4096,
        };

        let context = extract_frame_context(&frame, &config);
        assert!(context.is_some());
        let ctx = context.unwrap();

        // Should contain lines 3-7
        assert!(ctx.contains("line 3"));
        assert!(ctx.contains("line 5"));
        assert!(ctx.contains("line 7"));
        // Line 5 should be marked
        assert!(ctx.contains(">    5 |"));

        // Cleanup
        std::fs::remove_file(&test_file).ok();
    }

    #[test]
    fn test_extract_frame_context_file_start() {
        // Create a temp file
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_context_start.py");
        std::fs::write(&test_file, "line 1\nline 2\nline 3\nline 4\nline 5\n").unwrap();

        let frame = StackFrame::new().with_file(test_file.clone()).with_line(1);

        let config = SourceContextConfig {
            context_lines: 3,
            context_root: None,
            max_context_chars: 4096,
        };

        let context = extract_frame_context(&frame, &config);
        assert!(context.is_some());
        let ctx = context.unwrap();

        // Should start at line 1 (not go negative)
        assert!(ctx.contains(">    1 |"));
        assert!(ctx.contains("line 1"));

        // Cleanup
        std::fs::remove_file(&test_file).ok();
    }

    #[test]
    fn test_extract_frame_context_file_end() {
        // Create a temp file
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_context_end.py");
        std::fs::write(&test_file, "line 1\nline 2\nline 3\n").unwrap();

        let frame = StackFrame::new().with_file(test_file.clone()).with_line(3);

        let config = SourceContextConfig {
            context_lines: 5,
            context_root: None,
            max_context_chars: 4096,
        };

        let context = extract_frame_context(&frame, &config);
        assert!(context.is_some());
        let ctx = context.unwrap();

        // Should handle EOF gracefully
        assert!(ctx.contains("line 3"));

        // Cleanup
        std::fs::remove_file(&test_file).ok();
    }

    #[test]
    fn test_extract_frame_context_missing_file() {
        let frame = StackFrame::new()
            .with_file("/nonexistent/file.py")
            .with_line(10);

        let config = SourceContextConfig::default();
        let context = extract_frame_context(&frame, &config);
        assert!(context.is_none());
    }

    #[test]
    fn test_extract_frame_context_no_line() {
        let frame = StackFrame::new().with_file("/some/file.py");
        // No line number

        let config = SourceContextConfig::default();
        let context = extract_frame_context(&frame, &config);
        assert!(context.is_none());
    }

    #[test]
    fn test_extract_stack_trace_context_user_only() {
        // Create temp files
        let temp_dir = std::env::temp_dir();
        let user_file = temp_dir.join("user_code.py");
        let framework_file = temp_dir.join("framework.py");
        std::fs::write(&user_file, "user line 1\nuser line 2\nuser line 3\n").unwrap();
        std::fs::write(&framework_file, "framework line 1\nframework line 2\n").unwrap();

        let mut trace = StackTrace::new(Language::Python, "test");
        trace.add_frame(
            StackFrame::new()
                .with_function("framework_func")
                .with_file(framework_file.clone())
                .with_line(1)
                .with_is_user_code(false),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("user_func")
                .with_file(user_file.clone())
                .with_line(2)
                .with_is_user_code(true),
        );

        let config = SourceContextConfig {
            context_lines: 1,
            context_root: None,
            max_context_chars: 4096,
        };

        let context = extract_stack_trace_context(&trace, &config);

        // Should only contain user code context
        assert!(context.contains("user_func"));
        assert!(context.contains("user line 2"));
        // Should NOT contain framework code
        assert!(!context.contains("framework_func"));

        // Cleanup
        std::fs::remove_file(&user_file).ok();
        std::fs::remove_file(&framework_file).ok();
    }

    // ========================================================================
    // Phase 3.5: Enhanced Prompt Construction Tests
    // ========================================================================

    #[test]
    fn test_format_frame_for_prompt_full() {
        let frame = StackFrame::new()
            .with_function("process_data")
            .with_file("/app/src/main.py")
            .with_line(42)
            .with_column(10);

        let formatted = format_frame_for_prompt(&frame);
        assert!(formatted.contains("process_data"));
        assert!(formatted.contains("/app/src/main.py:42:10"));
    }

    #[test]
    fn test_format_frame_for_prompt_no_column() {
        let frame = StackFrame::new()
            .with_function("main")
            .with_file("main.rs")
            .with_line(10);

        let formatted = format_frame_for_prompt(&frame);
        assert!(formatted.contains("main"));
        assert!(formatted.contains("main.rs:10"));
        assert!(!formatted.contains(":10:"));
    }

    #[test]
    fn test_format_frame_for_prompt_no_line() {
        let frame = StackFrame::new()
            .with_function("unknown_func")
            .with_file("module.py");

        let formatted = format_frame_for_prompt(&frame);
        assert!(formatted.contains("unknown_func"));
        assert!(formatted.contains("at module.py"));
    }

    #[test]
    fn test_format_frame_for_prompt_empty() {
        let frame = StackFrame::new();
        let formatted = format_frame_for_prompt(&frame);
        assert_eq!(formatted, "<unknown frame>");
    }

    #[test]
    fn test_select_frames_user_first() {
        let mut trace = StackTrace::new(Language::Python, "test");
        trace.add_frame(
            StackFrame::new()
                .with_function("stdlib")
                .with_is_user_code(false),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("user1")
                .with_is_user_code(true),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("user2")
                .with_is_user_code(true),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("framework")
                .with_is_user_code(false),
        );

        let selected = select_frames_for_prompt(&trace, 3);

        // User frames should be first
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].function, Some("user1".to_string()));
        assert_eq!(selected[1].function, Some("user2".to_string()));
        // Then framework/stdlib to fill remaining
        assert!(!selected[2].is_user_code);
    }

    #[test]
    fn test_select_frames_respects_limit() {
        let mut trace = StackTrace::new(Language::Python, "test");
        for i in 0..10 {
            trace.add_frame(
                StackFrame::new()
                    .with_function(format!("func{}", i))
                    .with_is_user_code(true),
            );
        }

        let selected = select_frames_for_prompt(&trace, 3);
        assert_eq!(selected.len(), 3);
    }

    #[test]
    fn test_python_keyerror_hint() {
        let trace = StackTrace::new(Language::Python, "test").with_error_type("KeyError");

        let hints = get_error_hints(&trace);
        assert!(hints.is_some());
        assert!(hints.unwrap().contains("dict"));
    }

    #[test]
    fn test_rust_ownership_hint() {
        let trace = StackTrace::new(Language::Rust, "test")
            .with_error_type("error[E0382]")
            .with_error_message("borrow of moved value");

        let hints = get_error_hints(&trace);
        assert!(hints.is_some());
        assert!(hints.unwrap().contains("Ownership"));
    }

    #[test]
    fn test_rust_panic_unwrap_hint() {
        let trace = StackTrace::new(Language::Rust, "test")
            .with_error_type("panic")
            .with_error_message("called unwrap() on a None value");

        let hints = get_error_hints(&trace);
        assert!(hints.is_some());
        assert!(hints.unwrap().contains("unwrap"));
    }

    #[test]
    fn test_js_undefined_hint() {
        let trace = StackTrace::new(Language::JavaScript, "test")
            .with_error_type("TypeError")
            .with_error_message("Cannot read property 'x' of undefined");

        let hints = get_error_hints(&trace);
        assert!(hints.is_some());
        assert!(hints.unwrap().contains("optional chaining"));
    }

    #[test]
    fn test_java_null_hint() {
        let trace = StackTrace::new(Language::Java, "test")
            .with_error_type("java.lang.NullPointerException");

        let hints = get_error_hints(&trace);
        assert!(hints.is_some());
        assert!(hints.unwrap().contains("null"));
    }

    #[test]
    fn test_cpp_segfault_hint() {
        let trace = StackTrace::new(Language::Cpp, "test")
            .with_error_type("signal")
            .with_error_message("Segmentation fault");

        let hints = get_error_hints(&trace);
        assert!(hints.is_some());
        assert!(hints.unwrap().contains("Segfault"));
    }

    #[test]
    fn test_build_prompt_basic() {
        let mut trace = StackTrace::new(Language::Python, "test")
            .with_error_type("KeyError")
            .with_error_message("'missing_key'");
        trace.add_frame(
            StackFrame::new()
                .with_function("process")
                .with_file("main.py")
                .with_line(10)
                .with_is_user_code(true),
        );

        let prompt = build_stack_trace_prompt("test input", &trace, None);

        assert!(prompt.contains("ERROR TYPE: KeyError"));
        assert!(prompt.contains("ERROR MESSAGE: 'missing_key'"));
        assert!(prompt.contains("STACK TRACE"));
        assert!(prompt.contains("process"));
        assert!(prompt.contains("HINT:"));
    }

    #[test]
    fn test_build_prompt_with_source_context() {
        let trace = StackTrace::new(Language::Python, "test").with_error_type("ValueError");

        let source_ctx = "  10 | x = int(value)\n> 11 | result = x / 0\n  12 | return result";
        let prompt = build_stack_trace_prompt("test", &trace, Some(source_ctx));

        assert!(prompt.contains("SOURCE CONTEXT:"));
        assert!(prompt.contains("x = int(value)"));
        assert!(prompt.contains("result = x / 0"));
    }

    #[test]
    fn test_build_prompt_marks_user_frames() {
        let mut trace = StackTrace::new(Language::Python, "test").with_error_type("Error");
        trace.add_frame(
            StackFrame::new()
                .with_function("user_func")
                .with_is_user_code(true),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("framework_func")
                .with_is_user_code(false),
        );

        let prompt = build_stack_trace_prompt("", &trace, None);

        // User frames should be marked with >
        assert!(prompt.contains(">[0] user_func"));
        // Framework frames should not have >
        assert!(prompt.contains(" [1] framework_func"));
    }

    #[test]
    fn test_no_hint_for_unknown_language() {
        let trace = StackTrace::new(Language::Unknown, "test").with_error_type("SomeError");

        let hints = get_error_hints(&trace);
        assert!(hints.is_none());
    }

    #[test]
    fn test_go_panic_nil_hint() {
        let trace = StackTrace::new(Language::Go, "test")
            .with_error_type("panic")
            .with_error_message("runtime error: invalid memory address or nil pointer dereference");

        let hints = get_error_hints(&trace);
        assert!(hints.is_some());
        assert!(hints.unwrap().contains("nil"));
    }

    // ========================================================================
    // Phase 3.6: Structured Output Tests
    // ========================================================================

    #[test]
    fn test_show_frames_flag_parsing() {
        let cli = Cli::parse_from(["why", "--show-frames", "error"]);
        assert!(cli.show_frames);
    }

    #[test]
    fn test_stack_frame_json_from_frame() {
        let frame = StackFrame::new()
            .with_function("test_func")
            .with_file("/app/main.py")
            .with_line(42)
            .with_column(10)
            .with_is_user_code(true);

        let json_frame = StackFrameJson::from(&frame);
        assert_eq!(json_frame.function, Some("test_func".to_string()));
        assert_eq!(json_frame.file, Some("/app/main.py".to_string()));
        assert_eq!(json_frame.line, Some(42));
        assert_eq!(json_frame.column, Some(10));
        assert!(json_frame.is_user_code);
    }

    #[test]
    fn test_stack_frame_json_serializes() {
        let frame = StackFrame::new()
            .with_function("main")
            .with_file("main.rs")
            .with_line(10);

        let json_frame = StackFrameJson::from(&frame);
        let serialized = serde_json::to_string(&json_frame).unwrap();

        assert!(serialized.contains("\"function\":\"main\""));
        assert!(serialized.contains("\"file\":\"main.rs\""));
        assert!(serialized.contains("\"line\":10"));
        // column should be omitted since it's None
        assert!(!serialized.contains("column"));
    }

    #[test]
    fn test_stack_trace_json_from_trace() {
        let mut trace = StackTrace::new(Language::Python, "raw")
            .with_error_type("KeyError")
            .with_error_message("'x'");
        trace.add_frame(
            StackFrame::new()
                .with_function("main")
                .with_file("main.py")
                .with_line(10)
                .with_is_user_code(true),
        );

        let json_trace = StackTraceJson::from(&trace);
        assert_eq!(json_trace.language, Language::Python);
        assert_eq!(json_trace.error_type, "KeyError");
        assert_eq!(json_trace.error_message, "'x'");
        assert_eq!(json_trace.frames.len(), 1);
        assert!(json_trace.root_cause_frame.is_some());
    }

    #[test]
    fn test_stack_trace_json_schema() {
        let mut trace = StackTrace::new(Language::Python, "test")
            .with_error_type("KeyError")
            .with_error_message("missing");
        trace.add_frame(
            StackFrame::new()
                .with_function("process")
                .with_file("/app/main.py")
                .with_line(42)
                .with_is_user_code(true),
        );

        let json_trace = StackTraceJson::from(&trace);
        let serialized = serde_json::to_string_pretty(&json_trace).unwrap();

        // Verify JSON structure
        assert!(serialized.contains("\"language\": \"python\""));
        assert!(serialized.contains("\"error_type\": \"KeyError\""));
        assert!(serialized.contains("\"error_message\": \"missing\""));
        assert!(serialized.contains("\"frames\": ["));
        assert!(serialized.contains("\"root_cause_frame\": {"));
    }

    #[test]
    fn test_json_frames_array_serialization() {
        let mut trace = StackTrace::new(Language::JavaScript, "test");
        trace.add_frame(
            StackFrame::new()
                .with_function("outer")
                .with_file("app.js")
                .with_line(100)
                .with_is_user_code(true),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("inner")
                .with_file("lib.js")
                .with_line(50)
                .with_is_user_code(false),
        );

        let json_trace = StackTraceJson::from(&trace);
        assert_eq!(json_trace.frames.len(), 2);
        assert_eq!(json_trace.frames[0].function, Some("outer".to_string()));
        assert_eq!(json_trace.frames[1].function, Some("inner".to_string()));
    }

    #[test]
    fn test_json_root_cause_frame_identification() {
        let mut trace = StackTrace::new(Language::Python, "test");
        // First add non-user frames
        trace.add_frame(
            StackFrame::new()
                .with_function("framework_internal")
                .with_is_user_code(false),
        );
        // Then add user frame (this should be root cause)
        trace.add_frame(
            StackFrame::new()
                .with_function("user_handler")
                .with_file("/app/handler.py")
                .with_line(25)
                .with_is_user_code(true),
        );

        let json_trace = StackTraceJson::from(&trace);
        let root = json_trace.root_cause_frame.unwrap();
        assert_eq!(root.function, Some("user_handler".to_string()));
        assert!(root.is_user_code);
    }

    #[test]
    fn test_format_file_line_full() {
        let path = PathBuf::from("/app/src/main.rs");
        let result = format_file_line(&path, Some(42), Some(10));
        // Check that the base parts are there (without ANSI codes for testing)
        assert!(result.contains("/app/src/main.rs"));
        assert!(result.contains("42"));
        assert!(result.contains("10"));
    }

    #[test]
    fn test_format_file_line_no_column() {
        let path = PathBuf::from("main.py");
        let result = format_file_line(&path, Some(100), None);
        assert!(result.contains("main.py"));
        assert!(result.contains("100"));
    }

    #[test]
    fn test_format_file_line_no_line() {
        let path = PathBuf::from("module.js");
        let result = format_file_line(&path, None, None);
        assert!(result.contains("module.js"));
    }

    #[test]
    fn test_print_frames_does_not_panic_empty() {
        let trace = StackTrace::new(Language::Python, "test");
        // This should not panic, just print "No frames parsed"
        print_frames(&trace);
    }

    #[test]
    fn test_print_frames_does_not_panic_with_frames() {
        let mut trace = StackTrace::new(Language::Python, "test")
            .with_error_type("KeyError")
            .with_error_message("'x'");
        trace.add_frame(
            StackFrame::new()
                .with_function("main")
                .with_file("/app/main.py")
                .with_line(10)
                .with_is_user_code(true),
        );
        trace.add_frame(
            StackFrame::new()
                .with_function("flask.dispatch")
                .with_file("/site-packages/flask/app.py")
                .with_line(100)
                .with_is_user_code(false),
        );

        // This should not panic
        print_frames(&trace);
    }

    // ========================================================================
    // Phase 4.3: Output Capture Integration Tests
    // ========================================================================

    #[test]
    fn test_capture_flag_parsing() {
        let cli = Cli::parse_from(["why", "--capture", "echo", "hello"]);
        assert!(cli.capture);
        assert_eq!(cli.error, vec!["echo".to_string(), "hello".to_string()]);
    }

    #[test]
    fn test_capture_all_flag_parsing() {
        let cli = Cli::parse_from(["why", "--capture", "--capture-all", "echo", "hello"]);
        assert!(cli.capture);
        assert!(cli.capture_all);
    }

    #[test]
    fn test_capture_result_struct() {
        let result = CaptureResult {
            command: "test cmd".to_string(),
            exit_code: 1,
            stdout: "stdout output".to_string(),
            stderr: "stderr output".to_string(),
        };

        assert_eq!(result.command, "test cmd");
        assert_eq!(result.exit_code, 1);
        assert_eq!(result.stdout, "stdout output");
        assert_eq!(result.stderr, "stderr output");
    }

    #[test]
    fn test_capture_command_empty_error() {
        let result = run_capture_command(&[], false);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No command specified"));
    }

    #[test]
    fn test_capture_command_success() {
        // Run a simple command that succeeds
        let result = run_capture_command(&["echo".to_string(), "hello".to_string()], false);
        assert!(result.is_ok());
        let capture = result.unwrap();
        assert_eq!(capture.exit_code, 0);
        assert_eq!(capture.command, "echo hello");
    }

    #[test]
    fn test_capture_command_failure() {
        // Run a command that fails
        let result = run_capture_command(&["false".to_string()], false);
        assert!(result.is_ok());
        let capture = result.unwrap();
        assert_eq!(capture.exit_code, 1);
    }

    #[test]
    fn test_capture_command_with_stderr() {
        // Run a command that writes to stderr
        let result = run_capture_command(
            &[
                "sh".to_string(),
                "-c".to_string(),
                "echo error >&2".to_string(),
            ],
            false,
        );
        assert!(result.is_ok());
        let capture = result.unwrap();
        assert!(capture.stderr.contains("error"));
    }

    #[test]
    fn test_capture_all_includes_stdout() {
        // With capture_all, stdout should be captured too
        let result = run_capture_command(
            &["echo".to_string(), "test_output".to_string()],
            true, // capture_all = true
        );
        assert!(result.is_ok());
        let capture = result.unwrap();
        assert!(capture.stdout.contains("test_output"));
    }

    #[test]
    fn test_capture_without_capture_all_empty_stdout() {
        // Without capture_all, stdout should not be captured
        let result = run_capture_command(
            &["echo".to_string(), "test_output".to_string()],
            false, // capture_all = false
        );
        assert!(result.is_ok());
        let capture = result.unwrap();
        // stdout is not captured when capture_all is false
        assert!(capture.stdout.is_empty());
    }

    #[test]
    fn test_capture_preserves_exit_code() {
        // Test various exit codes
        let result = run_capture_command(
            &["sh".to_string(), "-c".to_string(), "exit 42".to_string()],
            false,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().exit_code, 42);

        let result = run_capture_command(
            &["sh".to_string(), "-c".to_string(), "exit 0".to_string()],
            false,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().exit_code, 0);
    }

    // ========================================================================
    // Phase 4.4: Interactive Confirmation Mode Tests
    // ========================================================================

    #[test]
    fn test_confirm_flag_parsing() {
        let cli = Cli::parse_from(["why", "--confirm", "error"]);
        assert!(cli.confirm);
    }

    #[test]
    fn test_auto_flag_parsing() {
        let cli = Cli::parse_from(["why", "--auto", "error"]);
        assert!(cli.auto);
    }

    #[test]
    fn test_confirm_and_auto_flags_together() {
        // Both can be specified, --auto overrides --confirm
        let cli = Cli::parse_from(["why", "--confirm", "--auto", "error"]);
        assert!(cli.confirm);
        assert!(cli.auto);
    }

    #[test]
    fn test_contains_error_patterns_detects_error() {
        assert!(contains_error_patterns("error: something went wrong"));
        assert!(contains_error_patterns("Error: missing file"));
        assert!(contains_error_patterns("ERROR: failed"));
    }

    #[test]
    fn test_contains_error_patterns_detects_exception() {
        assert!(contains_error_patterns(
            "Traceback (most recent call last):\n  exception here"
        ));
        assert!(contains_error_patterns("Exception in thread \"main\""));
        assert!(contains_error_patterns("EXCEPTION: null pointer"));
    }

    #[test]
    fn test_contains_error_patterns_detects_panic() {
        assert!(contains_error_patterns("panic: something bad happened"));
        assert!(contains_error_patterns("thread 'main' panicked at"));
        assert!(contains_error_patterns("PANIC: assertion failed"));
    }

    #[test]
    fn test_contains_error_patterns_detects_failed() {
        assert!(contains_error_patterns("Build failed"));
        assert!(contains_error_patterns("Test FAILED"));
        assert!(contains_error_patterns("Command failed with exit code 1"));
    }

    #[test]
    fn test_contains_error_patterns_detects_signals() {
        assert!(contains_error_patterns("Segmentation fault"));
        assert!(contains_error_patterns("SIGSEGV"));
        assert!(contains_error_patterns("SIGABRT"));
    }

    #[test]
    fn test_contains_error_patterns_negative() {
        // These should NOT match
        assert!(!contains_error_patterns("Everything is fine"));
        assert!(!contains_error_patterns("Build succeeded"));
        assert!(!contains_error_patterns("Hello world"));
        assert!(!contains_error_patterns(""));
    }

    #[test]
    fn test_smart_default_with_error_pattern() {
        // When error patterns are present, should default to yes
        let stderr_with_error = "error: compilation failed\nmissing semicolon";
        assert!(contains_error_patterns(stderr_with_error));

        // When no error patterns, should default to no
        let stderr_no_error = "some output\nwarning: unused variable";
        assert!(!contains_error_patterns(stderr_no_error));
    }

    // ========================================================================
    // Phase 4.5: Configuration and Customization Tests
    // ========================================================================

    #[test]
    fn test_hook_config_flag_parsing() {
        let cli = Cli::parse_from(["why", "--hook-config"]);
        assert!(cli.hook_config);
    }

    #[test]
    fn test_config_default_values() {
        let config = Config::default();
        assert!(!config.hook.auto_explain);
        assert!(config.hook.skip_exit_codes.contains(&0));
        assert!(config.hook.skip_exit_codes.contains(&130));
        assert_eq!(config.hook.min_stderr_lines, 1);
    }

    #[test]
    fn test_config_missing_file_uses_defaults() {
        // Non-existent path should return defaults
        let config = Config::load_from_path(Some(PathBuf::from("/nonexistent/config.toml")));
        assert!(!config.hook.auto_explain);
        assert_eq!(config.hook.min_stderr_lines, 1);
    }

    #[test]
    fn test_config_none_path_uses_defaults() {
        let config = Config::load_from_path(None);
        assert!(!config.hook.auto_explain);
    }

    #[test]
    fn test_config_skip_exit_codes() {
        let config = Config::default();
        assert!(config.should_skip_exit_code(0)); // Success
        assert!(config.should_skip_exit_code(130)); // Ctrl+C
        assert!(!config.should_skip_exit_code(1)); // General error
        assert!(!config.should_skip_exit_code(127)); // Command not found
    }

    #[test]
    fn test_config_ignore_patterns_default() {
        let config = Config::default();
        assert!(config.should_ignore_command("cd /tmp"));
        assert!(config.should_ignore_command("ls -la"));
        assert!(config.should_ignore_command("echo hello"));
        assert!(config.should_ignore_command("pwd"));
        assert!(config.should_ignore_command("clear"));
    }

    #[test]
    fn test_config_ignore_patterns_negative() {
        let config = Config::default();
        assert!(!config.should_ignore_command("npm run build"));
        assert!(!config.should_ignore_command("cargo test"));
        assert!(!config.should_ignore_command("python script.py"));
    }

    #[test]
    fn test_config_toml_parsing() {
        // Create a temp config file
        let temp_dir = std::env::temp_dir();
        let config_file = temp_dir.join("test_why_config.toml");
        std::fs::write(
            &config_file,
            r#"
[hook]
auto_explain = true
skip_exit_codes = [0, 130, 255]
min_stderr_lines = 5

[hook.ignore_commands]
patterns = ["^git status$", "^make$"]
        "#,
        )
        .unwrap();

        let config = Config::load_from_path(Some(config_file.clone()));
        assert!(config.hook.auto_explain);
        assert!(config.hook.skip_exit_codes.contains(&255));
        assert_eq!(config.hook.min_stderr_lines, 5);
        assert!(config.should_ignore_command("git status"));
        assert!(config.should_ignore_command("make"));

        // Cleanup
        std::fs::remove_file(&config_file).ok();
    }

    #[test]
    fn test_config_env_why_hook_auto() {
        let mut config = Config::default();
        assert!(!config.hook.auto_explain);

        // Simulate WHY_HOOK_AUTO=1
        std::env::set_var("WHY_HOOK_AUTO", "1");
        config.apply_env_overrides();
        assert!(config.hook.auto_explain);

        // Cleanup
        std::env::remove_var("WHY_HOOK_AUTO");
    }

    #[test]
    fn test_config_env_why_hook_disable() {
        std::env::set_var("WHY_HOOK_DISABLE", "1");
        assert!(Config::is_hook_disabled());

        std::env::set_var("WHY_HOOK_DISABLE", "0");
        assert!(!Config::is_hook_disabled());

        std::env::remove_var("WHY_HOOK_DISABLE");
        assert!(!Config::is_hook_disabled());
    }

    #[test]
    fn test_generate_default_config_is_valid_toml() {
        let config_str = generate_default_config();
        let parsed: Result<Config, _> = toml::from_str(&config_str);
        assert!(parsed.is_ok(), "Generated config should be valid TOML");
    }

    #[test]
    fn test_config_path_returns_some() {
        // Config path should return Some on most systems
        let path = Config::config_path();
        // The dirs crate returns None on some systems (like CI)
        // so we just verify it either returns a path or None
        if let Some(p) = path {
            assert!(p.ends_with("config.toml"));
        }
    }

    // ========================================================================
    // Phase 4.6: Installation Helpers Tests
    // ========================================================================

    #[test]
    fn test_hook_install_flag_parsing() {
        let cli = Cli::parse_from(["why", "--hook-install", "bash"]);
        assert_eq!(cli.hook_install, Some(Shell::Bash));
    }

    #[test]
    fn test_hook_uninstall_flag_parsing() {
        let cli = Cli::parse_from(["why", "--hook-uninstall", "zsh"]);
        assert_eq!(cli.hook_uninstall, Some(Shell::Zsh));
    }

    #[test]
    fn test_get_shell_config_path_bash() {
        let path = get_shell_config_path(Shell::Bash);
        if let Some(p) = path {
            assert!(p.to_string_lossy().ends_with(".bashrc"));
        }
    }

    #[test]
    fn test_get_shell_config_path_zsh() {
        let path = get_shell_config_path(Shell::Zsh);
        if let Some(p) = path {
            assert!(p.to_string_lossy().ends_with(".zshrc"));
        }
    }

    #[test]
    fn test_get_shell_config_path_fish() {
        let path = get_shell_config_path(Shell::Fish);
        if let Some(p) = path {
            assert!(p.to_string_lossy().ends_with("why.fish"));
        }
    }

    #[test]
    fn test_generate_hook_with_markers_contains_markers() {
        let hook = generate_hook_with_markers(Shell::Bash);
        assert!(hook.contains(HOOK_MARKER_START));
        assert!(hook.contains(HOOK_MARKER_END));
    }

    #[test]
    fn test_generate_hook_bash_content() {
        let hook = generate_hook_with_markers(Shell::Bash);
        assert!(hook.contains("__why_prompt_command"));
        assert!(hook.contains("PROMPT_COMMAND"));
    }

    #[test]
    fn test_generate_hook_zsh_content() {
        let hook = generate_hook_with_markers(Shell::Zsh);
        assert!(hook.contains("__why_precmd"));
        assert!(hook.contains("add-zsh-hook"));
    }

    #[test]
    fn test_generate_hook_fish_content() {
        let hook = generate_hook_with_markers(Shell::Fish);
        assert!(hook.contains("__why_postexec"));
        assert!(hook.contains("fish_postexec"));
    }

    #[test]
    fn test_hooks_already_installed_detection() {
        // Create temp file with hooks
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_hook_detection.sh");

        // Write content with hook markers
        let content = format!(
            "# some config\n{}\nsome hook code\n{}\n# more config\n",
            HOOK_MARKER_START, HOOK_MARKER_END
        );
        std::fs::write(&temp_file, content).unwrap();

        assert!(hooks_already_installed(&temp_file));

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    #[test]
    fn test_hooks_not_installed_detection() {
        // Create temp file without hooks
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_hook_not_installed.sh");

        std::fs::write(&temp_file, "# some config\necho hello\n").unwrap();

        assert!(!hooks_already_installed(&temp_file));

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    #[test]
    fn test_hooks_installed_nonexistent_file() {
        let nonexistent = PathBuf::from("/nonexistent/file.sh");
        assert!(!hooks_already_installed(&nonexistent));
    }

    #[test]
    fn test_hook_install_and_uninstall_roundtrip() {
        // Create a temp directory to simulate a config file
        let temp_dir = std::env::temp_dir();
        let temp_config = temp_dir.join("test_roundtrip_bashrc");

        // Start with some existing content
        let original_content = "# Existing bash config\nexport PATH=$PATH:/usr/local/bin\n";
        std::fs::write(&temp_config, original_content).unwrap();

        // Manually simulate install by appending hook code
        let mut content = std::fs::read_to_string(&temp_config).unwrap();
        content.push('\n');
        content.push_str(&generate_hook_with_markers(Shell::Bash));
        std::fs::write(&temp_config, content).unwrap();

        // Verify hooks are installed
        assert!(hooks_already_installed(&temp_config));

        // Simulate uninstall by removing the hook block
        let content = std::fs::read_to_string(&temp_config).unwrap();
        let mut new_content = String::new();
        let mut in_hook_block = false;

        for line in content.lines() {
            if line.trim() == HOOK_MARKER_START {
                in_hook_block = true;
                continue;
            }
            if line.trim() == HOOK_MARKER_END {
                in_hook_block = false;
                continue;
            }
            if !in_hook_block {
                new_content.push_str(line);
                new_content.push('\n');
            }
        }
        std::fs::write(&temp_config, &new_content).unwrap();

        // Verify hooks are uninstalled
        assert!(!hooks_already_installed(&temp_config));

        // Original content should be preserved
        let final_content = std::fs::read_to_string(&temp_config).unwrap();
        assert!(final_content.contains("export PATH"));

        // Cleanup
        std::fs::remove_file(&temp_config).ok();
    }

    #[test]
    fn test_hook_install_idempotent() {
        let temp_dir = std::env::temp_dir();
        let temp_config = temp_dir.join("test_idempotent_bashrc");

        // Write hook once
        let hook = generate_hook_with_markers(Shell::Bash);
        std::fs::write(&temp_config, &hook).unwrap();

        // Check it's installed
        assert!(hooks_already_installed(&temp_config));

        // Content should only have one set of markers
        let content = std::fs::read_to_string(&temp_config).unwrap();
        let start_count = content.matches(HOOK_MARKER_START).count();
        let end_count = content.matches(HOOK_MARKER_END).count();
        assert_eq!(start_count, 1);
        assert_eq!(end_count, 1);

        // Cleanup
        std::fs::remove_file(&temp_config).ok();
    }

    // ========================================================================
    // Watch Mode Tests (Feature 2)
    // ========================================================================

    // Phase 2.1: File Watch Infrastructure Tests

    #[test]
    fn test_file_watcher_path_validation_nonexistent() {
        let result = FileWatcher::new(PathBuf::from("/nonexistent/path/to/file.log"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn test_file_watcher_path_validation_directory() {
        let result = FileWatcher::new(PathBuf::from("/tmp"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Not a file"));
    }

    #[test]
    fn test_file_watcher_seek_to_end() {
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_file_watcher_seek.log");
        std::fs::write(&temp_file, "existing content\n").unwrap();

        let watcher = FileWatcher::new(temp_file.clone()).unwrap();

        // Position should be at end of file
        assert_eq!(watcher.position, 17); // "existing content\n" = 17 bytes

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    #[test]
    fn test_file_watcher_read_new_content() {
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_file_watcher_read.log");
        std::fs::write(&temp_file, "initial\n").unwrap();

        let mut watcher = FileWatcher::new(temp_file.clone()).unwrap();

        // Initially no new content
        let content = watcher.read_new_content().unwrap();
        assert!(content.is_none());

        // Append new content
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&temp_file)
            .unwrap();
        writeln!(file, "new line").unwrap();
        drop(file);

        // Now there should be new content
        let content = watcher.read_new_content().unwrap();
        assert!(content.is_some());
        assert_eq!(content.unwrap().trim(), "new line");

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    #[test]
    fn test_file_truncation_detection() {
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_file_truncation.log");
        std::fs::write(&temp_file, "lots of initial content here\n").unwrap();

        let mut watcher = FileWatcher::new(temp_file.clone()).unwrap();
        assert!(watcher.position > 0);

        // Truncate file (simulate log rotation)
        std::fs::write(&temp_file, "new\n").unwrap();

        // Read should reset position and return new content
        let content = watcher.read_new_content().unwrap();
        assert!(content.is_some());
        assert_eq!(content.unwrap().trim(), "new");

        // Position should now be at 4 (length of "new\n")
        assert_eq!(watcher.position, 4);

        // Cleanup
        std::fs::remove_file(&temp_file).ok();
    }

    // Phase 2.3: Error Detection Engine Tests

    #[test]
    fn test_error_detector_case_variants() {
        let detector = ErrorDetector::new(None, 50);

        assert!(detector.is_error_line("error: something went wrong"));
        assert!(detector.is_error_line("Error: something went wrong"));
        assert!(detector.is_error_line("ERROR: something went wrong"));
    }

    #[test]
    fn test_error_detector_keywords() {
        let detector = ErrorDetector::new(None, 50);

        assert!(detector.is_error_line("failed: build"));
        assert!(detector.is_error_line("exception: ValueError"));
        assert!(detector.is_error_line("panic: runtime error"));
        assert!(detector.is_error_line("Traceback (most recent call last):"));
    }

    #[test]
    fn test_error_detector_stack_trace_indicators() {
        let detector = ErrorDetector::new(None, 50);

        assert!(detector.is_error_line("    at Object.<anonymous> (/path/file.js:10:5)"));
        assert!(detector.is_error_line("  File \"/path/to/file.py\", line 10"));
        assert!(detector.is_error_line("    at /home/user/project/main.rs:42"));
    }

    #[test]
    fn test_error_detector_custom_pattern() {
        let pattern = Regex::new(r"ERR-\d+").unwrap();
        let detector = ErrorDetector::new(Some(pattern), 50);

        assert!(detector.is_error_line("ERR-12345: custom error"));
        assert!(!detector.is_error_line("error: standard error")); // Custom pattern only
    }

    #[test]
    fn test_multiline_aggregation() {
        let mut detector = ErrorDetector::new(None, 50);

        // First error line starts aggregation
        let result1 = detector.process_line("Error: something went wrong");
        assert!(result1.is_none()); // Not flushed yet

        // Continuation line
        let result2 = detector.process_line("    at main.rs:10");
        assert!(result2.is_none()); // Still aggregating

        // Blank lines end the error
        detector.process_line("");
        let result = detector.process_line("");
        assert!(result.is_some());

        let error = result.unwrap();
        assert!(error.content.contains("Error: something went wrong"));
        assert!(error.content.contains("at main.rs:10"));
    }

    #[test]
    fn test_aggregation_window_limit() {
        let mut detector = ErrorDetector::new(None, 5); // Max 5 lines

        detector.process_line("Error: start");
        detector.process_line("line 1");
        detector.process_line("line 2");
        detector.process_line("line 3");
        detector.process_line("line 4");

        // 6th line should trigger flush
        let result = detector.process_line("line 5");
        assert!(result.is_some());
    }

    #[test]
    fn test_error_dedup_hash() {
        let error1 = DetectedError::new("Error: something failed".to_string());
        let error2 = DetectedError::new("Error: something failed".to_string());

        assert_eq!(error1.content_hash, error2.content_hash);
    }

    #[test]
    fn test_error_dedup_ignores_timestamp() {
        let error1 = DetectedError::new("2024-01-01 12:34:56 Error: failed".to_string());
        let error2 = DetectedError::new("2024-01-02 23:45:67 Error: failed".to_string());

        // Hashes should be the same since timestamps are stripped
        assert_eq!(error1.content_hash, error2.content_hash);
    }

    #[test]
    fn test_error_dedup_ignores_line_numbers() {
        let error1 = DetectedError::new("Error at line 10".to_string());
        let error2 = DetectedError::new("Error at line 20".to_string());

        // Hashes should be the same since line numbers are stripped
        assert_eq!(error1.content_hash, error2.content_hash);
    }

    #[test]
    fn test_error_deduplicator_duplicate_detection() {
        let mut dedup = ErrorDeduplicator::new(Duration::from_secs(300));

        let error1 = DetectedError::new("Error: test".to_string());
        assert!(!dedup.is_duplicate(&error1)); // First time - not a duplicate

        let error2 = DetectedError::new("Error: test".to_string());
        assert!(dedup.is_duplicate(&error2)); // Second time - duplicate
    }

    #[test]
    fn test_error_deduplicator_different_errors() {
        let mut dedup = ErrorDeduplicator::new(Duration::from_secs(300));

        let error1 = DetectedError::new("Error: first error".to_string());
        let error2 = DetectedError::new("Error: second error".to_string());

        assert!(!dedup.is_duplicate(&error1));
        assert!(!dedup.is_duplicate(&error2));
    }

    // Phase 2.4: Watch Mode CLI Interface Tests

    #[test]
    fn test_watch_flag_file_path() {
        let args = Cli::try_parse_from(["why", "--watch", "/var/log/app.log"]).unwrap();
        assert_eq!(args.watch.as_deref(), Some("/var/log/app.log"));
    }

    #[test]
    fn test_watch_flag_command() {
        let args = Cli::try_parse_from(["why", "--watch", "npm run dev"]).unwrap();
        assert_eq!(args.watch.as_deref(), Some("npm run dev"));
    }

    #[test]
    fn test_watch_flag_short() {
        let args = Cli::try_parse_from(["why", "-w", "/tmp/test.log"]).unwrap();
        assert_eq!(args.watch.as_deref(), Some("/tmp/test.log"));
    }

    #[test]
    fn test_debounce_flag_parsing() {
        let args =
            Cli::try_parse_from(["why", "--watch", "/tmp/test.log", "--debounce", "1000"]).unwrap();
        assert_eq!(args.debounce, 1000);
    }

    #[test]
    fn test_debounce_flag_default() {
        let args = Cli::try_parse_from(["why", "--watch", "/tmp/test.log"]).unwrap();
        assert_eq!(args.debounce, 500); // Default value
    }

    #[test]
    fn test_no_dedup_flag() {
        let args = Cli::try_parse_from(["why", "--watch", "/tmp/test.log", "--no-dedup"]).unwrap();
        assert!(args.no_dedup);
    }

    #[test]
    fn test_pattern_flag_parsing() {
        let args =
            Cli::try_parse_from(["why", "--watch", "/tmp/test.log", "--pattern", "ERR-\\d+"])
                .unwrap();
        assert_eq!(args.pattern.as_deref(), Some("ERR-\\d+"));
    }

    #[test]
    fn test_clear_flag() {
        let args = Cli::try_parse_from(["why", "--watch", "/tmp/test.log", "--clear"]).unwrap();
        assert!(args.clear);
    }

    #[test]
    fn test_quiet_flag() {
        let args = Cli::try_parse_from(["why", "--watch", "/tmp/test.log", "--quiet"]).unwrap();
        assert!(args.quiet);
    }

    #[test]
    fn test_quiet_flag_short() {
        let args = Cli::try_parse_from(["why", "-w", "/tmp/test.log", "-q"]).unwrap();
        assert!(args.quiet);
    }

    // Phase 2.5: Watch Mode UX Tests

    #[test]
    fn test_watch_config_default() {
        let config = WatchConfig::default();
        assert_eq!(config.debounce_ms, 500);
        assert!(config.dedup);
        assert_eq!(config.dedup_ttl, Duration::from_secs(300));
        assert!(config.pattern.is_none());
        assert!(!config.clear);
        assert!(!config.quiet);
        assert_eq!(config.max_aggregation_lines, 50);
    }

    #[test]
    fn test_watch_session_toggle_pause() {
        let config = WatchConfig::default();
        let mut session = WatchSession::new(config);

        assert!(!session.is_paused());
        session.toggle_pause();
        assert!(session.is_paused());
        session.toggle_pause();
        assert!(!session.is_paused());
    }

    #[test]
    fn test_watch_session_toggle_dedup() {
        let config = WatchConfig::default();
        let mut session = WatchSession::new(config);

        assert!(session.config().dedup);
        session.toggle_dedup();
        assert!(!session.config().dedup);
        session.toggle_dedup();
        assert!(session.config().dedup);
    }

    #[test]
    fn test_watch_session_running_flag() {
        let config = WatchConfig::default();
        let session = WatchSession::new(config);

        let flag = session.running_flag();
        assert!(session.is_running());
        assert!(flag.load(std::sync::atomic::Ordering::SeqCst));

        session.stop();
        assert!(!session.is_running());
        assert!(!flag.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_watch_session_error_counting() {
        let config = WatchConfig {
            dedup: false, // Disable dedup for this test
            ..WatchConfig::default()
        };
        let mut session = WatchSession::new(config);

        // Process a few error lines
        session.process_line("Error: first");
        session.flush();
        session.process_line("Error: second");
        session.flush();

        assert!(session.status().contains("2/"));
    }

    #[test]
    fn test_watch_session_paused_ignores_errors() {
        let config = WatchConfig::default();
        let mut session = WatchSession::new(config);

        session.toggle_pause();
        let result = session.process_line("Error: should be ignored");
        assert!(result.is_none());
    }

    #[test]
    fn test_is_file_target_existing_file() {
        let temp_dir = std::env::temp_dir();
        let temp_file = temp_dir.join("test_is_file_target.log");
        std::fs::write(&temp_file, "").unwrap();

        assert!(is_file_target(&temp_file.to_string_lossy()));

        std::fs::remove_file(&temp_file).ok();
    }

    #[test]
    fn test_is_file_target_path_with_slash() {
        assert!(is_file_target("/var/log/app.log"));
        assert!(is_file_target("./local/file.log"));
    }

    #[test]
    fn test_is_file_target_command() {
        assert!(!is_file_target("npm run dev"));
        assert!(!is_file_target("cargo build"));
    }

    #[test]
    fn test_chrono_lite_now_format() {
        let time = chrono_lite_now();
        // Should be in HH:MM:SS format
        assert_eq!(time.len(), 8);
        assert!(time.chars().nth(2) == Some(':'));
        assert!(time.chars().nth(5) == Some(':'));
    }

    // ========================================================================
    // Daemon Mode Tests (Feature 5)
    // ========================================================================

    // Phase 5.1: IPC Architecture Tests

    #[test]
    #[cfg(unix)]
    fn test_socket_path_fallback_tmp() {
        // Temporarily unset XDG_RUNTIME_DIR
        let original = std::env::var("XDG_RUNTIME_DIR").ok();
        std::env::remove_var("XDG_RUNTIME_DIR");

        let path = get_socket_path();

        // Restore
        if let Some(val) = original {
            std::env::set_var("XDG_RUNTIME_DIR", val);
        }

        // Should be /tmp/why-<uid>.sock format
        let path_str = path.to_string_lossy();
        assert!(path_str.starts_with("/tmp/why-"));
        assert!(path_str.ends_with(".sock"));
    }

    #[test]
    #[cfg(unix)]
    fn test_pid_path_fallback_tmp() {
        // Temporarily unset XDG_RUNTIME_DIR
        let original = std::env::var("XDG_RUNTIME_DIR").ok();
        std::env::remove_var("XDG_RUNTIME_DIR");

        let path = get_pid_path();

        // Restore
        if let Some(val) = original {
            std::env::set_var("XDG_RUNTIME_DIR", val);
        }

        // Should be /tmp/why-<uid>.pid format
        let path_str = path.to_string_lossy();
        assert!(path_str.starts_with("/tmp/why-"));
        assert!(path_str.ends_with(".pid"));
    }

    #[test]
    fn test_protocol_request_serialization() {
        let request = DaemonRequest {
            action: DaemonAction::Explain,
            input: Some("error message".to_string()),
            options: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"action\":\"explain\""));
        assert!(json.contains("\"input\":\"error message\""));
    }

    #[test]
    fn test_protocol_request_ping() {
        let request = DaemonRequest {
            action: DaemonAction::Ping,
            input: None,
            options: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"action\":\"ping\""));
    }

    #[test]
    fn test_protocol_request_with_options() {
        let request = DaemonRequest {
            action: DaemonAction::Explain,
            input: Some("error".to_string()),
            options: Some(DaemonRequestOptions {
                stream: true,
                json: true,
                ..Default::default()
            }),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"stream\":true"));
        assert!(json.contains("\"json\":true"));
    }

    #[test]
    fn test_protocol_response_parsing() {
        let json = r#"{"type":"complete","explanation":{"error":"test","summary":"sum","explanation":"exp","suggestion":"sug"}}"#;
        let response: DaemonResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.response_type, DaemonResponseType::Complete);
        assert!(response.explanation.is_some());
        let exp = response.explanation.unwrap();
        assert_eq!(exp.error, "test");
        assert_eq!(exp.summary, "sum");
    }

    #[test]
    fn test_protocol_streaming_response() {
        let json = r#"{"type":"token","content":"Hello"}"#;
        let response: DaemonResponse = serde_json::from_str(json).unwrap();

        assert_eq!(response.response_type, DaemonResponseType::Token);
        assert_eq!(response.content.as_deref(), Some("Hello"));
    }

    #[test]
    fn test_protocol_pong_response() {
        let response = DaemonResponse::pong();
        let json = serde_json::to_string(&response).unwrap();

        assert!(json.contains("\"type\":\"pong\""));
    }

    #[test]
    fn test_protocol_stats_response() {
        let stats = DaemonStats {
            uptime_seconds: 3600,
            requests_served: 100,
            avg_response_time_ms: 150.5,
            memory_mb: 512.0,
            model_family: "Qwen".to_string(),
            model_loaded: true,
        };

        let response = DaemonResponse::stats(stats);
        let json = serde_json::to_string(&response).unwrap();

        assert!(json.contains("\"type\":\"stats\""));
        assert!(json.contains("\"uptime_seconds\":3600"));
        assert!(json.contains("\"requests_served\":100"));
    }

    #[test]
    fn test_protocol_error_response() {
        let response = DaemonResponse::error("Something went wrong");
        let json = serde_json::to_string(&response).unwrap();

        assert!(json.contains("\"type\":\"error\""));
        assert!(json.contains("\"error\":\"Something went wrong\""));
    }

    // Phase 5.3: Daemon Management CLI Tests

    #[test]
    fn test_daemon_subcommand_exists() {
        // Verify the daemon subcommand enum is properly defined
        let start = DaemonCommand::Start {
            foreground: false,
            idle_timeout: 30,
        };
        let stop = DaemonCommand::Stop { force: false };
        let restart = DaemonCommand::Restart { foreground: false };
        let status = DaemonCommand::Status;
        let install = DaemonCommand::InstallService;
        let uninstall = DaemonCommand::UninstallService;

        // Basic struct checks
        assert!(matches!(start, DaemonCommand::Start { .. }));
        assert!(matches!(stop, DaemonCommand::Stop { .. }));
        assert!(matches!(restart, DaemonCommand::Restart { .. }));
        assert!(matches!(status, DaemonCommand::Status));
        assert!(matches!(install, DaemonCommand::InstallService));
        assert!(matches!(uninstall, DaemonCommand::UninstallService));
    }

    #[test]
    fn test_daemon_start_options() {
        let start = DaemonCommand::Start {
            foreground: true,
            idle_timeout: 60,
        };
        if let DaemonCommand::Start {
            foreground,
            idle_timeout,
        } = start
        {
            assert!(foreground);
            assert_eq!(idle_timeout, 60);
        } else {
            panic!("Expected Start command");
        }
    }

    #[test]
    fn test_daemon_stop_options() {
        let stop = DaemonCommand::Stop { force: true };
        if let DaemonCommand::Stop { force } = stop {
            assert!(force);
        } else {
            panic!("Expected Stop command");
        }
    }

    // Phase 5.4: Client Mode Tests

    #[test]
    fn test_use_daemon_flag_parsing() {
        let args = Cli::try_parse_from(["why", "--use-daemon", "error message"]).unwrap();
        assert!(args.use_daemon);
    }

    #[test]
    fn test_use_daemon_flag_short_parsing() {
        let args = Cli::try_parse_from(["why", "-D", "error message"]).unwrap();
        assert!(args.use_daemon);
    }

    #[test]
    fn test_daemon_required_flag() {
        let args = Cli::try_parse_from(["why", "--daemon-required", "error message"]).unwrap();
        assert!(args.daemon_required);
    }

    #[test]
    fn test_no_auto_start_flag() {
        let args = Cli::try_parse_from(["why", "--use-daemon", "--no-auto-start", "error message"])
            .unwrap();
        assert!(args.use_daemon);
        assert!(args.no_auto_start);
    }

    // Phase 5.6: Resource Management Tests

    #[test]
    fn test_format_duration_seconds() {
        assert_eq!(format_duration(30), "30s");
        assert_eq!(format_duration(59), "59s");
    }

    #[test]
    fn test_format_duration_minutes() {
        assert_eq!(format_duration(60), "1m 0s");
        assert_eq!(format_duration(125), "2m 5s");
        assert_eq!(format_duration(3599), "59m 59s");
    }

    #[test]
    fn test_format_duration_hours() {
        assert_eq!(format_duration(3600), "1h 0m");
        assert_eq!(format_duration(7380), "2h 3m");
    }

    #[test]
    fn test_daemon_stats_default() {
        let stats = DaemonStats::default();
        assert_eq!(stats.uptime_seconds, 0);
        assert_eq!(stats.requests_served, 0);
        assert_eq!(stats.avg_response_time_ms, 0.0);
        assert!(!stats.model_loaded);
    }

    #[test]
    fn test_error_explanation_response_from() {
        let exp = ErrorExplanation {
            error: "test error".to_string(),
            summary: "test summary".to_string(),
            explanation: "test explanation".to_string(),
            suggestion: "test suggestion".to_string(),
        };

        let resp = ErrorExplanationResponse::from(&exp);
        assert_eq!(resp.error, "test error");
        assert_eq!(resp.summary, "test summary");
        assert_eq!(resp.explanation, "test explanation");
        assert_eq!(resp.suggestion, "test suggestion");
    }

    #[test]
    fn test_daemon_action_serialization() {
        assert_eq!(
            serde_json::to_string(&DaemonAction::Explain).unwrap(),
            "\"explain\""
        );
        assert_eq!(
            serde_json::to_string(&DaemonAction::Ping).unwrap(),
            "\"ping\""
        );
        assert_eq!(
            serde_json::to_string(&DaemonAction::Shutdown).unwrap(),
            "\"shutdown\""
        );
        assert_eq!(
            serde_json::to_string(&DaemonAction::Stats).unwrap(),
            "\"stats\""
        );
    }

    #[test]
    fn test_daemon_response_type_serialization() {
        assert_eq!(
            serde_json::to_string(&DaemonResponseType::Token).unwrap(),
            "\"token\""
        );
        assert_eq!(
            serde_json::to_string(&DaemonResponseType::Complete).unwrap(),
            "\"complete\""
        );
        assert_eq!(
            serde_json::to_string(&DaemonResponseType::Error).unwrap(),
            "\"error\""
        );
        assert_eq!(
            serde_json::to_string(&DaemonResponseType::Pong).unwrap(),
            "\"pong\""
        );
        assert_eq!(
            serde_json::to_string(&DaemonResponseType::Stats).unwrap(),
            "\"stats\""
        );
        assert_eq!(
            serde_json::to_string(&DaemonResponseType::ShutdownAck).unwrap(),
            "\"shutdown_ack\""
        );
    }

    #[test]
    #[cfg(unix)]
    fn test_is_daemon_running_no_socket() {
        // Make sure no socket exists
        let socket_path = get_socket_path();
        std::fs::remove_file(&socket_path).ok();

        assert!(!is_daemon_running());
    }
}
