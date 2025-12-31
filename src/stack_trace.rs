//! Stack trace parsing and intelligence for multiple programming languages.
//!
//! This module provides parsers for stack traces from Python, Rust, JavaScript,
//! Go, Java, and C/C++, with features for:
//! - Auto-detection of language from stack trace patterns
//! - Extraction of file, line, function information
//! - User code vs framework code classification
//! - Source context extraction

use serde::Serialize;
use std::fmt;
use std::path::{Path, PathBuf};

// ============================================================================
// Core Types
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

// ============================================================================
// Parser Trait and Registry
// ============================================================================

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
// Python Parser
// ============================================================================

/// Python stack trace parser
pub struct PythonStackTraceParser;

impl PythonStackTraceParser {
    /// Parse a Python exception line like "KeyError: 'missing_key'" or "ValueError: invalid literal"
    fn parse_exception_line(line: &str) -> Option<(String, String)> {
        let exception_patterns = ["Error:", "Exception:", "Warning:", "Interrupt:"];

        for pattern in exception_patterns {
            if let Some(pos) = line.find(pattern) {
                let before = &line[..pos + pattern.len() - 1];
                let error_type = before
                    .split_whitespace()
                    .last()
                    .unwrap_or(before)
                    .to_string();
                let message = line[pos + pattern.len()..].trim().to_string();
                return Some((error_type, message));
            }
        }

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

        if let Some(start) = trimmed.find('"') {
            if let Some(end) = trimmed[start + 1..].find('"') {
                let file_path = &trimmed[start + 1..start + 1 + end];
                frame.file = Some(PathBuf::from(file_path));
                frame.is_user_code = !Self::is_framework_path(file_path);
            }
        }

        if let Some(line_pos) = trimmed.find(", line ") {
            let after_line = &trimmed[line_pos + 7..];
            let line_end = after_line
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(after_line.len());
            if let Ok(line_num) = after_line[..line_end].parse::<u32>() {
                frame.line = Some(line_num);
            }
        }

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

            if trimmed == "Traceback (most recent call last):" {
                in_traceback = true;
                continue;
            }

            if trimmed.starts_with("During handling of the above exception")
                || trimmed.starts_with("The above exception was the direct cause")
            {
                in_traceback = true;
                continue;
            }

            if in_traceback {
                if let Some(mut frame) = Self::parse_file_line(line) {
                    if let Some(next_line) = lines.peek() {
                        let next_trimmed = next_line.trim();
                        if !next_trimmed.starts_with("File")
                            && !next_trimmed.is_empty()
                            && Self::parse_exception_line(next_trimmed).is_none()
                        {
                            frame.context = Some(next_trimmed.to_string());
                            lines.next();
                        }
                    }
                    trace.add_frame(frame);
                    continue;
                }

                if let Some((error_type, message)) = Self::parse_exception_line(trimmed) {
                    trace.error_type = error_type;
                    trace.error_message = message;
                    in_traceback = false;
                    continue;
                }
            } else if let Some((error_type, message)) = Self::parse_exception_line(trimmed) {
                if trace.error_type.is_empty() {
                    trace.error_type = error_type;
                    trace.error_message = message;
                }
            }
        }

        if trace.error_type.is_empty() && !trace.frames.is_empty() {
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

// ============================================================================
// Rust Parser
// ============================================================================

/// Rust stack trace parser
pub struct RustStackTraceParser;

impl RustStackTraceParser {
    fn parse_panic_line(line: &str) -> Option<(String, Option<StackFrame>)> {
        if !line.contains("panicked at") {
            return None;
        }

        let mut message = String::new();
        let mut frame = None;

        if let Some(at_pos) = line.find("panicked at ") {
            let after_at = &line[at_pos + 12..];

            if let Some(stripped) = after_at.strip_prefix('\'') {
                if let Some(end_quote) = stripped.find('\'') {
                    message = stripped[..end_quote].to_string();
                    let after_msg = &stripped[end_quote + 1..];
                    if let Some(loc) = after_msg.strip_prefix(", ") {
                        frame = Self::parse_location(loc);
                    }
                }
            } else {
                frame = Self::parse_location(after_at.trim_end_matches(':'));
            }
        }

        Some((message, frame))
    }

    fn parse_location(loc: &str) -> Option<StackFrame> {
        let parts: Vec<&str> = loc.rsplitn(3, ':').collect();
        if parts.len() >= 2 {
            let mut frame = StackFrame::new();

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

            if let Some(ref file) = frame.file {
                frame.is_user_code = !Self::is_framework_path(&file.to_string_lossy());
            }

            return Some(frame);
        }
        None
    }

    fn parse_backtrace_frame(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();

        if let Some(colon_pos) = trimmed.find(':') {
            let num_part = trimmed[..colon_pos].trim();
            if num_part.chars().all(|c| c.is_ascii_digit()) {
                let func_part = trimmed[colon_pos + 1..].trim();
                let mut frame = StackFrame::new().with_function(func_part);
                frame.is_user_code = !Self::is_framework_path(func_part);
                return Some(frame);
            }
        }

        if let Some(loc) = trimmed.strip_prefix("at ") {
            return Self::parse_location(loc);
        }

        None
    }

    fn parse_compiler_error(line: &str) -> Option<(String, String)> {
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

    fn parse_source_location(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();
        if let Some(loc) = trimmed.strip_prefix("-->") {
            return Self::parse_location(loc.trim());
        }
        None
    }

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

            if let Some((error_type, message)) = Self::parse_compiler_error(trimmed) {
                trace.error_type = error_type;
                trace.error_message = message;
                continue;
            }

            if let Some(frame) = Self::parse_source_location(trimmed) {
                trace.add_frame(frame);
                continue;
            }

            if trimmed == "stack backtrace:" {
                in_backtrace = true;
                continue;
            }

            if in_backtrace {
                if let Some(frame) = Self::parse_backtrace_frame(trimmed) {
                    trace.add_frame(frame);
                }
            }
        }

        Some(trace)
    }
}

// ============================================================================
// JavaScript Parser
// ============================================================================

/// JavaScript/Node.js stack trace parser
pub struct JavaScriptStackTraceParser;

impl JavaScriptStackTraceParser {
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
                if let Some(after_colon) = rest.strip_prefix(':') {
                    return Some((error_type.to_string(), after_colon.trim().to_string()));
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

    fn parse_frame_line(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();
        if !trimmed.starts_with("at ") {
            return None;
        }

        let content = &trimmed[3..].trim();
        let mut frame = StackFrame::new();

        if let Some(paren_start) = content.find(" (") {
            if let Some(paren_end) = content.rfind(')') {
                let func_name = &content[..paren_start];
                frame.function = Some(func_name.to_string());

                let location = &content[paren_start + 2..paren_end];
                Self::parse_location(location, &mut frame);
            }
        } else if content.contains(':') {
            Self::parse_location(content, &mut frame);
        } else {
            frame.function = Some(content.to_string());
        }

        frame.is_user_code = !Self::is_framework_path(&frame);

        Some(frame)
    }

    fn parse_location(loc: &str, frame: &mut StackFrame) {
        let parts: Vec<&str> = loc.rsplitn(3, ':').collect();

        if parts.len() >= 2 {
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

            if trace.error_type.is_empty() {
                if let Some((error_type, message)) = Self::parse_error_line(trimmed) {
                    trace.error_type = error_type;
                    trace.error_message = message;
                    continue;
                }
            }

            if let Some(frame) = Self::parse_frame_line(line) {
                trace.add_frame(frame);
            }
        }

        Some(trace)
    }
}

// ============================================================================
// Go Parser
// ============================================================================

/// Go stack trace parser
pub struct GoStackTraceParser;

impl GoStackTraceParser {
    fn parse_panic_line(line: &str) -> Option<String> {
        if let Some(rest) = line.strip_prefix("panic:") {
            return Some(rest.trim().to_string());
        }
        None
    }

    fn parse_frame_line(line: &str, prev_function: &mut Option<String>) -> Option<StackFrame> {
        let trimmed = line.trim();

        if trimmed.ends_with(')') && trimmed.contains('(') {
            let paren_pos = trimmed.find('(').unwrap();
            *prev_function = Some(trimmed[..paren_pos].to_string());
            return None;
        }

        if trimmed.contains(".go:") {
            let mut frame = StackFrame::new();

            if let Some(ref func) = prev_function {
                frame.function = Some(func.clone());
                frame.is_user_code = !Self::is_framework_function(func);
            }
            *prev_function = None;

            if let Some(colon_pos) = trimmed.rfind(':') {
                let file_part = &trimmed[..colon_pos];
                let after_colon = &trimmed[colon_pos + 1..];

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

            if trace.error_message.is_empty() {
                if let Some(message) = Self::parse_panic_line(trimmed) {
                    trace.error_type = "panic".to_string();
                    trace.error_message = message;
                    continue;
                }
            }

            if trimmed.starts_with("goroutine ") {
                continue;
            }

            if let Some(frame) = Self::parse_frame_line(line, &mut prev_function) {
                trace.add_frame(frame);
            }
        }

        Some(trace)
    }
}

// ============================================================================
// Java Parser
// ============================================================================

/// Java/JVM stack trace parser
pub struct JavaStackTraceParser;

impl JavaStackTraceParser {
    fn parse_exception_line(line: &str) -> Option<(String, String)> {
        let trimmed = line.trim();

        if trimmed.starts_with("Exception in thread") {
            if let Some(quote_end) = trimmed[21..].find('"') {
                let after_thread = &trimmed[21 + quote_end + 1..].trim();
                return Self::parse_exception_type(after_thread);
            }
        }

        if let Some(rest) = trimmed.strip_prefix("Caused by:") {
            let after = rest.trim();
            return Self::parse_exception_type(after);
        }

        if trimmed.contains("Exception") || trimmed.contains("Error") {
            return Self::parse_exception_type(trimmed);
        }

        None
    }

    fn parse_exception_type(text: &str) -> Option<(String, String)> {
        if let Some(colon_pos) = text.find(':') {
            let error_type = text[..colon_pos].trim().to_string();
            let message = text[colon_pos + 1..].trim().to_string();
            return Some((error_type, message));
        }
        let error_type = text.split_whitespace().next()?.to_string();
        if error_type.contains('.')
            || error_type.ends_with("Exception")
            || error_type.ends_with("Error")
        {
            return Some((error_type, String::new()));
        }
        None
    }

    fn parse_frame_line(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();

        if !trimmed.starts_with("at ") {
            return None;
        }

        let content = &trimmed[3..];
        let mut frame = StackFrame::new();

        if let Some(paren_start) = content.find('(') {
            if let Some(paren_end) = content.find(')') {
                let method_full = &content[..paren_start];
                frame.function = Some(method_full.to_string());
                frame.is_user_code = !Self::is_framework_class(method_full);

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
            if trace.error_type.is_empty() {
                if let Some((error_type, message)) = Self::parse_exception_line(line) {
                    trace.error_type = error_type;
                    trace.error_message = message;
                    continue;
                }
            }

            if line.trim().starts_with("Caused by:") {
                if let Some((error_type, message)) = Self::parse_exception_line(line) {
                    trace.error_type = error_type;
                    trace.error_message = message;
                }
                continue;
            }

            if let Some(frame) = Self::parse_frame_line(line) {
                trace.add_frame(frame);
            }
        }

        Some(trace)
    }
}

// ============================================================================
// C/C++ Parser
// ============================================================================

/// C/C++ stack trace parser
pub struct CppStackTraceParser;

impl CppStackTraceParser {
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

    fn parse_gdb_frame(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();

        if !trimmed.starts_with('#') {
            return None;
        }

        let after_hash = &trimmed[1..];
        let num_end = after_hash.find(|c: char| !c.is_ascii_digit()).unwrap_or(0);
        if num_end == 0 {
            return None;
        }

        let rest = after_hash[num_end..].trim();
        let mut frame = StackFrame::new();

        let content = if rest.starts_with("0x") {
            if let Some(space) = rest.find(' ') {
                rest[space..].trim()
            } else {
                rest
            }
        } else {
            rest
        };

        let (func_part, location_part) = if let Some(at_pos) = content.find(" at ") {
            (&content[..at_pos], Some(&content[at_pos + 4..]))
        } else if let Some(from_pos) = content.find(" from ") {
            (&content[..from_pos], None)
        } else {
            (content, None)
        };

        let func_name = if let Some(in_pos) = func_part.find(" in ") {
            &func_part[in_pos + 4..]
        } else if let Some(rest) = func_part.strip_prefix("in ") {
            rest
        } else {
            func_part
        };

        let func_clean = if let Some(paren) = func_name.find('(') {
            &func_name[..paren]
        } else {
            func_name
        };

        if !func_clean.is_empty() {
            frame.function = Some(func_clean.trim().to_string());
            frame.is_user_code = !Self::is_framework_function(func_clean);
        }

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

    fn parse_asan_frame(line: &str) -> Option<StackFrame> {
        let trimmed = line.trim();

        if !trimmed.starts_with('#') {
            return None;
        }

        let mut frame = StackFrame::new();

        if let Some(in_pos) = trimmed.find(" in ") {
            let after_in = &trimmed[in_pos + 4..];

            if let Some(space) = after_in.rfind(' ') {
                let func = &after_in[..space];
                let loc = &after_in[space + 1..];

                frame.function = Some(func.trim().to_string());
                frame.is_user_code = !Self::is_framework_function(func);

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
            if trace.error_type.is_empty() {
                if let Some((error_type, message)) = Self::parse_asan_line(line) {
                    trace.error_type = error_type;
                    trace.error_message = message;
                    continue;
                }
            }

            if trace.error_message.is_empty() {
                if let Some(signal) = Self::parse_signal_line(line) {
                    trace.error_type = "signal".to_string();
                    trace.error_message = signal;
                    continue;
                }
            }

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
// Source Context
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
            max_context_chars: 4096,
        }
    }
}

/// Extract source context for a stack frame
#[allow(dead_code)]
pub fn extract_frame_context(frame: &StackFrame, config: &SourceContextConfig) -> Option<String> {
    let file_path = frame.file.as_ref()?;
    let line_num = frame.line?;

    let resolved_path = resolve_source_path(file_path, &config.context_root)?;

    let contents = std::fs::read_to_string(&resolved_path).ok()?;
    let lines: Vec<&str> = contents.lines().collect();

    let line_idx = line_num.saturating_sub(1) as usize;
    let start = line_idx.saturating_sub(config.context_lines);
    let end = (line_idx + config.context_lines + 1).min(lines.len());

    if start >= lines.len() {
        return None;
    }

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

/// Resolve a source file path, optionally prepending context_root
#[allow(dead_code)]
pub fn resolve_source_path(path: &Path, context_root: &Option<PathBuf>) -> Option<PathBuf> {
    if path.is_absolute() {
        if path.exists() {
            return Some(path.to_path_buf());
        }
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

/// Extract context for all user code frames in a stack trace
#[allow(dead_code)]
pub fn extract_stack_trace_context(trace: &StackTrace, config: &SourceContextConfig) -> String {
    let mut total_context = String::new();
    let mut total_chars = 0;

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
// JSON Types
// ============================================================================

/// JSON representation of a stack frame for structured output
#[derive(Debug, Serialize)]
pub struct StackFrameJson {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub column: Option<u32>,
    pub is_user_code: bool,
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
pub struct StackTraceJson {
    pub language: Language,
    pub error_type: String,
    pub error_message: String,
    pub frames: Vec<StackFrameJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub root_cause_frame: Option<StackFrameJson>,
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
        let frame = StackFrame::new();
        assert!(frame.function.is_none());
        assert!(frame.file.is_none());
        assert!(frame.line.is_none());
        assert!(frame.column.is_none());
        assert!(frame.is_user_code);
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
            .with_file("test.py")
            .with_line(42);

        let json_frame = StackFrameJson::from(&frame);
        let json = serde_json::to_string(&json_frame).unwrap();
        assert!(json.contains("test_func"));
        assert!(json.contains("42"));
    }
}
