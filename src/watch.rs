//! Watch mode functionality for monitoring files and commands for errors.
//!
//! This module provides the types and utilities for watch mode, which monitors
//! files or command output for errors and explains them as they appear.
//!
//! Note: The main watch mode implementation is currently in main.rs due to
//! tight coupling with the inference and CLI systems. This module exports
//! the core types used by watch mode.

use regex::Regex;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

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
    pub fn compute_hash(content: &str) -> u64 {
        let normalized = Self::normalize_for_hash(content);
        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        hasher.finish()
    }

    /// Normalize content for hashing by stripping timestamps and line numbers
    pub fn normalize_for_hash(content: &str) -> String {
        let timestamp_re =
            Regex::new(r"(?:\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|\[\d{10,}\]|\d{2}/\d{2}/\d{4})")
                .unwrap();
        let line_re = Regex::new(r"(?::\d+:|line \d+|at line \d+)").unwrap();

        let mut normalized = timestamp_re.replace_all(content, "").to_string();
        normalized = line_re.replace_all(&normalized, "").to_string();
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
        self.cleanup();

        if let Some(&last_seen) = self.seen.get(&error.content_hash) {
            if last_seen.elapsed() < self.ttl {
                return true;
            }
        }

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
        if let Some(ref pattern) = self.pattern {
            return pattern.is_match(line);
        }

        let lower = line.to_lowercase();

        let starts_with_error = lower.starts_with("error")
            || lower.starts_with("e:")
            || lower.starts_with("err:")
            || lower.starts_with("fatal")
            || lower.starts_with("panic")
            || lower.starts_with("exception")
            || lower.starts_with("traceback");

        let contains_error = lower.contains("error:")
            || lower.contains("error[e")
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

        let is_stack_trace = line.trim().starts_with("at ")
            || line.contains("File \"")
            || line.contains("at /")
            || lower.contains("traceback (most recent call last)");

        starts_with_error || contains_error || is_stack_trace
    }

    /// Process a line and return detected errors
    pub fn process_line(&mut self, line: &str) -> Option<DetectedError> {
        let trimmed = line.trim();
        let is_blank = trimmed.is_empty();
        let is_error = self.is_error_line(line);

        if is_blank {
            self.blank_count += 1;
        } else {
            self.blank_count = 0;
        }

        if is_error {
            self.in_error = true;
            self.aggregation_buffer.push(line.to_string());
            None
        } else if self.in_error {
            // Flush if we hit two blank lines or reach max buffer size
            if (is_blank && self.blank_count >= 2)
                || self.aggregation_buffer.len() >= self.max_lines
            {
                return self.flush_error();
            } else if !is_blank {
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
