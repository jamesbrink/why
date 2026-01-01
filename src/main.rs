use anyhow::{bail, Context, Result};
use clap::{CommandFactory, Parser};
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
use std::env;
use std::fs::File;
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

// Import from the library crate
use why::cli::{Cli, DaemonCommand};
use why::config::{print_hook_config, Config};
use why::daemon::{
    get_pid_path, get_socket_path, DaemonAction, DaemonRequest, DaemonResponse, DaemonResponseType,
    DaemonStats, ErrorExplanationResponse,
};
use why::hooks::{install_hook, uninstall_hook};
use why::model::{
    build_prompt, detect_model_family, get_model_path, is_degenerate_response, is_echo_response,
    run_inference_with_callback, ModelFamily, ModelPathInfo, SamplingParams, TokenCallback,
    MAX_RETRIES,
};
use why::output::{
    contains_error_patterns, format_file_line, interpret_exit_code, parse_response, print_colored,
    print_debug_section, print_frames, print_stats,
};
use why::stack_trace::{StackTraceJson, StackTraceParserRegistry};
use why::watch::{DetectedError, ErrorDeduplicator, ErrorDetector, WatchConfig};

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
            for line in reader.lines().map_while(Result::ok) {
                // Echo to terminal
                eprintln!("{}", line);
                // Send for processing
                let _ = stderr_tx.send(line);
            }
        })
    });

    // Spawn thread for stdout (just pass through, optionally process)
    let stdout_tx = line_tx;
    let stdout_reader = cmd_watcher.stdout_reader.take();
    let stdout_handle = stdout_reader.map(|reader| {
        thread::spawn(move || {
            for line in reader.lines().map_while(Result::ok) {
                // Echo to terminal
                println!("{}", line);
                // Send for processing
                let _ = stdout_tx.send(line);
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
#
# NOTE: Hook is disabled by default. Run `why --enable` to activate.

__why_stderr_file="/tmp/why_stderr_$$"
__why_last_cmd=""
__why_state_file="${{XDG_STATE_HOME:-$HOME/.local/state}}/why/hook_enabled"

# Check if hook is enabled
__why_is_enabled() {{
    [[ "$WHY_HOOK_ENABLE" == "1" ]] && return 0
    [[ "$WHY_HOOK_DISABLE" == "1" ]] && return 1
    [[ -f "$__why_state_file" ]] && [[ "$(cat "$__why_state_file" 2>/dev/null)" == "1" ]] && return 0
    return 1
}}

# Capture stderr while still displaying it
exec 2> >(tee -a "$__why_stderr_file" >&2)

__why_preexec() {{
    __why_last_cmd="$1"
    # Clear stderr capture file before each command
    : > "$__why_stderr_file" 2>/dev/null
}}

__why_prompt_command() {{
    local exit_code=$?
    if __why_is_enabled && [[ $exit_code -ne 0 && $exit_code -ne 130 && -n "$__why_last_cmd" ]]; then
        local output=""
        if [[ -f "$__why_stderr_file" && -s "$__why_stderr_file" ]]; then
            output=$(tail -100 "$__why_stderr_file" 2>/dev/null)
        fi
        if [[ -n "$output" ]]; then
            why --exit-code "$exit_code" --last-command "$__why_last_cmd" --last-output "$output" 2>/dev/null
        else
            why --exit-code "$exit_code" --last-command "$__why_last_cmd" 2>/dev/null
        fi
    fi
    __why_last_cmd=""
    return $exit_code
}}

trap '__why_preexec "$BASH_COMMAND"' DEBUG

# Preserve existing PROMPT_COMMAND
if [[ -z "$PROMPT_COMMAND" ]]; then
    PROMPT_COMMAND="__why_prompt_command"
else
    PROMPT_COMMAND="__why_prompt_command; $PROMPT_COMMAND"
fi

# Cleanup on exit
trap 'rm -f "$__why_stderr_file" 2>/dev/null' EXIT
"#
    );
}

fn print_zsh_hook() {
    println!(
        r#"# why - shell hook for zsh
# Add this to your ~/.zshrc:
#   eval "$(why --hook zsh)"
#
# NOTE: Hook is disabled by default. Run `why --enable` to activate.

__why_stderr_file="/tmp/why_stderr_$$"
__why_last_cmd=""
__why_state_file="${{XDG_STATE_HOME:-$HOME/.local/state}}/why/hook_enabled"

# Check if hook is enabled
__why_is_enabled() {{
    [[ "$WHY_HOOK_ENABLE" == "1" ]] && return 0
    [[ "$WHY_HOOK_DISABLE" == "1" ]] && return 1
    [[ -f "$__why_state_file" ]] && [[ "$(cat "$__why_state_file" 2>/dev/null)" == "1" ]] && return 0
    return 1
}}

# Capture stderr while still displaying it
exec 2> >(tee -a "$__why_stderr_file" >&2)

__why_preexec() {{
    __why_last_cmd="$1"
    # Clear stderr capture file before each command
    : > "$__why_stderr_file" 2>/dev/null
}}

__why_precmd() {{
    local exit_code=$?
    if __why_is_enabled && [[ $exit_code -ne 0 && $exit_code -ne 130 && -n "$__why_last_cmd" ]]; then
        local output=""
        if [[ -f "$__why_stderr_file" && -s "$__why_stderr_file" ]]; then
            output=$(tail -100 "$__why_stderr_file" 2>/dev/null)
        fi
        if [[ -n "$output" ]]; then
            why --exit-code "$exit_code" --last-command "$__why_last_cmd" --last-output "$output" 2>/dev/null
        else
            why --exit-code "$exit_code" --last-command "$__why_last_cmd" 2>/dev/null
        fi
    fi
    __why_last_cmd=""
    return $exit_code
}}

autoload -Uz add-zsh-hook
add-zsh-hook preexec __why_preexec
add-zsh-hook precmd __why_precmd

# Cleanup on exit
trap 'rm -f "$__why_stderr_file" 2>/dev/null' EXIT
"#
    );
}

fn print_fish_hook() {
    println!(
        r#"# why - shell hook for fish
# Add this to your ~/.config/fish/config.fish:
#   why --hook fish | source
#
# NOTE: Hook is disabled by default. Run `why --enable` to activate.

set -g __why_stderr_file "/tmp/why_stderr_"(echo %self)
set -g __why_state_file "$HOME/.local/state/why/hook_enabled"
if set -q XDG_STATE_HOME
    set __why_state_file "$XDG_STATE_HOME/why/hook_enabled"
end

# Check if hook is enabled
function __why_is_enabled
    test "$WHY_HOOK_ENABLE" = "1"; and return 0
    test "$WHY_HOOK_DISABLE" = "1"; and return 1
    test -f $__why_state_file; and test (cat $__why_state_file 2>/dev/null) = "1"; and return 0
    return 1
end

function __why_preexec --on-event fish_preexec
    # Clear stderr capture file before each command
    echo -n > $__why_stderr_file 2>/dev/null
end

function __why_postexec --on-event fish_postexec
    set -l exit_code $status
    if __why_is_enabled; and test $exit_code -ne 0 -a $exit_code -ne 130
        set -l output ""
        if test -f $__why_stderr_file -a -s $__why_stderr_file
            set output (tail -100 $__why_stderr_file 2>/dev/null | string collect)
        end
        if test -n "$output"
            why --exit-code $exit_code --last-command "$argv" --last-output "$output" 2>/dev/null
        else
            why --exit-code $exit_code --last-command "$argv" 2>/dev/null
        end
    end
end

# Cleanup on exit
function __why_cleanup --on-event fish_exit
    rm -f $__why_stderr_file 2>/dev/null
end
"#
    );
}

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
        "  {:<20} {:<12} ─────────────────────────────────────",
        "───────────────────", "──────────"
    );
    println!(
        "  {:<20} {:<12} Qwen2.5-Coder 0.5B - best quality {}",
        "why-qwen2_5-coder",
        "~530MB",
        "(default)".dimmed()
    );
    println!(
        "  {:<20} {:<12} Qwen3 0.6B - newest Qwen",
        "why-qwen3", "~639MB"
    );
    println!(
        "  {:<20} {:<12} Gemma 3 270M - Google",
        "why-gemma3", "~292MB"
    );
    println!(
        "  {:<20} {:<12} SmolLM2 135M - smallest/fastest",
        "why-smollm2", "~145MB"
    );
    println!();
    println!("{}", "Template Families".bold());
    println!();
    println!(
        "  Use {} to override auto-detection:",
        "--template <family>".cyan()
    );
    println!();
    println!("  {:<12} ChatML format (Qwen, SmolLM)", "qwen".green());
    println!("  {:<12} Gemma format (<start_of_turn>)", "gemma".green());
    println!("  {:<12} ChatML format (alias)", "smollm".green());
    println!();
}

fn print_provider_list() {
    use why::providers::{get_api_key_env_var, list_providers};

    println!("{}", "Available AI Providers".bold());
    println!();
    println!(
        "  {:<15} {:<35} {}",
        "Provider".blue().bold(),
        "Description".blue().bold(),
        "Status".blue().bold()
    );
    println!(
        "  {:<15} {:<35} ─────────",
        "───────────────", "───────────────────────────────────"
    );

    for (provider_type, description, available) in list_providers() {
        let status = if available {
            "Available".green().to_string()
        } else {
            let env_var = get_api_key_env_var(provider_type);
            format!("Set {}", env_var).red().to_string()
        };
        println!("  {:<15} {:<35} {}", provider_type, description, status);
    }

    println!();
    println!("{}", "Usage".bold());
    println!();
    println!("  Use {} to select a provider:", "--provider <name>".cyan());
    println!(
        "    {} --provider anthropic \"segmentation fault\"",
        "why".green()
    );
    println!();
    println!(
        "  Or set {} in your config or environment:",
        "WHY_PROVIDER".cyan()
    );
    println!("    export WHY_PROVIDER=anthropic");
    println!();
    println!("{}", "Environment Variables".bold());
    println!();
    println!("  {:<25} Anthropic API key", "ANTHROPIC_API_KEY".blue());
    println!("  {:<25} OpenAI API key", "OPENAI_API_KEY".blue());
    println!("  {:<25} OpenRouter API key", "OPENROUTER_API_KEY".blue());
    println!("  {:<25} Override default provider", "WHY_PROVIDER".blue());
    println!("  {:<25} Override model for provider", "WHY_MODEL".blue());
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

    // Handle --list-providers
    if cli.list_providers {
        print_provider_list();
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

    // Handle --enable
    if cli.enable {
        return why::hooks::enable_hook();
    }

    // Handle --disable
    if cli.disable {
        return why::hooks::disable_hook();
    }

    // Handle --status
    if cli.status {
        why::hooks::print_hook_status();
        return Ok(());
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

    // Check if hook is disabled via environment variable or state file
    if (Config::is_hook_disabled() || !why::hooks::is_hook_enabled())
        && (cli.capture || cli.exit_code.is_some())
    {
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

        if let Some(ref output) = cli.last_output {
            // Full context: command + output + exit code
            format!(
                "Command: {}\nExit code: {} ({})\n\nOutput:\n{}\n\nExplain why this command failed.",
                command.trim(),
                exit_code,
                interpretation,
                output.trim()
            )
        } else {
            // Basic context: just command + exit code
            format!(
                "Command: {}\nExit code: {} ({})\n\nExplain why this command failed.",
                command.trim(),
                exit_code,
                interpretation
            )
        }
    } else if cli.exit_code.is_some() || cli.last_command.is_some() {
        // Partial hook mode - try to use what we have
        if let Some(exit_code) = cli.exit_code {
            let interpretation = interpret_exit_code(exit_code);
            if let Some(ref output) = cli.last_output {
                format!(
                    "Exit code: {} ({})\n\nOutput:\n{}\n\nExplain this error.",
                    exit_code,
                    interpretation,
                    output.trim()
                )
            } else {
                format!(
                    "Exit code: {} ({})\n\nExplain what this exit code means.",
                    exit_code, interpretation
                )
            }
        } else if let Some(ref command) = cli.last_command {
            if let Some(ref output) = cli.last_output {
                format!(
                    "Command failed: {}\n\nOutput:\n{}\n\nExplain why this failed.",
                    command.trim(),
                    output.trim()
                )
            } else {
                format!(
                    "Command failed: {}\n\nExplain why this might have failed.",
                    command.trim()
                )
            }
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
