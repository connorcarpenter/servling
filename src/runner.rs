//! CLI execution logic for LLM backends.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::Serialize;
use serde_json::Value;

use crate::core::OutcomeClassification;
use crate::token_usage::TokenUsage;

struct StreamLine {
    text: String,
    newline: bool,
}

#[derive(Debug, Serialize)]
struct RunnerMetadata {
    program: String,
    args: Vec<String>,
    working_dir: String,
    explicit_env_keys: Vec<String>,
    auth_basis: RunnerAuthBasis,
}

#[derive(Debug, Serialize)]
struct RunnerAuthBasis {
    home: Option<String>,
    xdg_config_home: Option<String>,
    xdg_cache_home: Option<String>,
    xdg_data_home: Option<String>,
    copilot_github_token_present: bool,
    gh_token_present: bool,
    github_token_present: bool,
    copilot_config_dir: Option<String>,
    copilot_config_dir_exists: bool,
}

fn format_stream_line(line: &str) -> Option<StreamLine> {
    let trimmed = line.trim();
    if !trimmed.starts_with('{') {
        return Some(StreamLine {
            text: line.to_string(),
            newline: true,
        });
    }

    let value: Value = match serde_json::from_str(trimmed) {
        Ok(val) => val,
        Err(_) => {
            return Some(StreamLine {
                text: line.to_string(),
                newline: true,
            })
        }
    };

    let line_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match line_type {
        "stream_event" => format_claude_stream_event(&value),
        "user" => format_claude_user_event(&value),
        "assistant" | "result" | "system" => None,
        _ => Some(StreamLine {
            text: line.to_string(),
            newline: true,
        }),
    }
}

fn format_claude_stream_event(value: &Value) -> Option<StreamLine> {
    let event = value.get("event")?;
    let event_type = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match event_type {
        "content_block_delta" => {
            let delta = event.get("delta")?;
            let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if delta_type == "text_delta" {
                let text = delta.get("text").and_then(|v| v.as_str()).unwrap_or("");
                if text.is_empty() {
                    return None;
                }
                return Some(StreamLine {
                    text: text.to_string(),
                    newline: false,
                });
            }
            None
        }
        "content_block_start" => {
            let block = event.get("content_block")?;
            let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if block_type == "tool_use" {
                let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                return Some(StreamLine {
                    text: format!("\n[tool] {}", name),
                    newline: true,
                });
            }
            None
        }
        "message_stop" => Some(StreamLine {
            text: String::new(),
            newline: true,
        }),
        _ => None,
    }
}

fn format_claude_user_event(value: &Value) -> Option<StreamLine> {
    if let Some(tool_result) = value.get("tool_use_result") {
        let stdout = tool_result
            .get("stdout")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let stderr = tool_result
            .get("stderr")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let content = if !stdout.trim().is_empty() {
            stdout
        } else {
            stderr
        };
        if !content.trim().is_empty() {
            return Some(StreamLine {
                text: format!("\n[tool_result] {}", truncate_line(content, 200)),
                newline: true,
            });
        }
    }

    let content_array = value
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array());

    if let Some(items) = content_array {
        for item in items {
            let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if item_type == "tool_result" {
                let content = item.get("content").and_then(|v| v.as_str()).unwrap_or("");
                if !content.trim().is_empty() {
                    return Some(StreamLine {
                        text: format!("\n[tool_result] {}", truncate_line(content, 200)),
                        newline: true,
                    });
                }
            }
        }
    }

    None
}

fn wait_with_streaming(
    mut child: Child,
    timeout: Duration,
    stream: bool,
) -> Result<(ExitStatus, Vec<u8>, Vec<u8>), std::io::Error> {
    let stdout_handle = child.stdout.take();
    let stderr_handle = child.stderr.take();

    let stdout_buf = Arc::new(Mutex::new(Vec::new()));
    let stderr_buf = Arc::new(Mutex::new(Vec::new()));

    let stdout_buf_clone = Arc::clone(&stdout_buf);
    let stderr_buf_clone = Arc::clone(&stderr_buf);

    let stdout_thread = stdout_handle.map(|stdout| {
        thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if stream {
                        if let Some(rendered) = format_stream_line(&line) {
                            if rendered.newline {
                                let _ = writeln!(std::io::stdout(), "{}", rendered.text);
                            } else {
                                let _ = write!(std::io::stdout(), "{}", rendered.text);
                                let _ = std::io::stdout().flush();
                            }
                        }
                    }
                    let mut buf = stdout_buf_clone.lock().unwrap();
                    buf.extend_from_slice(line.as_bytes());
                    buf.push(b'\n');
                }
            }
        })
    });

    let stderr_thread = stderr_handle.map(|stderr| {
        thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if stream {
                        if let Some(rendered) = format_stream_line(&line) {
                            if rendered.newline {
                                let _ = writeln!(std::io::stderr(), "{}", rendered.text);
                            } else {
                                let _ = write!(std::io::stderr(), "{}", rendered.text);
                                let _ = std::io::stderr().flush();
                            }
                        }
                    }
                    let mut buf = stderr_buf_clone.lock().unwrap();
                    buf.extend_from_slice(line.as_bytes());
                    buf.push(b'\n');
                }
            }
        })
    });

    let start = Instant::now();
    let poll_interval = Duration::from_millis(100);

    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                if let Some(t) = stdout_thread {
                    let _ = t.join();
                }
                if let Some(t) = stderr_thread {
                    let _ = t.join();
                }

                let stdout = match Arc::try_unwrap(stdout_buf) {
                    Ok(mutex) => mutex.into_inner().unwrap_or_default(),
                    Err(arc) => arc.lock().unwrap().clone(),
                };
                let stderr = match Arc::try_unwrap(stderr_buf) {
                    Ok(mutex) => mutex.into_inner().unwrap_or_default(),
                    Err(arc) => arc.lock().unwrap().clone(),
                };

                return Ok((status, stdout, stderr));
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "Process timed out",
                    ));
                }
                thread::sleep(poll_interval);
            }
            Err(e) => return Err(e),
        }
    }
}

pub struct CliRunnerConfig {
    pub working_dir: PathBuf,
    pub max_runtime_seconds: u32,
    pub stream_output: bool,
    pub env: Vec<(String, String)>,
}

pub struct CliRunnerOutcome {
    pub exit_code: Option<i32>,
    pub classification: OutcomeClassification,
    pub elapsed_seconds: f64,
    pub stdout_path: Option<String>,
    pub stderr_path: Option<String>,
    pub token_usage: Option<TokenUsage>,
}

pub fn run_cli_runner(
    command_template: &str,
    mission_dir: &Path,
    config: &CliRunnerConfig,
    _extract_error: bool,
    prompt_override: Option<String>,
    input_file: Option<&Path>,
    output_file: Option<&Path>,
) -> Result<CliRunnerOutcome> {
    let mut expanded_cmd = command_template.to_string();
    if let Some(path) = input_file {
        expanded_cmd = expanded_cmd.replace("{input_file}", &path.display().to_string());
    }
    if let Some(path) = output_file {
        expanded_cmd = expanded_cmd.replace("{output_file}", &path.display().to_string());
    }

    let parts: Vec<&str> = expanded_cmd.split_whitespace().collect();

    if parts.is_empty() {
        return Ok(CliRunnerOutcome {
            exit_code: None,
            classification: OutcomeClassification::EnvironmentError,
            elapsed_seconds: 0.0,
            stdout_path: None,
            stderr_path: None,
            token_usage: None,
        });
    }

    let program = parts[0];
    let args: Vec<&str> = parts[1..].to_vec();

    let start = Instant::now();
    let timeout = Duration::from_secs(config.max_runtime_seconds as u64);

    let mission_dir_abs =
        std::fs::canonicalize(mission_dir).unwrap_or_else(|_| mission_dir.to_path_buf());

    let prompt = if let Some(p) = prompt_override {
        Some(p)
    } else {
        let mission_path = mission_dir.join("MISSION.md");
        if mission_path.exists() {
            std::fs::read_to_string(&mission_path).ok()
        } else {
            None
        }
    };

    let mut cmd = Command::new(program);
    cmd.args(&args)
        .current_dir(&config.working_dir)
        .env("TESAKI_MISSION_DIR", &mission_dir_abs)
        .envs(config.env.iter().map(|(key, value)| (key, value)))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let use_stdin = prompt.is_some() && input_file.is_none();
    if use_stdin {
        cmd.stdin(Stdio::piped());
    } else {
        cmd.stdin(Stdio::null());
    }

    let mut child = cmd.spawn()?;

    if use_stdin {
        if let Some(mut stdin) = child.stdin.take() {
            if let Some(p) = prompt {
                let _ = stdin.write_all(p.as_bytes());
            }
        }
    }

    let (status, stdout_bytes, stderr_bytes) =
        match wait_with_streaming(child, timeout, config.stream_output) {
            Ok(result) => result,
            Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
                return Ok(CliRunnerOutcome {
                    exit_code: None,
                    classification: OutcomeClassification::Timeout,
                    elapsed_seconds: start.elapsed().as_secs_f64(),
                    stdout_path: None,
                    stderr_path: None,
                    token_usage: None,
                });
            }
            Err(e) => return Err(e.into()),
        };

    let output = std::process::Output {
        status,
        stdout: stdout_bytes,
        stderr: stderr_bytes,
    };

    let elapsed = start.elapsed().as_secs_f64();
    let exit_code = output.status.code();
    let classification = classify_process_output(&output);

    let output_dir = mission_dir.join("RUNNER_OUTPUT");
    let _ = std::fs::create_dir_all(&output_dir);

    let metadata = build_runner_metadata(program, &args, config);
    if let Ok(metadata_json) = serde_json::to_string_pretty(&metadata) {
        let metadata_path = output_dir.join("runner_metadata.json");
        let _ = std::fs::write(&metadata_path, metadata_json);
    }

    let stdout_path = if !output.stdout.is_empty() {
        let path = output_dir.join("runner_stdout.txt");
        let _ = std::fs::write(&path, &output.stdout);
        Some(path.display().to_string())
    } else {
        None
    };

    let stderr_path = if !output.stderr.is_empty() {
        let path = output_dir.join("runner_stderr.txt");
        let _ = std::fs::write(&path, &output.stderr);
        Some(path.display().to_string())
    } else {
        None
    };

    let stderr_str = String::from_utf8_lossy(&output.stderr);
    let token_usage = TokenUsage::parse(&stderr_str);
    let token_usage = if token_usage.has_data() {
        Some(token_usage)
    } else {
        None
    };

    if let Some(ref usage) = token_usage {
        let usage_path = output_dir.join("token_usage.json");
        if let Ok(usage_json) = serde_json::to_string_pretty(usage) {
            let _ = std::fs::write(&usage_path, usage_json);
        }
    }

    Ok(CliRunnerOutcome {
        exit_code,
        classification,
        elapsed_seconds: elapsed,
        stdout_path,
        stderr_path,
        token_usage,
    })
}

fn build_runner_metadata(program: &str, args: &[&str], config: &CliRunnerConfig) -> RunnerMetadata {
    let explicit_env_keys = config.env.iter().map(|(key, _)| key.clone()).collect();
    RunnerMetadata {
        program: program.to_string(),
        args: args.iter().map(|arg| (*arg).to_string()).collect(),
        working_dir: config.working_dir.display().to_string(),
        explicit_env_keys,
        auth_basis: collect_auth_basis(program, args),
    }
}

fn collect_auth_basis(program: &str, args: &[&str]) -> RunnerAuthBasis {
    let home = std::env::var("HOME").ok();
    let xdg_config_home = std::env::var("XDG_CONFIG_HOME").ok();
    let xdg_cache_home = std::env::var("XDG_CACHE_HOME").ok();
    let xdg_data_home = std::env::var("XDG_DATA_HOME").ok();
    let copilot_config_dir = if program == "copilot" {
        resolve_copilot_config_dir(args, home.as_deref())
    } else {
        None
    };
    let copilot_config_dir_exists = copilot_config_dir
        .as_ref()
        .map(|dir| Path::new(dir).exists())
        .unwrap_or(false);

    RunnerAuthBasis {
        home,
        xdg_config_home,
        xdg_cache_home,
        xdg_data_home,
        copilot_github_token_present: std::env::var_os("COPILOT_GITHUB_TOKEN").is_some(),
        gh_token_present: std::env::var_os("GH_TOKEN").is_some(),
        github_token_present: std::env::var_os("GITHUB_TOKEN").is_some(),
        copilot_config_dir,
        copilot_config_dir_exists,
    }
}

fn resolve_copilot_config_dir(args: &[&str], home: Option<&str>) -> Option<String> {
    if let Some(config_dir) = find_flag_value(args, "--config-dir") {
        return Some(config_dir.to_string());
    }
    home.map(|home| format!("{home}/.copilot"))
}

fn find_flag_value<'a>(args: &'a [&'a str], flag: &str) -> Option<&'a str> {
    let mut iter = args.iter().copied();
    while let Some(arg) = iter.next() {
        if arg == flag {
            return iter.next();
        }
        let prefix = format!("{flag}=");
        if let Some(value) = arg.strip_prefix(&prefix) {
            return Some(value);
        }
    }
    None
}

fn is_rate_limited(output: &std::process::Output) -> bool {
    let stdout = String::from_utf8_lossy(&output.stdout).to_lowercase();
    let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
    let combined = format!("{} {}", stdout, stderr);

    let patterns = [
        "rate limit",
        "rate-limit",
        "ratelimit",
        "hit your limit",
        "you've hit your limit",
        "reached your limit",
        "limit reached",
        "message limit",
        "usage exhausted",
        "quota exceeded",
        "insufficient credits",
        "credit balance is too low",
        "out of credits",
        "payment required",
        "billing",
        "too many requests",
        "429",
        "try again later",
        "try again after",
        "resets ",
        "reset every five hours",
        "usage limit",
        "api limit",
    ];

    patterns.iter().any(|p| combined.contains(p))
}

fn is_environment_failure(output: &std::process::Output) -> bool {
    let stdout = String::from_utf8_lossy(&output.stdout).to_lowercase();
    let stderr = String::from_utf8_lossy(&output.stderr).to_lowercase();
    let combined = format!("{} {}", stdout, stderr);

    let patterns = [
        "no authentication information found",
        "to authenticate",
        "failed to connect to websocket",
        "stream disconnected before completion",
        "operation not permitted",
        "error sending request for url",
        "network is unreachable",
        "connection refused",
        "connection timed out",
        "dns",
        "temporary failure in name resolution",
        "could not resolve host",
    ];

    patterns.iter().any(|p| combined.contains(p))
}

fn classify_process_output(output: &std::process::Output) -> OutcomeClassification {
    if output.status.success() {
        OutcomeClassification::Ok
    } else if is_rate_limited(output) {
        OutcomeClassification::RateLimited
    } else if is_environment_failure(output) {
        OutcomeClassification::EnvironmentError
    } else {
        OutcomeClassification::Failed
    }
}

fn truncate_line(input: &str, max_chars: usize) -> String {
    let mut line = input.lines().next().unwrap_or("").trim().to_string();
    if line.len() > max_chars {
        line.truncate(max_chars);
        line.push_str("…");
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;

    fn make_output(stdout: &str, stderr: &str) -> std::process::Output {
        std::process::Output {
            status: std::process::ExitStatus::from_raw(256),
            stdout: stdout.as_bytes().to_vec(),
            stderr: stderr.as_bytes().to_vec(),
        }
    }

    #[test]
    fn test_rate_limit_detection() {
        let output = make_output("You've hit your limit", "");
        assert!(is_rate_limited(&output));
    }

    #[test]
    fn test_rate_limit_detection_for_claude_usage_limit_wording() {
        let output = make_output(
            "",
            "Message limit reached. Your usage limit will reset every five hours.",
        );
        assert!(is_rate_limited(&output));
    }

    #[test]
    fn test_rate_limit_detection_for_credit_balance_wording() {
        let output = make_output("", "Credit balance is too low. Try again after topping up.");
        assert!(is_rate_limited(&output));
    }

    #[test]
    fn test_environment_failure_detection_for_missing_auth() {
        let output = make_output("", "Error: No authentication information found.");
        assert!(is_environment_failure(&output));
        assert_eq!(
            classify_process_output(&output),
            OutcomeClassification::EnvironmentError
        );
    }

    #[test]
    fn test_environment_failure_detection_for_network_block() {
        let output = make_output(
            "",
            "failed to connect to websocket: IO error: Operation not permitted (os error 1)",
        );
        assert!(is_environment_failure(&output));
        assert_eq!(
            classify_process_output(&output),
            OutcomeClassification::EnvironmentError
        );
    }

    #[test]
    fn test_find_flag_value_supports_split_and_equals_forms() {
        let args = ["--config-dir", "/tmp/copilot"];
        assert_eq!(find_flag_value(&args, "--config-dir"), Some("/tmp/copilot"));

        let args = ["--config-dir=/tmp/copilot"];
        assert_eq!(find_flag_value(&args, "--config-dir"), Some("/tmp/copilot"));
    }

    #[test]
    fn test_resolve_copilot_config_dir_defaults_to_home() {
        let args: [&str; 0] = [];
        assert_eq!(
            resolve_copilot_config_dir(&args, Some("/home/tester")),
            Some("/home/tester/.copilot".to_string())
        );
    }
}
