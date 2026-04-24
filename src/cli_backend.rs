//! Unified CLI-based LLM backend.

use anyhow::Result;
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::core::{LLMRequest, LLMResponse};
use crate::runner::{run_cli_runner, CliRunnerConfig};

/// A unified workhorse for any CLI-based LLM agent.
pub struct CliBackend {
    pub name: &'static str,
    pub command_template: String,
}

/// Return type of [`CliBackend::prepare_temp_files`]: optional input
/// NamedTempFile + its path, and optional output NamedTempFile + its
/// path. Each file is `Some` only when the command template
/// references `{input_file}`/`{mission_dir}` or `{output_file}`
/// respectively; `None` otherwise.
type TempFilePair = (
    Option<tempfile::NamedTempFile>,
    Option<PathBuf>,
    Option<tempfile::NamedTempFile>,
    Option<PathBuf>,
);

impl CliBackend {
    // Distinct CLI-invocation parameters (paths, model, reasoning, etc.)
    // — grouping them into a struct would save no code and hide intent.
    #[allow(clippy::too_many_arguments)]
    pub fn expand_command(
        &self,
        base_cmd: &str,
        working_dir: &Path,
        writable_roots: &[PathBuf],
        input_path: Option<&Path>,
        output_path: Option<&Path>,
        model: Option<&str>,
        reasoning_effort: Option<&str>,
    ) -> String {
        let mut cmd = base_cmd.to_string();

        let provider_args =
            crate::core::backend_reasoning_cli_args(self.name, reasoning_effort).join(" ");
        cmd = cmd.replace("{provider_args}", &provider_args);

        if let Some(m) = model {
            cmd = format!("{} --model {}", cmd, m);
        }

        let mission_dir = input_path.and_then(|p| p.parent()).unwrap_or(working_dir);
        let writable_root = writable_roots
            .first()
            .map(PathBuf::as_path)
            .unwrap_or(working_dir);
        let add_dir_args = writable_roots
            .iter()
            .skip(1)
            .filter(|root| root.as_path() != working_dir)
            .map(|root| format!("--add-dir {}", root.display()))
            .collect::<Vec<_>>()
            .join(" ");
        let claude_settings = build_claude_settings(working_dir, writable_roots);

        cmd.replace(
            "{input_file}",
            &input_path
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        )
        .replace("{mission_dir}", &mission_dir.display().to_string())
        .replace("{working_dir}", &working_dir.display().to_string())
        .replace("{writable_root}", &writable_root.display().to_string())
        .replace("{add_dir_args}", &add_dir_args)
        .replace("{claude_settings}", &claude_settings)
        .replace(
            "{output_file}",
            &output_path
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        )
    }

    pub fn prepare_temp_files(
        &self,
        prompt: &str,
        temp_dir: &Path,
    ) -> Result<TempFilePair> {
        let mut temp_input = None;
        let mut input_path = None;
        if self.command_template.contains("{input_file}")
            || self.command_template.contains("{mission_dir}")
        {
            let mut file = tempfile::NamedTempFile::new_in(temp_dir)?;
            file.write_all(prompt.as_bytes())?;
            input_path = Some(file.path().to_path_buf());
            temp_input = Some(file);
        }

        let mut temp_output = None;
        let mut output_path = None;
        if self.command_template.contains("{output_file}") {
            let file = tempfile::NamedTempFile::new_in(temp_dir)?;
            output_path = Some(file.path().to_path_buf());
            temp_output = Some(file);
        }

        Ok((temp_input, input_path, temp_output, output_path))
    }

    pub fn execute_with_expansion(
        &self,
        request: &LLMRequest,
        extract_error: bool,
        model_expander: Option<fn(&str) -> String>,
    ) -> Result<LLMResponse> {
        let temp_dir = request.preferred_temp_dir();
        std::fs::create_dir_all(&temp_dir)?;
        let (temp_input, input_path, temp_output, output_path) =
            self.prepare_temp_files(&request.prompt, &temp_dir)?;
        let writable_roots = request.writable_roots();

        let model = request.model.as_deref().map(|m| {
            if let Some(expander) = model_expander {
                expander(m)
            } else {
                m.to_string()
            }
        });

        let cmd = self.expand_command(
            &self.command_template,
            &request.working_dir,
            &writable_roots,
            input_path.as_deref().or(request.input_file.as_deref()),
            output_path.as_deref(),
            model.as_deref(),
            request.reasoning_effort.as_deref(),
        );

        let config = CliRunnerConfig {
            working_dir: request.working_dir.clone(),
            max_runtime_seconds: request.max_runtime_seconds,
            stream_output: request.stream_output,
            env: request.runtime_env.clone(),
        };

        let mission_dir = input_path
            .as_deref()
            .or(request.input_file.as_deref())
            .and_then(|p| p.parent())
            .unwrap_or(&request.working_dir);

        let outcome = run_cli_runner(
            &cmd,
            mission_dir,
            &config,
            extract_error,
            if temp_input.is_some() || request.input_file.is_some() {
                None
            } else {
                Some(request.prompt.clone())
            },
            input_path.as_deref().or(request.input_file.as_deref()),
            output_path.as_deref(),
        )?;

        let stdout_text = outcome
            .stdout_path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_default();
        let stderr_text = outcome
            .stderr_path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_default();

        let text = if let Some(out_p) = output_path {
            std::fs::read_to_string(out_p).unwrap_or_default()
        } else if outcome.classification == crate::core::OutcomeClassification::Ok {
            if self.name == "copilot" {
                // Copilot CLI wraps the model's output in rich terminal UI with box-drawing
                // characters, tool-call traces, and status lines.  The actual model response
                // (typically a JSON object) appears as the final paragraph after the last
                // double-newline.  Additionally, ANSI escape sequences and raw control chars
                // sometimes leak into the output and must be stripped before JSON parsing.
                extract_copilot_result_text(&stdout_text)
            } else if self.name == "cursor" {
                // Cursor Agent `--output-format stream-json` matches Claude-style JSONL
                // (starts with a `system` event).  `--output-format json` emits a single
                // `result` object; extract that when stream-json parsing does not apply.
                extract_claude_result_text(&stdout_text)
                    .or_else(|| extract_compact_agent_result_text(&stdout_text))
                    .unwrap_or(stdout_text)
            } else {
                // Claude CLI outputs JSONL streaming events.  Extract the actual
                // text content from the `{"type":"result","result":"..."}` event
                // so downstream parsers receive the clean response, not the event stream.
                extract_claude_result_text(&stdout_text).unwrap_or(stdout_text)
            }
        } else if stdout_text.is_empty() {
            stderr_text
        } else if stderr_text.is_empty() {
            stdout_text
        } else {
            format!("STDOUT:\n{stdout_text}\n\nSTDERR:\n{stderr_text}")
        };

        drop(temp_input);
        drop(temp_output);

        Ok(LLMResponse {
            text,
            exit_code: outcome.exit_code,
            classification: outcome.classification,
            backend_name: Some(self.name.to_string()),
            stdout_path: outcome.stdout_path,
            stderr_path: outcome.stderr_path,
            token_usage: outcome.token_usage,
            elapsed_seconds: outcome.elapsed_seconds,
        })
    }
}

fn build_claude_settings(working_dir: &Path, writable_roots: &[PathBuf]) -> String {
    let additional_directories = writable_roots
        .iter()
        .filter(|root| root.as_path() != working_dir)
        .map(|root| claude_path(root))
        .collect::<Vec<_>>();
    // Sandbox is explicitly disabled: Claude Code's bwrap (bubblewrap) layer
    // requires CAP_NET_ADMIN for loopback namespace creation, which is not
    // available in typical server/CI environments.  The --permission-mode
    // bypassPermissions flag (set in the command template) already covers the
    // approval model; filesystem scope is handled via additionalDirectories.
    json!({
        "sandbox": {
            "enabled": false,
        },
        "permissions": {
            "defaultMode": "acceptEdits",
            "additionalDirectories": additional_directories,
        }
    })
    .to_string()
}

fn claude_path(path: &Path) -> String {
    let rendered = path.display().to_string();
    if rendered.starts_with('/') {
        format!("//{}", rendered.trim_start_matches('/'))
    } else {
        rendered
    }
}

/// Extract the actual text response from Claude CLI's JSONL streaming output.
///
/// Claude CLI emits a sequence of JSONL events.  The final result is in either:
/// - `{"type":"result","subtype":"success","result":"<text>"}` — the canonical path
/// - `{"type":"assistant","message":{"content":[{"type":"text","text":"<text>"}]}}` — fallback
///
/// Cursor Agent `--output-format json` (non-streaming) prints one JSON object with
/// `type: "result"` / `subtype: "success"` and a string `result` field.
fn extract_compact_agent_result_text(stdout: &str) -> Option<String> {
    let line = stdout.trim();
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    if v.get("type").and_then(|t| t.as_str()) == Some("result")
        && v.get("subtype").and_then(|s| s.as_str()) == Some("success")
    {
        return v
            .get("result")
            .and_then(|r| r.as_str())
            .map(|s| s.to_string());
    }
    None
}

/// Returns `None` if the input doesn't look like Claude JSONL (i.e., falls back to
/// treating `stdout_text` as raw text, preserving existing non-Claude behavior).
fn extract_claude_result_text(stdout: &str) -> Option<String> {
    // Only trigger on Claude JSONL: the first line must start with '{"type":"system"'
    let first_line = stdout.lines().next()?;
    if !first_line.contains("\"type\":\"system\"") {
        return None;
    }

    // Try the result event first (most reliable).
    for line in stdout.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if v.get("type").and_then(|t| t.as_str()) == Some("result")
                && v.get("subtype").and_then(|s| s.as_str()) == Some("success")
            {
                if let Some(result) = v.get("result").and_then(|r| r.as_str()) {
                    if !result.is_empty() {
                        return Some(result.to_string());
                    }
                }
            }
        }
    }

    // Fallback: reconstruct text from content_block_delta text_delta events.
    let mut text = String::new();
    for line in stdout.lines() {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if v.get("type").and_then(|t| t.as_str()) == Some("stream_event") {
                let event = v.get("event")?;
                if event.get("type").and_then(|t| t.as_str()) == Some("content_block_delta") {
                    if let Some(delta_text) = event
                        .get("delta")
                        .and_then(|d| d.get("text"))
                        .and_then(|t| t.as_str())
                    {
                        text.push_str(delta_text);
                    }
                }
            }
        }
    }
    if !text.is_empty() {
        Some(text)
    } else {
        None
    }
}

/// Strip ANSI escape sequences and ASCII control characters (U+0000–U+001F, excluding
/// tab/LF/CR) from `text`.  This makes the text safe to embed in JSON strings and
/// removes terminal-specific artefacts that would cause JSON parse failures.
fn strip_control_chars_and_ansi(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut result = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == 0x1b {
            // Skip ANSI/VT escape sequence:  ESC '[' <params> <letter>
            i += 1;
            if i < bytes.len() && bytes[i] == b'[' {
                i += 1;
                while i < bytes.len() && !(bytes[i] >= 0x40 && bytes[i] <= 0x7e) {
                    i += 1;
                }
                if i < bytes.len() {
                    i += 1; // consume the terminating letter
                }
            }
        } else if bytes[i] < 0x20 && bytes[i] != b'\t' && bytes[i] != b'\n' && bytes[i] != b'\r' {
            // Skip raw control characters (null, BEL, BS, etc.)
            i += 1;
        } else {
            // Preserve complete UTF-8 code points
            let ch_start = i;
            i += 1;
            while i < bytes.len() && (bytes[i] & 0xC0) == 0x80 {
                i += 1;
            }
            if let Ok(s) = std::str::from_utf8(&bytes[ch_start..i]) {
                result.push_str(s);
            }
        }
    }
    result
}

/// Extract the actual model response from Copilot CLI's rich terminal output.
///
/// The Copilot CLI decorates its output with status lines (●, │, └ box-drawing chars)
/// that describe file reads and tool calls.  The real response — typically a JSON
/// object — appears as the final paragraph (double-newline separated block) in the
/// output and always starts with `{` or `[`.
///
/// If no such paragraph is found (e.g. Copilot returned only conversational prose),
/// the function returns the full stripped stdout so that the caller's JSON parser can
/// report a sensible error rather than seeing a garbled string with control chars.
fn extract_copilot_result_text(stdout: &str) -> String {
    let stripped = strip_control_chars_and_ansi(stdout);

    // Walk paragraphs from the end, looking for one that starts with a JSON token.
    let paragraphs: Vec<&str> = stripped.split("\n\n").collect();
    if let Some(last_json_para) = paragraphs.iter().rev().find(|para| {
        let t = para.trim();
        !t.is_empty() && (t.starts_with('{') || t.starts_with('['))
    }) {
        return last_json_para.trim().to_string();
    }

    // Fallback: no clean JSON paragraph found — return the stripped full text so
    // the caller can attempt extraction and report a useful parse error.
    stripped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_command_uses_writable_root_placeholders() {
        let backend = CliBackend {
            name: "test",
            command_template: "tool --root {writable_root} {add_dir_args}".to_string(),
        };
        let working_dir = PathBuf::from("/repo/candidate");
        let writable_roots = vec![
            PathBuf::from("/repo/candidate"),
            PathBuf::from("/repo/generated"),
        ];

        let expanded = backend.expand_command(
            &backend.command_template,
            &working_dir,
            &writable_roots,
            None,
            None,
            None,
            None,
        );

        assert_eq!(
            expanded,
            "tool --root /repo/candidate --add-dir /repo/generated"
        );
    }

    #[test]
    fn expand_command_embeds_claude_settings() {
        let backend = CliBackend {
            name: "claude",
            command_template: "claude --settings {claude_settings}".to_string(),
        };
        let working_dir = PathBuf::from("/repo/candidate");
        let writable_roots = vec![
            PathBuf::from("/repo/candidate"),
            PathBuf::from("/repo/generated"),
        ];

        let expanded = backend.expand_command(
            &backend.command_template,
            &working_dir,
            &writable_roots,
            None,
            None,
            None,
            None,
        );

        // Sandbox is intentionally disabled (see `build_claude_settings`
        // comment); the `--permission-mode bypassPermissions` flag in the
        // command template covers the approval model. Filesystem scope is
        // enforced via `additionalDirectories`. Assertions were updated in
        // tandem with the sandbox-disable fix in commit dd621b3.
        assert!(expanded.contains("\"enabled\":false"));
        assert!(expanded.contains("\"defaultMode\":\"acceptEdits\""));
        assert!(expanded.contains("\"additionalDirectories\":[\"//repo/generated\"]"));
    }

    #[test]
    fn expand_command_injects_codex_provider_args_from_request_fields() {
        let backend = CliBackend {
            name: "codex",
            command_template: "codex {provider_args} exec -C {working_dir} -".to_string(),
        };
        let working_dir = PathBuf::from("/repo/candidate");

        let expanded = backend.expand_command(
            &backend.command_template,
            &working_dir,
            std::slice::from_ref(&working_dir),
            None,
            None,
            Some("gpt-5.3-codex"),
            Some("medium"),
        );

        assert!(expanded.contains("--model gpt-5.3-codex"));
        assert!(expanded.contains("model_reasoning_effort=\"medium\""));
        assert!(!expanded.contains("{provider_args}"));
    }

    #[test]
    fn strip_control_chars_removes_ansi_and_null() {
        let input = "\x1b[32mHello\x1b[0m \x00world\nnewline ok\ttab ok";
        let output = strip_control_chars_and_ansi(input);
        assert_eq!(output, "Hello world\nnewline ok\ttab ok");
    }

    #[test]
    fn extract_copilot_result_text_finds_last_json_paragraph() {
        // Simulates the rich terminal UI that Copilot CLI wraps around the response.
        let stdout = "\
\u{25CF} Read .tmp55ptnp\n  \u{2502} ~/path/to/file\n  \u{2514} 136 lines read\n\n\
\u{25CF} Output verdict JSON (shell)\n  \u{2502} cat << 'EOF'\n  \u{2502} {\"verdict\":\"falsifies\",\"reasoning\":\"truncated\u{2026}\"}\n  \u{2514} 7 lines...\n\n\
{\"verdict\":\"falsifies\",\"reasoning\":\"The full reasoning here.\"}\n\n";
        let result = extract_copilot_result_text(stdout);
        assert_eq!(
            result,
            "{\"verdict\":\"falsifies\",\"reasoning\":\"The full reasoning here.\"}"
        );
    }

    #[test]
    fn extract_copilot_result_text_strips_control_chars_from_json() {
        // Simulates ANSI escape code leaking into the JSON value.
        let stdout = "\u{25CF} Header\n\n{\"key\":\"\x1b[32mvalue\x1b[0m\"}\n\n";
        let result = extract_copilot_result_text(stdout);
        assert_eq!(result, "{\"key\":\"value\"}");
    }

    #[test]
    fn extract_copilot_result_text_fallback_when_no_json() {
        let stdout = "\u{25CF} Some UI\n\nJust prose, no JSON here.\n\n";
        let result = extract_copilot_result_text(stdout);
        // Fallback returns the stripped full text
        assert!(result.contains("Just prose, no JSON here."));
        assert!(!result.contains('\x1b'));
    }

    #[test]
    fn extract_compact_agent_result_text_reads_cursor_json_mode() {
        let stdout = r#"{"type":"result","subtype":"success","is_error":false,"result":"\nHi","session_id":"c9596ba2-e8b1-4985-883c-7f22683923e4"}"#;
        let result = extract_compact_agent_result_text(stdout).expect("parsed");
        assert_eq!(result, "\nHi");
    }
}
