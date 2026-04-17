//! Claude Code CLI session backend — resumable turns via `claude --resume`.
//!
//! Each turn runs `claude --print --permission-mode bypassPermissions
//! --output-format stream-json --include-partial-messages [--resume <id>]
//! <message>` as a child process and ingests the JSON-lines output.
//!
//! The session ID is extracted from the `{"type":"system","subtype":"init","session_id":"..."}`
//! event emitted at the start of every turn.  On subsequent turns the same ID
//! is passed via `--resume` so Claude picks up the conversation from where it
//! left off.
//!
//! ## Capabilities
//!
//! - Session resume via `provider_session_ref` (the session ID string).
//! - Structured event stream (`stream-json` output → `SessionEvent`s).
//! - Provider-pinned sessions: the Claude CLI only accesses its own session
//!   storage, so sessions cannot be moved across providers.
//! - No live interrupt (the child process runs to completion per turn).
//!
//! ## Auth
//!
//! Claude CLI reads auth from `~/.claude/` (API key or Claude Pro OAuth).
//! No env vars are required by this backend.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde_json::Value;

use crate::core::{Backend, BackendMetadata, ProviderCapabilities, ProviderKind, TransportKind};
use crate::session::{
    InteractiveSession, ProviderSessionHandle, ProviderSessionListing, SessionBackend,
    SessionContentKind, SessionEvent, SessionResumeRequest, SessionRuntimeStatus,
    SessionStartRequest, SessionStopReason, SessionTransportError, UserTurnRequest,
};

// ─── backend ─────────────────────────────────────────────────────────────────

pub struct ClaudeSessionBackend {
    command: Option<String>,
}

impl ClaudeSessionBackend {
    pub fn new(command: Option<String>) -> Self {
        Self { command }
    }

    pub fn check_available() -> Result<()> {
        crate::claude_agent::ClaudeAgent::check_available()
    }

    fn capabilities() -> ProviderCapabilities {
        ProviderCapabilities::session_turns_with_resume()
    }
}

impl Backend for ClaudeSessionBackend {
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            name: "claude",
            provider_kind: ProviderKind::Claude,
            transport_kind: TransportKind::CliResumableTurns,
            capabilities: Self::capabilities(),
        }
    }
}

impl SessionBackend for ClaudeSessionBackend {
    fn start_session(&self, request: &SessionStartRequest) -> Result<Box<dyn InteractiveSession>> {
        Ok(Box::new(ClaudeSession::new(
            self.command.clone(),
            request.working_dir.clone(),
            request.model.clone(),
            None, // no prior session ref → new conversation
        )))
    }

    fn resume_session(
        &self,
        request: &SessionResumeRequest,
    ) -> Result<Box<dyn InteractiveSession>> {
        Ok(Box::new(ClaudeSession::new(
            self.command.clone(),
            request.working_dir.clone(),
            request.model.clone(),
            Some(request.provider_session_ref.clone()),
        )))
    }

    /// Returns sessions from `claude sessions list --output-format json`.
    ///
    /// Returns an empty list if the subcommand is unavailable or produces no output —
    /// this is a best-effort operation; a missing list never blocks a session start.
    fn list_sessions(&self) -> Result<Vec<ProviderSessionListing>> {
        let output = Command::new("claude")
            .args(["sessions", "list", "--output-format", "json"])
            .output();

        let Ok(out) = output else {
            return Ok(Vec::new());
        };
        if !out.status.success() {
            return Ok(Vec::new());
        }

        let text = String::from_utf8_lossy(&out.stdout);
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let mut listings = Vec::new();

        // Claude outputs a JSON array of session objects or JSON-lines; handle both.
        let trimmed = text.trim();
        if trimmed.starts_with('[') {
            if let Ok(arr) = serde_json::from_str::<Vec<Value>>(trimmed) {
                for entry in arr {
                    if let Some(listing) = parse_session_listing(&entry, &cwd) {
                        listings.push(listing);
                    }
                }
            }
        } else {
            for line in trimmed.lines().filter(|l| !l.trim().is_empty()) {
                if let Ok(value) = serde_json::from_str::<Value>(line) {
                    if let Some(listing) = parse_session_listing(&value, &cwd) {
                        listings.push(listing);
                    }
                }
            }
        }

        Ok(listings)
    }
}

fn parse_session_listing(value: &Value, cwd: &Path) -> Option<ProviderSessionListing> {
    let session_id = value
        .get("session_id")
        .or_else(|| value.get("id"))
        .and_then(Value::as_str)?
        .to_string();
    let title = value
        .get("title")
        .or_else(|| value.get("summary"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let updated_at = value
        .get("updated_at")
        .and_then(Value::as_str)
        .map(str::to_string);
    Some(ProviderSessionListing {
        provider_session_ref: session_id,
        working_dir: cwd.to_path_buf(),
        title,
        updated_at,
    })
}

// ─── session ─────────────────────────────────────────────────────────────────

struct ClaudeSession {
    command: Option<String>,
    working_dir: PathBuf,
    model: Option<String>,
    handle_state: Mutex<ProviderSessionHandle>,
    queued_events: Mutex<VecDeque<SessionEvent>>,
}

impl ClaudeSession {
    fn new(
        command: Option<String>,
        working_dir: PathBuf,
        model: Option<String>,
        provider_session_ref: Option<String>,
    ) -> Self {
        Self {
            command,
            working_dir,
            model,
            handle_state: Mutex::new(ProviderSessionHandle::new(
                ProviderKind::Claude,
                TransportKind::CliResumableTurns,
                provider_session_ref,
                ClaudeSessionBackend::capabilities(),
                SessionRuntimeStatus::Ready,
            )),
            queued_events: Mutex::new(VecDeque::new()),
        }
    }

    fn run_turn(&self, message: &str) -> Result<SessionStopReason> {
        self.push_event(SessionEvent::StatusChanged {
            status: SessionRuntimeStatus::Starting,
        });
        self.set_status(SessionRuntimeStatus::Running);

        let output = self.execute_claude_turn(message)?;
        let stop_reason = self.ingest_output(&output.stdout)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let msg = if stderr.is_empty() {
                format!(
                    "claude session turn failed with exit {:?}",
                    output.status.code()
                )
            } else {
                format!(
                    "claude session turn failed with exit {:?}: {}",
                    output.status.code(),
                    stderr
                )
            };
            let error = SessionTransportError::new(msg.clone());
            self.push_event(SessionEvent::Error {
                error: error.clone(),
            });
            self.set_status(SessionRuntimeStatus::Failed);
            bail!(msg);
        }

        self.push_event(SessionEvent::StatusChanged {
            status: SessionRuntimeStatus::Ready,
        });
        self.set_status(SessionRuntimeStatus::Ready);
        Ok(stop_reason)
    }

    fn execute_claude_turn(&self, message: &str) -> Result<std::process::Output> {
        let provider_session_ref = self
            .handle_state
            .lock()
            .unwrap()
            .provider_session_ref
            .clone();

        let mut cmd = build_turn_command(
            self.command.as_deref(),
            provider_session_ref.as_deref(),
            &self.working_dir,
            self.model.as_deref(),
            message,
        );
        cmd.output().context("failed to run claude session turn")
    }

    fn ingest_output(&self, stdout: &[u8]) -> Result<SessionStopReason> {
        let text =
            String::from_utf8(stdout.to_vec()).context("claude session stdout was not UTF-8")?;
        let mut stop_reason = SessionStopReason::EndTurn;

        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let value: Value = match serde_json::from_str(line) {
                Ok(v) => v,
                Err(_) => continue,
            };

            match value.get("type").and_then(Value::as_str) {
                // init event — extract and store the session ID
                Some("system") => {
                    if value.get("subtype").and_then(Value::as_str) == Some("init") {
                        if let Some(session_id) =
                            value.get("session_id").and_then(Value::as_str)
                        {
                            let mut handle = self.handle_state.lock().unwrap();
                            if handle.provider_session_ref.as_deref() != Some(session_id) {
                                handle.provider_session_ref = Some(session_id.to_string());
                                drop(handle);
                                self.push_event(SessionEvent::SessionStarted {
                                    provider_session_ref: Some(session_id.to_string()),
                                });
                            }
                        }
                        self.push_event(SessionEvent::StatusChanged {
                            status: SessionRuntimeStatus::Running,
                        });
                    }
                }

                // assistant message — extract text content chunks
                Some("assistant") => {
                    if let Some(message) = value.get("message") {
                        if let Some(content) = message.get("content").and_then(Value::as_array) {
                            for block in content {
                                match block.get("type").and_then(Value::as_str) {
                                    Some("text") => {
                                        if let Some(text) =
                                            block.get("text").and_then(Value::as_str)
                                        {
                                            if !text.is_empty() {
                                                self.push_event(SessionEvent::ContentChunk {
                                                    kind: SessionContentKind::Assistant,
                                                    text: text.to_string(),
                                                });
                                            }
                                        }
                                    }
                                    Some("thinking") => {
                                        if let Some(text) =
                                            block.get("thinking").and_then(Value::as_str)
                                        {
                                            if !text.is_empty() {
                                                self.push_event(SessionEvent::ContentChunk {
                                                    kind: SessionContentKind::Thought,
                                                    text: text.to_string(),
                                                });
                                            }
                                        }
                                    }
                                    Some("tool_use") => {
                                        let tool_name = block
                                            .get("name")
                                            .and_then(Value::as_str)
                                            .unwrap_or("unknown_tool")
                                            .to_string();
                                        let call_id = block
                                            .get("id")
                                            .and_then(Value::as_str)
                                            .map(str::to_string);
                                        self.push_event(SessionEvent::ToolCall {
                                            tool_name,
                                            call_id,
                                        });
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }

                // result event — extract session ID (in case init was missed) and stop reason
                Some("result") => {
                    // Update session_id from result if we haven't already seen it from init
                    if let Some(session_id) = value.get("session_id").and_then(Value::as_str) {
                        let mut handle = self.handle_state.lock().unwrap();
                        if handle.provider_session_ref.is_none() {
                            handle.provider_session_ref = Some(session_id.to_string());
                            drop(handle);
                            self.push_event(SessionEvent::SessionStarted {
                                provider_session_ref: Some(session_id.to_string()),
                            });
                        }
                    }

                    let is_error = value
                        .get("is_error")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    let subtype = value.get("subtype").and_then(Value::as_str).unwrap_or("");
                    stop_reason = if is_error || subtype == "error" {
                        SessionStopReason::Unknown("error".to_string())
                    } else {
                        SessionStopReason::EndTurn
                    };
                    self.push_event(SessionEvent::TurnCompleted {
                        stop_reason: stop_reason.clone(),
                    });
                }

                _ => {}
            }
        }

        Ok(stop_reason)
    }

    fn push_event(&self, event: SessionEvent) {
        self.queued_events.lock().unwrap().push_back(event);
    }

    fn set_status(&self, status: SessionRuntimeStatus) {
        self.handle_state.lock().unwrap().status = status;
    }
}

impl InteractiveSession for ClaudeSession {
    fn handle(&self) -> ProviderSessionHandle {
        self.handle_state.lock().unwrap().clone()
    }

    fn status(&self) -> SessionRuntimeStatus {
        self.handle_state.lock().unwrap().status.clone()
    }

    fn send_user_turn(&self, request: &UserTurnRequest) -> Result<SessionStopReason> {
        self.run_turn(&request.message)
    }

    fn interrupt(&self) -> Result<()> {
        bail!("claude session backend does not support live interrupt")
    }

    fn next_event(&self, timeout: Duration) -> Result<Option<SessionEvent>> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(event) = self.queued_events.lock().unwrap().pop_front() {
                return Ok(Some(event));
            }
            if Instant::now() >= deadline {
                return Ok(None);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

// ─── command builder ─────────────────────────────────────────────────────────

/// Build the `claude` command for a single session turn.
///
/// Extracted for testability.
fn build_turn_command(
    custom_command: Option<&str>,
    provider_session_ref: Option<&str>,
    working_dir: &Path,
    model: Option<&str>,
    message: &str,
) -> Command {
    let mut cmd = if let Some(custom) = custom_command {
        let mut parts = custom.split_whitespace();
        let program = parts.next().unwrap_or("claude");
        let mut command = Command::new(program);
        command.args(parts);
        command
    } else {
        Command::new("claude")
    };

    // Always run in the requested working directory.
    cmd.current_dir(working_dir);

    // Resume an existing session if we have a ref; otherwise start a new one.
    if let Some(session_id) = provider_session_ref {
        cmd.arg("--resume").arg(session_id);
    }

    cmd.arg("--print")
        .arg("--permission-mode")
        .arg("bypassPermissions")
        .arg("--output-format")
        .arg("stream-json")
        .arg("--include-partial-messages");

    if let Some(m) = model {
        cmd.arg("--model").arg(m);
    }

    cmd.arg(message);
    cmd
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_session_command_has_print_flag() {
        let working_dir = PathBuf::from("/tmp/ws");
        let cmd = build_turn_command(None, None, &working_dir, None, "hello");
        let args: Vec<_> = cmd.get_args().collect();
        let args_str: Vec<_> = args.iter().map(|a| a.to_string_lossy()).collect();
        assert!(
            args_str.contains(&"--print".into()),
            "command must include --print; got: {args_str:?}"
        );
        assert!(
            args_str.contains(&"stream-json".into()),
            "command must include stream-json; got: {args_str:?}"
        );
        assert!(
            !args_str.contains(&"--resume".into()),
            "new session must not include --resume; got: {args_str:?}"
        );
    }

    #[test]
    fn resume_session_command_has_resume_flag() {
        let working_dir = PathBuf::from("/tmp/ws");
        let session_id = "01234567-abcd-ef01-2345-6789abcdef01";
        let cmd = build_turn_command(None, Some(session_id), &working_dir, None, "hello");
        let args: Vec<_> = cmd.get_args().collect();
        let args_str: Vec<_> = args.iter().map(|a| a.to_string_lossy()).collect();
        assert!(
            args_str.contains(&"--resume".into()),
            "resume must include --resume; got: {args_str:?}"
        );
        assert!(
            args_str.contains(&session_id.into()),
            "resume must include session_id; got: {args_str:?}"
        );
    }

    #[test]
    fn model_flag_is_passed_when_set() {
        let working_dir = PathBuf::from("/tmp/ws");
        let cmd = build_turn_command(
            None,
            None,
            &working_dir,
            Some("claude-sonnet-4-5"),
            "hello",
        );
        let args: Vec<_> = cmd.get_args().collect();
        let args_str: Vec<_> = args.iter().map(|a| a.to_string_lossy()).collect();
        assert!(
            args_str.contains(&"--model".into()),
            "must include --model; got: {args_str:?}"
        );
        assert!(
            args_str.contains(&"claude-sonnet-4-5".into()),
            "must include model name; got: {args_str:?}"
        );
    }

    #[test]
    fn ingest_output_extracts_session_id_from_init() {
        let working_dir = PathBuf::from("/tmp/ws");
        let session = ClaudeSession::new(None, working_dir, None, None);

        let output = r#"{"type":"system","subtype":"init","session_id":"test-session-abc","tools":[],"cwd":"/tmp"}
{"type":"assistant","message":{"content":[{"type":"text","text":"Hello!"}]},"session_id":"test-session-abc"}
{"type":"result","subtype":"success","is_error":false,"result":"Hello!","session_id":"test-session-abc","num_turns":1}
"#;

        let stop_reason = session.ingest_output(output.as_bytes()).unwrap();
        assert!(matches!(stop_reason, SessionStopReason::EndTurn));

        let handle = session.handle();
        assert_eq!(
            handle.provider_session_ref.as_deref(),
            Some("test-session-abc")
        );
    }

    #[test]
    fn ingest_output_emits_content_chunks() {
        let working_dir = PathBuf::from("/tmp/ws");
        let session = ClaudeSession::new(None, working_dir, None, None);

        let output = r#"{"type":"system","subtype":"init","session_id":"sess-1","tools":[],"cwd":"/tmp"}
{"type":"assistant","message":{"content":[{"type":"text","text":"Answer here."}]},"session_id":"sess-1"}
{"type":"result","subtype":"success","is_error":false,"result":"Answer here.","session_id":"sess-1","num_turns":1}
"#;

        session.ingest_output(output.as_bytes()).unwrap();

        let events: Vec<_> = {
            let mut q = session.queued_events.lock().unwrap();
            q.drain(..).collect()
        };

        let content: Vec<_> = events
            .iter()
            .filter(|e| matches!(e, SessionEvent::ContentChunk { kind: SessionContentKind::Assistant, .. }))
            .collect();
        assert_eq!(content.len(), 1);
        if let SessionEvent::ContentChunk { text, .. } = &content[0] {
            assert_eq!(text, "Answer here.");
        }
    }

    #[test]
    fn backend_metadata_is_claude_resumable_turns() {
        let backend = ClaudeSessionBackend::new(None);
        assert_eq!(backend.metadata().provider_kind, ProviderKind::Claude);
        assert_eq!(backend.metadata().transport_kind, TransportKind::CliResumableTurns);
        assert!(backend.metadata().capabilities.supports_resume());
    }

    /// Real end-to-end Claude probe — new turn + resume.
    ///
    /// Requires: `claude` CLI available in PATH and authenticated via ~/.claude.
    /// Run with:
    ///   cargo test -p servling real_claude -- --ignored --nocapture
    #[test]
    #[ignore = "requires live Claude CLI + ~/.claude auth"]
    fn real_claude_probe_basic_turn_and_resume() {
        use crate::session::{SessionBackend, SessionResumeRequest, SessionStartRequest};

        let backend = ClaudeSessionBackend::new(None);
        let working_dir = std::env::temp_dir().join("claude-probe-servling");
        std::fs::create_dir_all(&working_dir).unwrap();

        // New session
        let start_req = SessionStartRequest {
            working_dir: working_dir.clone(),
            writable_roots: vec![],
            model: None,
            reasoning_effort: None,
        };
        let session = backend.start_session(&start_req).unwrap();
        let stop = session
            .send_user_turn(&crate::session::UserTurnRequest {
                message: "Reply with exactly: PROBE_OK".to_string(),
            })
            .unwrap();

        let mut events = Vec::new();
        while let Ok(Some(ev)) = session.next_event(Duration::from_millis(100)) {
            events.push(ev);
        }

        let handle = session.handle();
        let session_ref = handle.provider_session_ref.clone().expect("must have session_ref");
        println!("session_ref: {session_ref}");
        println!("stop: {stop:?}");
        println!("events: {events:#?}");

        let has_content = events.iter().any(|e| {
            matches!(e, SessionEvent::ContentChunk { kind: SessionContentKind::Assistant, .. })
        });
        assert!(has_content, "expected assistant content chunk");
        assert!(matches!(stop, crate::session::SessionStopReason::EndTurn));

        // Resume
        let resume_req = SessionResumeRequest {
            working_dir: working_dir.clone(),
            writable_roots: vec![],
            model: None,
            reasoning_effort: None,
            provider_session_ref: session_ref.clone(),
        };
        let resumed = backend.resume_session(&resume_req).unwrap();
        let stop2 = resumed
            .send_user_turn(&crate::session::UserTurnRequest {
                message: "Reply with exactly: RESUME_OK".to_string(),
            })
            .unwrap();

        let mut resume_events = Vec::new();
        while let Ok(Some(ev)) = resumed.next_event(Duration::from_millis(100)) {
            resume_events.push(ev);
        }

        println!("resume stop: {stop2:?}");
        println!("resume events: {resume_events:#?}");

        let resumed_handle = resumed.handle();
        assert_eq!(
            resumed_handle.provider_session_ref.as_deref(),
            Some(session_ref.as_str()),
            "resume must keep the same session_ref"
        );

        let _ = std::fs::remove_dir_all(working_dir);
    }
}
