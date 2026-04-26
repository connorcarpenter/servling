//! Codex-backed session backend using resumable `codex exec --json` turns.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::core::{Backend, BackendMetadata, ProviderCapabilities, ProviderKind, TransportKind};
use crate::session::{
    InteractiveSession, ProviderSessionHandle, ProviderSessionListing, SessionBackend,
    SessionContentKind, SessionEvent, SessionResumeRequest, SessionRuntimeStatus,
    SessionStartRequest, SessionStopReason, SessionTransportError, UserTurnRequest,
};

pub struct CodexSessionBackend {
    command: Option<String>,
}

impl CodexSessionBackend {
    pub fn new(command: Option<String>) -> Self {
        Self { command }
    }

    pub fn check_available() -> Result<()> {
        crate::codex_agent::CodexAgent::check_available()
    }

    fn capabilities() -> ProviderCapabilities {
        ProviderCapabilities::session_turns_with_resume()
    }
}

impl Backend for CodexSessionBackend {
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            name: "codex",
            provider_kind: ProviderKind::Codex,
            transport_kind: TransportKind::CliResumableTurns,
            capabilities: Self::capabilities(),
        }
    }
}

impl SessionBackend for CodexSessionBackend {
    fn start_session(&self, request: &SessionStartRequest) -> Result<Box<dyn InteractiveSession>> {
        Ok(Box::new(Arc::new(CodexSession::new(
            self.command.clone(),
            request.working_dir.clone(),
            request.writable_roots.clone(),
            request.model.clone(),
            request.reasoning_effort.clone(),
            None,
        ))))
    }

    fn resume_session(
        &self,
        request: &SessionResumeRequest,
    ) -> Result<Box<dyn InteractiveSession>> {
        Ok(Box::new(Arc::new(CodexSession::new(
            self.command.clone(),
            request.working_dir.clone(),
            request.writable_roots.clone(),
            request.model.clone(),
            request.reasoning_effort.clone(),
            Some(request.provider_session_ref.clone()),
        ))))
    }

    fn list_sessions(&self) -> Result<Vec<ProviderSessionListing>> {
        #[derive(Debug, Deserialize)]
        struct SessionIndexEntry {
            id: String,
            #[serde(default)]
            thread_name: Option<String>,
            #[serde(default)]
            updated_at: Option<String>,
        }

        let index_path = codex_home().join("session_index.jsonl");
        if !index_path.exists() {
            return Ok(Vec::new());
        }

        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let text = std::fs::read_to_string(&index_path)
            .with_context(|| format!("failed to read {}", index_path.display()))?;

        let mut listings = Vec::new();
        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let Ok(entry) = serde_json::from_str::<SessionIndexEntry>(line) else {
                continue;
            };
            listings.push(ProviderSessionListing {
                provider_session_ref: entry.id,
                working_dir: cwd.clone(),
                title: entry.thread_name,
                updated_at: entry.updated_at,
            });
        }
        Ok(listings)
    }
}

struct CodexSession {
    command: Option<String>,
    working_dir: PathBuf,
    writable_roots: Vec<PathBuf>,
    model: Option<String>,
    reasoning_effort: Option<String>,
    handle_state: Mutex<ProviderSessionHandle>,
    queued_events: Mutex<VecDeque<SessionEvent>>,
}

impl CodexSession {
    fn new(
        command: Option<String>,
        working_dir: PathBuf,
        writable_roots: Vec<PathBuf>,
        model: Option<String>,
        reasoning_effort: Option<String>,
        provider_session_ref: Option<String>,
    ) -> Self {
        let status = SessionRuntimeStatus::Ready;
        let root = working_dir.clone();
        Self {
            command,
            working_dir,
            writable_roots,
            model,
            reasoning_effort,
            handle_state: Mutex::new(ProviderSessionHandle::new(
                ProviderKind::Codex,
                TransportKind::CliResumableTurns,
                provider_session_ref,
                CodexSessionBackend::capabilities(),
                status,
            ).with_working_root(root)),
            queued_events: Mutex::new(VecDeque::new()),
        }
    }

    fn run_turn(&self, message: &str) -> Result<SessionStopReason> {
        self.push_event(SessionEvent::StatusChanged {
            status: SessionRuntimeStatus::Running,
        });
        self.push_event(SessionEvent::StatusChanged {
            status: SessionRuntimeStatus::Starting,
        });
        self.set_status(SessionRuntimeStatus::Running);

        let output = self.execute_codex_turn(message)?;
        let stop_reason = self.ingest_output(&output.stdout)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let message = if stderr.is_empty() {
                format!(
                    "Codex session turn failed with exit {:?}",
                    output.status.code()
                )
            } else {
                format!(
                    "Codex session turn failed with exit {:?}: {}",
                    output.status.code(),
                    stderr
                )
            };
            let error = SessionTransportError::new(message);
            self.push_event(SessionEvent::Error {
                error: error.clone(),
            });
            self.set_status(SessionRuntimeStatus::Failed);
            bail!(error.message);
        }

        self.push_event(SessionEvent::StatusChanged {
            status: SessionRuntimeStatus::Ready,
        });
        self.set_status(SessionRuntimeStatus::Ready);
        Ok(stop_reason)
    }

    fn execute_codex_turn(&self, message: &str) -> Result<std::process::Output> {
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
            &self.writable_roots,
            self.model.as_deref(),
            self.reasoning_effort.as_deref(),
            message,
        )?;
        cmd.output().context("failed to run Codex session turn")
    }

    fn ingest_output(&self, stdout: &[u8]) -> Result<SessionStopReason> {
        let text =
            String::from_utf8(stdout.to_vec()).context("Codex session stdout was not UTF-8")?;
        let mut stop_reason = SessionStopReason::EndTurn;

        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let value: Value = match serde_json::from_str(line) {
                Ok(value) => value,
                Err(_) => continue,
            };

            match value.get("type").and_then(Value::as_str) {
                Some("thread.started") => {
                    if let Some(thread_id) = value.get("thread_id").and_then(Value::as_str) {
                        let mut handle = self.handle_state.lock().unwrap();
                        if handle.provider_session_ref.as_deref() != Some(thread_id) {
                            handle.provider_session_ref = Some(thread_id.to_string());
                            self.push_event(SessionEvent::SessionStarted {
                                provider_session_ref: Some(thread_id.to_string()),
                            });
                        }
                    }
                }
                Some("turn.started") => {
                    self.push_event(SessionEvent::StatusChanged {
                        status: SessionRuntimeStatus::Running,
                    });
                }
                Some("item.completed") => {
                    if let Some(item) = value.get("item") {
                        self.maybe_push_item_event(item);
                    }
                }
                Some("turn.completed") => {
                    stop_reason = extract_stop_reason(&value).unwrap_or(SessionStopReason::EndTurn);
                    self.push_event(SessionEvent::TurnCompleted {
                        stop_reason: stop_reason.clone(),
                    });
                }
                Some("error") => {
                    let message = value
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("Codex session error")
                        .to_string();
                    self.push_event(SessionEvent::Error {
                        error: SessionTransportError::new(message),
                    });
                }
                _ => {}
            }
        }

        Ok(stop_reason)
    }

    fn maybe_push_item_event(&self, item: &Value) {
        let Some(item_type) = item.get("type").and_then(Value::as_str) else {
            return;
        };
        match item_type {
            "agent_message" => {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    self.push_event(SessionEvent::ContentChunk {
                        kind: SessionContentKind::Assistant,
                        text: text.to_string(),
                    });
                }
            }
            "reasoning" => {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    self.push_event(SessionEvent::ContentChunk {
                        kind: SessionContentKind::Thought,
                        text: text.to_string(),
                    });
                }
            }
            _ => {}
        }
    }

    fn push_event(&self, event: SessionEvent) {
        self.queued_events.lock().unwrap().push_back(event);
    }

    fn set_status(&self, status: SessionRuntimeStatus) {
        self.handle_state.lock().unwrap().status = status;
    }
}

#[async_trait::async_trait]
impl InteractiveSession for Arc<CodexSession> {
    fn handle(&self) -> ProviderSessionHandle {
        self.handle_state.lock().unwrap().clone()
    }

    fn status(&self) -> SessionRuntimeStatus {
        self.handle_state.lock().unwrap().status.clone()
    }

    async fn send_user_turn(&self, request: &UserTurnRequest) -> Result<SessionStopReason> {
        // Run the synchronous codex child-process invocation on the
        // blocking threadpool so the caller's async executor isn't
        // stalled for the duration of the CLI turn.
        let session = Arc::clone(self);
        let message = request.message.clone();
        blocking::unblock(move || session.run_turn(&message)).await
    }

    async fn interrupt(&self) -> Result<()> {
        bail!("codex session backend does not support live interrupt")
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

fn dedup_roots(working_dir: &Path, writable_roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    roots.push(working_dir.to_path_buf());
    for root in writable_roots {
        if !roots.iter().any(|existing| existing == root) {
            roots.push(root.clone());
        }
    }
    roots
}

fn extract_stop_reason(value: &Value) -> Option<SessionStopReason> {
    let raw = value
        .get("stop_reason")
        .or_else(|| value.get("reason"))
        .and_then(Value::as_str)?;
    Some(match raw {
        "end_turn" => SessionStopReason::EndTurn,
        "max_tokens" => SessionStopReason::MaxTokens,
        "max_turn_requests" => SessionStopReason::MaxTurnRequests,
        "refusal" => SessionStopReason::Refusal,
        "cancelled" => SessionStopReason::Cancelled,
        other => SessionStopReason::Unknown(other.to_string()),
    })
}

fn codex_home() -> PathBuf {
    std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".codex")))
        .unwrap_or_else(|| PathBuf::from(".codex"))
}

/// Build the `codex` command for a single turn.
///
/// Extracted for testability.  The split between new-session and resume-session
/// invocations matters because `codex exec resume` does **not** accept `-C` or
/// `--add-dir` — those flags exist only on the top-level `codex exec` subcommand.
fn build_turn_command(
    custom_command: Option<&str>,
    provider_session_ref: Option<&str>,
    working_dir: &Path,
    writable_roots: &[PathBuf],
    model: Option<&str>,
    reasoning_effort: Option<&str>,
    message: &str,
) -> Result<Command> {
    let mut cmd = if let Some(custom) = custom_command {
        let mut parts = custom.split_whitespace();
        let Some(program) = parts.next() else {
            bail!("Codex session command is empty");
        };
        let mut command = Command::new(program);
        command.args(parts);
        command
    } else {
        Command::new("codex")
    };

    let is_resume = match provider_session_ref {
        Some(session_id) => {
            cmd.arg("exec").arg("resume").arg(session_id);
            true
        }
        None => {
            cmd.arg("exec");
            false
        }
    };

    cmd.arg("--json")
        .arg("--skip-git-repo-check")
        .arg("-c")
        .arg("approval_policy=\"never\"")
        .arg("-c")
        // Match the Codex batch backend: workspace-write trips bubblewrap
        // loopback setup in the outer-launcher environment, so use the
        // non-bwrap mode and rely on the caller's writable-root contract.
        .arg("sandbox_mode=\"danger-full-access\"");

    for arg in crate::core::backend_reasoning_cli_args("codex", reasoning_effort) {
        cmd.arg(arg);
    }

    if !is_resume {
        cmd.arg("-C").arg(working_dir);
        for root in dedup_roots(working_dir, writable_roots) {
            if root != working_dir {
                cmd.arg("--add-dir").arg(root);
            }
        }
    }

    if let Some(model) = model {
        cmd.arg("--model").arg(model);
    }

    cmd.arg(message);
    Ok(cmd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_stop_reason_defaults_known_values() {
        let value = serde_json::json!({"stop_reason":"end_turn"});
        assert!(matches!(
            extract_stop_reason(&value),
            Some(SessionStopReason::EndTurn)
        ));
    }

    #[test]
    fn dedup_roots_keeps_working_dir_first() {
        let working_dir = PathBuf::from("/tmp/work");
        let roots = dedup_roots(
            &working_dir,
            &[working_dir.clone(), PathBuf::from("/tmp/other")],
        );
        assert_eq!(roots[0], working_dir);
        assert_eq!(roots.len(), 2);
    }

    /// New sessions must pass `-C <working_dir>` to give Codex a working root.
    #[test]
    fn new_session_command_includes_working_dir_flag() {
        let working_dir = PathBuf::from("/tmp/ws");
        let cmd = build_turn_command(
            None,
            None, // no session ref → new session
            &working_dir,
            &[],
            None,
            None,
            "hello",
        )
        .unwrap();
        let args: Vec<_> = cmd.get_args().collect();
        let args_str: Vec<_> = args.iter().map(|a| a.to_string_lossy()).collect();
        assert!(
            args_str.contains(&"-C".into()),
            "new session must include -C flag; got: {args_str:?}"
        );
        assert!(
            args_str.contains(&working_dir.display().to_string().into()),
            "new session must include working_dir; got: {args_str:?}"
        );
        // Must NOT contain "resume"
        assert!(
            !args_str.contains(&"resume".into()),
            "new session must not contain 'resume'; got: {args_str:?}"
        );
    }

    /// Resume sessions must NOT pass `-C` — `codex exec resume` rejects that flag.
    #[test]
    fn resume_command_omits_working_dir_flag() {
        let working_dir = PathBuf::from("/tmp/ws");
        let session_id = "019d0c92-30a3-7e33-ae3c-6bf2c8dca253";
        let cmd = build_turn_command(
            None,
            Some(session_id), // resume session
            &working_dir,
            &[PathBuf::from("/tmp/extra")],
            None,
            None,
            "hello",
        )
        .unwrap();
        let args: Vec<_> = cmd.get_args().collect();
        let args_str: Vec<_> = args.iter().map(|a| a.to_string_lossy()).collect();
        assert!(
            !args_str.contains(&"-C".into()),
            "resume must NOT include -C flag; got: {args_str:?}"
        );
        assert!(
            !args_str.contains(&"--add-dir".into()),
            "resume must NOT include --add-dir; got: {args_str:?}"
        );
        assert!(
            args_str.contains(&"resume".into()),
            "resume must include 'resume' subcommand; got: {args_str:?}"
        );
        assert!(
            args_str.contains(&session_id.into()),
            "resume must include session_id; got: {args_str:?}"
        );
    }

    /// Extra writable roots are passed as --add-dir for new sessions only.
    #[test]
    fn new_session_with_extra_roots_includes_add_dir() {
        let working_dir = PathBuf::from("/tmp/ws");
        let extra = PathBuf::from("/tmp/extra");
        let cmd = build_turn_command(
            None,
            None,
            &working_dir,
            std::slice::from_ref(&extra),
            None,
            None,
            "hello",
        )
        .unwrap();
        let args: Vec<_> = cmd.get_args().collect();
        let args_str: Vec<_> = args.iter().map(|a| a.to_string_lossy()).collect();
        assert!(
            args_str.contains(&"--add-dir".into()),
            "new session with extra roots must include --add-dir; got: {args_str:?}"
        );
        assert!(
            args_str.contains(&extra.display().to_string().into()),
            "new session must include extra root path; got: {args_str:?}"
        );
    }

    /// Real end-to-end Codex probe — basic turn + resume.
    ///
    /// Requires: `codex` CLI available in PATH and authenticated via `~/.codex/auth.json`.
    /// Run with:
    ///   cargo test -p servling real_codex -- --ignored --nocapture
    ///
    /// Auth: Codex uses OAuth tokens stored in `~/.codex/auth.json` (not env vars).
    /// No `OPENAI_API_KEY` or `CODEX_API_KEY` env vars are needed.
    #[test]
    #[ignore = "requires live Codex CLI + ~/.codex/auth.json OAuth tokens"]
    fn real_codex_probe_basic_turn_and_resume() {
        use crate::session::{SessionBackend, SessionResumeRequest, SessionStartRequest};
        use std::time::Duration;

        let backend = CodexSessionBackend::new(None);
        let working_dir = std::env::temp_dir().join("codex-probe-servling");
        std::fs::create_dir_all(&working_dir).expect("probe working dir should be creatable");

        // Scenario 1: new session turn
        let start_req = SessionStartRequest {
            working_dir: working_dir.clone(),
            writable_roots: vec![],
            model: None,
            reasoning_effort: None,
        };
        let session = backend
            .start_session(&start_req)
            .expect("start_session should succeed");
        let stop = futures::executor::block_on(session.send_user_turn(
            &crate::session::UserTurnRequest {
                message: "Reply with exactly: PROBE_OK".to_string(),
            },
        ))
        .expect("send_user_turn should succeed");

        // Drain events
        let mut events = Vec::new();
        while let Ok(Some(ev)) = session.next_event(Duration::from_millis(100)) {
            events.push(ev);
        }

        let handle = session.handle();
        let session_ref = handle
            .provider_session_ref
            .clone()
            .expect("should have session ref");
        println!("session_ref: {session_ref}");
        println!("stop_reason: {stop:?}");
        println!("events: {events:?}");

        // Verify turn completed and got content
        let has_content = events.iter().any(|e| {
            matches!(
                e,
                SessionEvent::ContentChunk {
                    kind: SessionContentKind::Assistant,
                    ..
                }
            )
        });
        assert!(has_content, "expected assistant content chunk");
        assert!(matches!(stop, crate::session::SessionStopReason::EndTurn));

        // Scenario 4: resume with the session_ref from above
        let resume_req = SessionResumeRequest {
            working_dir: working_dir.clone(),
            writable_roots: vec![],
            model: None,
            reasoning_effort: None,
            provider_session_ref: session_ref.clone(),
        };
        let resumed = backend
            .resume_session(&resume_req)
            .expect("resume_session should succeed");
        let stop2 = futures::executor::block_on(resumed.send_user_turn(
            &crate::session::UserTurnRequest {
                message: "Reply with exactly: RESUME_OK".to_string(),
            },
        ))
        .expect("resumed send_user_turn should succeed");

        // Drain resume events
        let mut resume_events = Vec::new();
        while let Ok(Some(ev)) = resumed.next_event(Duration::from_millis(100)) {
            resume_events.push(ev);
        }

        println!("resume stop_reason: {stop2:?}");
        println!("resume events: {resume_events:?}");

        let resumed_handle = resumed.handle();
        assert_eq!(
            resumed_handle.provider_session_ref.as_deref(),
            Some(session_ref.as_str()),
            "resume must keep the same session_ref"
        );

        let _ = std::fs::remove_dir_all(working_dir);
    }
}
