//! Cursor Agent session backend: `agent create-chat` plus resumable `--print` turns.

use std::collections::{HashSet, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
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

pub struct CursorSessionBackend {
    command: Option<String>,
}

impl CursorSessionBackend {
    pub fn new(command: Option<String>) -> Self {
        Self { command }
    }

    pub fn check_available() -> Result<()> {
        crate::cursor_agent::CursorAgent::check_available()
    }

    fn capabilities() -> ProviderCapabilities {
        ProviderCapabilities::session_turns_with_resume()
    }
}

impl Backend for CursorSessionBackend {
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            name: "cursor",
            provider_kind: ProviderKind::Cursor,
            transport_kind: TransportKind::CliResumableTurns,
            capabilities: Self::capabilities(),
        }
    }
}

impl SessionBackend for CursorSessionBackend {
    fn start_session(&self, request: &SessionStartRequest) -> Result<Box<dyn InteractiveSession>> {
        let chat_id = run_create_chat(self.command.as_deref())?;
        Ok(Box::new(CursorSession::new(
            self.command.clone(),
            request.working_dir.clone(),
            request.writable_roots.clone(),
            request.model.clone(),
            request.reasoning_effort.clone(),
            Some(chat_id),
        )))
    }

    fn resume_session(
        &self,
        request: &SessionResumeRequest,
    ) -> Result<Box<dyn InteractiveSession>> {
        Ok(Box::new(CursorSession::new(
            self.command.clone(),
            request.working_dir.clone(),
            request.writable_roots.clone(),
            request.model.clone(),
            request.reasoning_effort.clone(),
            Some(request.provider_session_ref.clone()),
        )))
    }

    fn list_sessions(&self) -> Result<Vec<ProviderSessionListing>> {
        list_cursor_chats_on_disk()
    }
}

struct CursorSession {
    command: Option<String>,
    working_dir: PathBuf,
    #[allow(dead_code)]
    writable_roots: Vec<PathBuf>,
    model: Option<String>,
    reasoning_effort: Option<String>,
    handle_state: Mutex<ProviderSessionHandle>,
    queued_events: Mutex<VecDeque<SessionEvent>>,
}

impl CursorSession {
    fn new(
        command: Option<String>,
        working_dir: PathBuf,
        writable_roots: Vec<PathBuf>,
        model: Option<String>,
        reasoning_effort: Option<String>,
        provider_session_ref: Option<String>,
    ) -> Self {
        let status = SessionRuntimeStatus::Ready;
        let mut events = VecDeque::new();
        if let Some(ref r) = provider_session_ref {
            events.push_back(SessionEvent::SessionStarted {
                provider_session_ref: Some(r.clone()),
            });
        }
        Self {
            command,
            working_dir,
            writable_roots,
            model,
            reasoning_effort,
            handle_state: Mutex::new(ProviderSessionHandle::new(
                ProviderKind::Cursor,
                TransportKind::CliResumableTurns,
                provider_session_ref,
                CursorSessionBackend::capabilities(),
                status,
            )),
            queued_events: Mutex::new(events),
        }
    }

    fn run_turn(&self, message: &str) -> Result<SessionStopReason> {
        self.push_event(SessionEvent::StatusChanged {
            status: SessionRuntimeStatus::Starting,
        });
        self.push_event(SessionEvent::StatusChanged {
            status: SessionRuntimeStatus::Running,
        });
        self.set_status(SessionRuntimeStatus::Running);

        let session_ref = self
            .handle_state
            .lock()
            .unwrap()
            .provider_session_ref
            .clone()
            .context("cursor session missing provider_session_ref")?;

        let output = self.execute_cursor_turn(&session_ref, message)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let message = if stderr.is_empty() {
                format!(
                    "Cursor Agent session turn failed with exit {:?}",
                    output.status.code()
                )
            } else {
                format!(
                    "Cursor Agent session turn failed with exit {:?}: {}",
                    output.status.code(),
                    stderr
                )
            };
            let error = SessionTransportError::new(message.clone());
            self.push_event(SessionEvent::Error {
                error: error.clone(),
            });
            self.set_status(SessionRuntimeStatus::Failed);
            bail!(message);
        }

        let stop_reason = self.ingest_output(&output.stdout)?;

        self.push_event(SessionEvent::StatusChanged {
            status: SessionRuntimeStatus::Ready,
        });
        self.set_status(SessionRuntimeStatus::Ready);
        Ok(stop_reason)
    }

    fn execute_cursor_turn(
        &self,
        session_ref: &str,
        message: &str,
    ) -> Result<std::process::Output> {
        let mut cmd = build_turn_command(
            self.command.as_deref(),
            session_ref,
            &self.working_dir,
            self.model.as_deref(),
            self.reasoning_effort.as_deref(),
        )?;

        let mission_dir_abs =
            std::fs::canonicalize(&self.working_dir).unwrap_or_else(|_| self.working_dir.clone());
        cmd.env("TESAKI_MISSION_DIR", mission_dir_abs);

        cmd.stdin(Stdio::piped());
        let mut child = cmd
            .spawn()
            .context("failed to spawn Cursor Agent session turn")?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(message.as_bytes())
                .context("failed to write prompt to Cursor Agent stdin")?;
        }
        child
            .wait_with_output()
            .context("failed to run Cursor Agent session turn")
    }

    fn ingest_output(&self, stdout: &[u8]) -> Result<SessionStopReason> {
        let text =
            String::from_utf8(stdout.to_vec()).context("Cursor session stdout was not UTF-8")?;
        let mut stop_reason = SessionStopReason::EndTurn;

        for line in text.lines().filter(|line| !line.trim().is_empty()) {
            let value: Value = match serde_json::from_str(line) {
                Ok(value) => value,
                Err(_) => continue,
            };

            match value.get("type").and_then(Value::as_str) {
                Some("system") => {
                    if let Some(sid) = value.get("session_id").and_then(Value::as_str) {
                        let mut handle = self.handle_state.lock().unwrap();
                        if handle.provider_session_ref.as_deref() != Some(sid) {
                            handle.provider_session_ref = Some(sid.to_string());
                            self.push_event(SessionEvent::SessionStarted {
                                provider_session_ref: Some(sid.to_string()),
                            });
                        }
                    }
                }
                Some("result") => {
                    if value.get("subtype").and_then(Value::as_str) == Some("success") {
                        if let Some(sid) = value.get("session_id").and_then(Value::as_str) {
                            let mut handle = self.handle_state.lock().unwrap();
                            if handle.provider_session_ref.as_deref() != Some(sid) {
                                handle.provider_session_ref = Some(sid.to_string());
                                self.push_event(SessionEvent::SessionStarted {
                                    provider_session_ref: Some(sid.to_string()),
                                });
                            }
                        }
                        if let Some(result_text) = value.get("result").and_then(Value::as_str) {
                            let t = result_text.trim().to_string();
                            if !t.is_empty() {
                                self.push_event(SessionEvent::ContentChunk {
                                    kind: SessionContentKind::Assistant,
                                    text: t,
                                });
                            }
                        }
                        stop_reason = SessionStopReason::EndTurn;
                        self.push_event(SessionEvent::TurnCompleted {
                            stop_reason: stop_reason.clone(),
                        });
                    } else if value.get("is_error").and_then(|v| v.as_bool()) == Some(true) {
                        let msg = value
                            .get("result")
                            .and_then(Value::as_str)
                            .or_else(|| value.get("message").and_then(Value::as_str))
                            .unwrap_or("Cursor Agent returned an error result")
                            .to_string();
                        self.push_event(SessionEvent::Error {
                            error: SessionTransportError::new(msg),
                        });
                    }
                }
                Some("error") => {
                    let msg = value
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("Cursor Agent error event")
                        .to_string();
                    self.push_event(SessionEvent::Error {
                        error: SessionTransportError::new(msg),
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

#[async_trait::async_trait]
impl InteractiveSession for CursorSession {
    fn handle(&self) -> ProviderSessionHandle {
        self.handle_state.lock().unwrap().clone()
    }

    fn status(&self) -> SessionRuntimeStatus {
        self.handle_state.lock().unwrap().status.clone()
    }

    async fn send_user_turn(&self, request: &UserTurnRequest) -> Result<SessionStopReason> {
        self.run_turn(&request.message)
    }

    async fn interrupt(&self) -> Result<()> {
        bail!("cursor session backend does not support live interrupt")
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

fn run_create_chat(custom_command: Option<&str>) -> Result<String> {
    let mut cmd = base_command(custom_command)?;
    cmd.arg("create-chat");
    let output = cmd.output().context("failed to run `agent create-chat`")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("`agent create-chat` failed: {stderr}");
    }
    let id = String::from_utf8(output.stdout)
        .context("create-chat stdout was not UTF-8")?
        .trim()
        .to_string();
    if id.is_empty() {
        bail!("`agent create-chat` returned an empty chat id");
    }
    Ok(id)
}

fn base_command(custom_command: Option<&str>) -> Result<Command> {
    if let Some(custom) = custom_command {
        let mut parts = custom.split_whitespace();
        let Some(program) = parts.next() else {
            bail!("Cursor session command is empty");
        };
        let mut command = Command::new(program);
        command.args(parts);
        Ok(command)
    } else {
        Ok(Command::new("agent"))
    }
}

/// Build one `agent` invocation: headless resume turn with stream-json on stdout.
fn build_turn_command(
    custom_command: Option<&str>,
    session_ref: &str,
    working_dir: &Path,
    model: Option<&str>,
    reasoning_effort: Option<&str>,
) -> Result<Command> {
    let _ = reasoning_effort;
    let mut cmd = base_command(custom_command)?;
    cmd.arg("--print")
        .arg("--trust")
        .arg("--workspace")
        .arg(working_dir)
        .arg("--force")
        .arg("--output-format")
        .arg("stream-json")
        .arg("--stream-partial-output")
        .arg("--resume")
        .arg(session_ref);
    if let Some(m) = model {
        cmd.arg("--model").arg(m);
    }
    Ok(cmd)
}

fn cursor_chats_root() -> PathBuf {
    std::env::var_os("CURSOR_CHATS_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cursor").join("chats"))
        })
        .unwrap_or_else(|| PathBuf::from(".cursor/chats"))
}

/// Best-effort listing by scanning `~/.cursor/chats/<project_hash>/<chat_uuid>/`.
fn list_cursor_chats_on_disk() -> Result<Vec<ProviderSessionListing>> {
    let root = cursor_chats_root();
    if !root.is_dir() {
        return Ok(Vec::new());
    }

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut seen = HashSet::new();
    let mut listings = Vec::new();

    let d1 = match std::fs::read_dir(&root) {
        Ok(d) => d,
        Err(_) => return Ok(Vec::new()),
    };

    for entry in d1.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let d2 = match std::fs::read_dir(&path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        for e2 in d2.flatten() {
            let name = e2.file_name().to_string_lossy().to_string();
            if !looks_like_chat_uuid(&name) || !e2.path().is_dir() {
                continue;
            }
            if seen.insert(name.clone()) {
                listings.push(ProviderSessionListing {
                    provider_session_ref: name,
                    working_dir: cwd.clone(),
                    title: None,
                    updated_at: None,
                });
            }
        }
    }

    Ok(listings)
}

fn looks_like_chat_uuid(name: &str) -> bool {
    let parts: Vec<&str> = name.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let lens = [8usize, 4, 4, 4, 12];
    parts
        .iter()
        .zip(lens.iter())
        .all(|(p, &len)| p.len() == len && p.chars().all(|c| c.is_ascii_hexdigit()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_turn_command_includes_resume_and_workspace() {
        let wd = PathBuf::from("/tmp/ws");
        let cmd = build_turn_command(
            None,
            "cb5aadae-1282-4467-ac09-67078e565027",
            &wd,
            Some("gpt-5"),
            None,
        )
        .unwrap();
        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert!(args.contains(&"--resume".into()));
        assert!(args.contains(&"cb5aadae-1282-4467-ac09-67078e565027".into()));
        assert!(args.contains(&"--workspace".into()));
        assert!(args.contains(&wd.display().to_string()));
        assert!(args.contains(&"--model".into()));
        assert!(args.contains(&"gpt-5".into()));
    }

    #[test]
    fn looks_like_chat_uuid_accepts_cursor_ids() {
        assert!(looks_like_chat_uuid("cb5aadae-1282-4467-ac09-67078e565027"));
        assert!(!looks_like_chat_uuid("not-a-uuid"));
    }

    /// Real end-to-end Cursor Agent probe — `create-chat`, turn, resume.
    ///
    /// Requires: `agent` on PATH and Cursor authentication (`agent login`).
    /// Run with:
    ///   cargo test -p servling real_cursor -- --ignored --nocapture
    #[test]
    #[ignore = "requires live Cursor Agent CLI + Cursor account auth"]
    fn real_cursor_probe_basic_turn_and_resume() {
        use crate::session::{
            SessionBackend, SessionResumeRequest, SessionStartRequest, UserTurnRequest,
        };
        use std::time::Duration;

        let backend = CursorSessionBackend::new(None);
        let working_dir = std::env::temp_dir().join("cursor-probe-servling");
        std::fs::create_dir_all(&working_dir).expect("probe working dir should be creatable");

        let start_req = SessionStartRequest {
            working_dir: working_dir.clone(),
            writable_roots: vec![],
            model: None,
            reasoning_effort: None,
        };
        let session = backend
            .start_session(&start_req)
            .expect("start_session should succeed");
        let stop = session
            .send_user_turn(&UserTurnRequest {
                message: "Reply with exactly: CURSOR_PROBE_OK".to_string(),
            })
            .expect("send_user_turn should succeed");

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
        let stop2 = resumed
            .send_user_turn(&UserTurnRequest {
                message: "Reply with exactly: CURSOR_RESUME_OK".to_string(),
            })
            .expect("resumed send_user_turn should succeed");

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
