//! Codex-backed session backend using resumable `codex exec --json` turns.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::Value;

use crate::core::{ProviderCapabilities, ProviderKind, TransportKind};
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

impl SessionBackend for CodexSessionBackend {
    fn name(&self) -> &'static str {
        "codex"
    }

    fn provider_kind(&self) -> ProviderKind {
        ProviderKind::Codex
    }

    fn transport_kind(&self) -> TransportKind {
        TransportKind::CliResumableTurns
    }

    fn capabilities(&self) -> ProviderCapabilities {
        Self::capabilities()
    }

    fn start_session(&self, request: &SessionStartRequest) -> Result<Box<dyn InteractiveSession>> {
        Ok(Box::new(CodexSession::new(
            self.command.clone(),
            request.working_dir.clone(),
            request.writable_roots.clone(),
            request.model.clone(),
            None,
        )))
    }

    fn resume_session(
        &self,
        request: &SessionResumeRequest,
    ) -> Result<Box<dyn InteractiveSession>> {
        Ok(Box::new(CodexSession::new(
            self.command.clone(),
            request.working_dir.clone(),
            request.writable_roots.clone(),
            request.model.clone(),
            Some(request.provider_session_ref.clone()),
        )))
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
    handle_state: Mutex<ProviderSessionHandle>,
    queued_events: Mutex<VecDeque<SessionEvent>>,
}

impl CodexSession {
    fn new(
        command: Option<String>,
        working_dir: PathBuf,
        writable_roots: Vec<PathBuf>,
        model: Option<String>,
        provider_session_ref: Option<String>,
    ) -> Self {
        let status = SessionRuntimeStatus::Ready;
        Self {
            command,
            working_dir,
            writable_roots,
            model,
            handle_state: Mutex::new(ProviderSessionHandle::new(
                ProviderKind::Codex,
                TransportKind::CliResumableTurns,
                provider_session_ref,
                CodexSessionBackend::capabilities(),
                status,
            )),
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
        let mut cmd = if let Some(custom) = self.command.clone() {
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

        let provider_session_ref = self
            .handle_state
            .lock()
            .unwrap()
            .provider_session_ref
            .clone();
        match provider_session_ref.as_deref() {
            Some(session_id) => {
                cmd.arg("exec").arg("resume").arg(session_id);
            }
            None => {
                cmd.arg("exec");
            }
        }

        cmd.arg("--json")
            .arg("--skip-git-repo-check")
            .arg("-C")
            .arg(&self.working_dir)
            .arg("-c")
            .arg("approval_policy=\"never\"")
            .arg("-c")
            .arg("sandbox_mode=\"workspace-write\"");

        for root in dedup_roots(&self.working_dir, &self.writable_roots) {
            if root != self.working_dir {
                cmd.arg("--add-dir").arg(root);
            }
        }

        if let Some(model) = self.model.as_deref() {
            cmd.arg("--model").arg(model);
        }

        cmd.arg(message);
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

impl InteractiveSession for CodexSession {
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
}
