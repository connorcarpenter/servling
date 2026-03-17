//! Interactive session lane for session-capable providers.
//!
//! This sits beside the existing batch `TurnRunner` lane. It is intentionally
//! small and capability-driven so `servling` can host real interactive
//! transports without pretending every provider is symmetrical.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;

use crate::core::{ProviderCapabilities, ProviderKind, TransportKind};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionRuntimeStatus {
    Starting,
    Ready,
    Running,
    Interrupting,
    Ended,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStopReason {
    EndTurn,
    MaxTokens,
    MaxTurnRequests,
    Refusal,
    Cancelled,
    Unknown(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionContentKind {
    User,
    Assistant,
    Thought,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionTransportError {
    pub message: String,
}

impl SessionTransportError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SessionTransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SessionTransportError {}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderSessionHandle {
    pub local_handle_id: Uuid,
    pub provider_kind: ProviderKind,
    pub transport_kind: TransportKind,
    pub provider_session_ref: Option<String>,
    pub capabilities: ProviderCapabilities,
    pub status: SessionRuntimeStatus,
}

impl ProviderSessionHandle {
    pub fn new(
        provider_kind: ProviderKind,
        transport_kind: TransportKind,
        provider_session_ref: Option<String>,
        capabilities: ProviderCapabilities,
        status: SessionRuntimeStatus,
    ) -> Self {
        Self {
            local_handle_id: Uuid::new_v4(),
            provider_kind,
            transport_kind,
            provider_session_ref,
            capabilities,
            status,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionStartRequest {
    pub working_dir: PathBuf,
    pub writable_roots: Vec<PathBuf>,
    pub model: Option<String>,
}

impl SessionStartRequest {
    pub fn new(working_dir: impl Into<PathBuf>) -> Self {
        let working_dir = working_dir.into();
        Self {
            writable_roots: vec![working_dir.clone()],
            working_dir,
            model: None,
        }
    }

    pub fn writable_roots(mut self, writable_roots: Vec<PathBuf>) -> Self {
        self.writable_roots = writable_roots;
        self
    }

    pub fn model(mut self, model: impl Into<Option<String>>) -> Self {
        self.model = model.into();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionResumeRequest {
    pub working_dir: PathBuf,
    pub writable_roots: Vec<PathBuf>,
    pub provider_session_ref: String,
    pub model: Option<String>,
}

impl SessionResumeRequest {
    pub fn new(provider_session_ref: impl Into<String>, working_dir: impl Into<PathBuf>) -> Self {
        let working_dir = working_dir.into();
        Self {
            provider_session_ref: provider_session_ref.into(),
            writable_roots: vec![working_dir.clone()],
            working_dir,
            model: None,
        }
    }

    pub fn writable_roots(mut self, writable_roots: Vec<PathBuf>) -> Self {
        self.writable_roots = writable_roots;
        self
    }

    pub fn model(mut self, model: impl Into<Option<String>>) -> Self {
        self.model = model.into();
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderSessionListing {
    pub provider_session_ref: String,
    pub working_dir: PathBuf,
    pub title: Option<String>,
    pub updated_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UserTurnRequest {
    pub message: String,
}

impl UserTurnRequest {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum SessionEvent {
    SessionStarted {
        provider_session_ref: Option<String>,
    },
    ContentChunk {
        kind: SessionContentKind,
        text: String,
    },
    ToolCall {
        tool_name: String,
        call_id: Option<String>,
    },
    ToolCallUpdate {
        call_id: Option<String>,
        state: String,
        detail: String,
    },
    StatusChanged {
        status: SessionRuntimeStatus,
    },
    TurnCompleted {
        stop_reason: SessionStopReason,
    },
    Warning {
        message: String,
    },
    Error {
        error: SessionTransportError,
    },
    SessionEnded,
}

pub trait InteractiveSession: Send + Sync {
    fn handle(&self) -> ProviderSessionHandle;
    fn status(&self) -> SessionRuntimeStatus;
    fn send_user_turn(&self, request: &UserTurnRequest) -> Result<SessionStopReason>;
    fn interrupt(&self) -> Result<()>;
    fn next_event(&self, timeout: Duration) -> Result<Option<SessionEvent>>;
}

pub trait SessionBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn provider_kind(&self) -> ProviderKind;
    fn transport_kind(&self) -> TransportKind;
    fn capabilities(&self) -> ProviderCapabilities;

    fn start_session(&self, request: &SessionStartRequest) -> Result<Box<dyn InteractiveSession>>;

    fn resume_session(
        &self,
        _request: &SessionResumeRequest,
    ) -> Result<Box<dyn InteractiveSession>> {
        bail!("{} does not support session resume", self.name())
    }

    fn list_sessions(&self) -> Result<Vec<ProviderSessionListing>> {
        Ok(Vec::new())
    }
}

pub type SessionBackendBox = Box<dyn SessionBackend>;
