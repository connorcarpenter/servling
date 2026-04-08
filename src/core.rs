//! Core Servling trait and shared data structures.

use crate::token_usage::TokenUsage;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Claude,
    Codex,
    Copilot,
    Composite,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportKind {
    CliBatch,
    CliResumableTurns,
    CliJsonRpc,
    CompositeBatchFallback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ProviderCapabilities {
    pub batch: Option<BatchCapabilities>,
    pub session: Option<SessionCapabilities>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct BatchCapabilities {
    pub fallback: BatchFallbackPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchFallbackPolicy {
    None,
    OnRateLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SessionCapabilities {
    pub resume: SessionResumeKind,
    pub control: SessionControlCapabilities,
    pub events: SessionEventCapabilities,
    pub affinity: SessionAffinity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionResumeKind {
    None,
    ProviderSessionRef,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SessionControlCapabilities {
    pub live_steering_while_running: bool,
    pub operator_interrupt: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct SessionEventCapabilities {
    pub structured_stream: bool,
    pub tool_call_events: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionAffinity {
    ProviderPinned,
    Portable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BackendMetadata {
    pub name: &'static str,
    pub provider_kind: ProviderKind,
    pub transport_kind: TransportKind,
    pub capabilities: ProviderCapabilities,
}

pub trait Backend {
    fn metadata(&self) -> BackendMetadata;

    fn name(&self) -> &'static str {
        self.metadata().name
    }

    fn provider_kind(&self) -> ProviderKind {
        self.metadata().provider_kind
    }

    fn transport_kind(&self) -> TransportKind {
        self.metadata().transport_kind
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.metadata().capabilities
    }
}

impl ProviderCapabilities {
    pub const fn batch_only() -> Self {
        Self {
            batch: Some(BatchCapabilities {
                fallback: BatchFallbackPolicy::None,
            }),
            session: None,
        }
    }

    pub const fn batch_fallback_chain() -> Self {
        Self {
            batch: Some(BatchCapabilities {
                fallback: BatchFallbackPolicy::OnRateLimit,
            }),
            session: None,
        }
    }

    pub const fn session_only() -> Self {
        Self {
            batch: None,
            session: Some(SessionCapabilities {
                resume: SessionResumeKind::None,
                control: SessionControlCapabilities {
                    live_steering_while_running: false,
                    operator_interrupt: false,
                },
                events: SessionEventCapabilities {
                    structured_stream: false,
                    tool_call_events: false,
                },
                affinity: SessionAffinity::Portable,
            }),
        }
    }

    pub const fn with_resume(mut self) -> Self {
        if let Some(session) = self.session.as_mut() {
            session.resume = SessionResumeKind::ProviderSessionRef;
        }
        self
    }

    pub const fn with_live_steering(mut self) -> Self {
        if let Some(session) = self.session.as_mut() {
            session.control.live_steering_while_running = true;
        }
        self
    }

    pub const fn with_operator_interrupt(mut self) -> Self {
        if let Some(session) = self.session.as_mut() {
            session.control.operator_interrupt = true;
        }
        self
    }

    pub const fn with_durable_provider_session_ref(mut self) -> Self {
        if let Some(session) = self.session.as_mut() {
            session.resume = SessionResumeKind::ProviderSessionRef;
        }
        self
    }

    pub const fn with_structured_event_stream(mut self) -> Self {
        if let Some(session) = self.session.as_mut() {
            session.events.structured_stream = true;
        }
        self
    }

    pub const fn with_tool_call_events(mut self) -> Self {
        if let Some(session) = self.session.as_mut() {
            session.events.tool_call_events = true;
        }
        self
    }

    pub const fn provider_pinned_session(mut self) -> Self {
        if let Some(session) = self.session.as_mut() {
            session.affinity = SessionAffinity::ProviderPinned;
        }
        self
    }

    pub const fn session_turns_with_resume() -> Self {
        Self::session_only()
            .with_resume()
            .with_durable_provider_session_ref()
            .with_structured_event_stream()
            .provider_pinned_session()
    }

    pub const fn session_jsonrpc() -> Self {
        Self::session_turns_with_resume().with_operator_interrupt()
    }

    pub const fn batch_with_fallback() -> Self {
        Self::batch_fallback_chain()
    }

    pub const fn copilot_acp() -> Self {
        Self::session_jsonrpc()
    }

    pub const fn supports_batch_mode(&self) -> bool {
        self.batch.is_some()
    }

    pub const fn supports_batch_fallback(&self) -> bool {
        matches!(
            self.batch,
            Some(BatchCapabilities {
                fallback: BatchFallbackPolicy::OnRateLimit
            })
        )
    }

    pub const fn supports_interactive_session_mode(&self) -> bool {
        self.session.is_some()
    }

    pub const fn supports_resume(&self) -> bool {
        matches!(
            self.session,
            Some(SessionCapabilities {
                resume: SessionResumeKind::ProviderSessionRef,
                ..
            })
        )
    }

    pub const fn supports_live_steering_while_running(&self) -> bool {
        matches!(
            self.session,
            Some(SessionCapabilities {
                control: SessionControlCapabilities {
                    live_steering_while_running: true,
                    ..
                },
                ..
            })
        )
    }

    pub const fn supports_operator_interrupt(&self) -> bool {
        matches!(
            self.session,
            Some(SessionCapabilities {
                control: SessionControlCapabilities {
                    operator_interrupt: true,
                    ..
                },
                ..
            })
        )
    }

    pub const fn supports_durable_provider_session_ref(&self) -> bool {
        self.supports_resume()
    }

    pub const fn supports_structured_event_stream(&self) -> bool {
        matches!(
            self.session,
            Some(SessionCapabilities {
                events: SessionEventCapabilities {
                    structured_stream: true,
                    ..
                },
                ..
            })
        )
    }

    pub const fn supports_tool_call_events(&self) -> bool {
        matches!(
            self.session,
            Some(SessionCapabilities {
                events: SessionEventCapabilities {
                    tool_call_events: true,
                    ..
                },
                ..
            })
        )
    }

    pub const fn session_provider_pinned(&self) -> bool {
        matches!(
            self.session,
            Some(SessionCapabilities {
                affinity: SessionAffinity::ProviderPinned,
                ..
            })
        )
    }
}

/// The batch execution lane for any AI agent provider.
pub trait TurnRunner: Backend + Send + Sync {
    /// Execute a raw prompt against the LLM and return a standardized response.
    fn execute(&self, request: &LLMRequest) -> Result<LLMResponse>;

    /// Optional: Describe how to invoke this as a CLI command.
    fn planned_invocation(&self, _request: &LLMRequest) -> Option<RunnerInvocation> {
        None
    }
}

/// Backwards-compatible alias for the existing batch lane API.
pub trait Servling: TurnRunner {}

impl<T: TurnRunner + ?Sized> Servling for T {}

impl Backend for Box<dyn Servling> {
    fn metadata(&self) -> BackendMetadata {
        (**self).metadata()
    }
}

/// Implement TurnRunner for boxed trait objects to allow delegation.
impl TurnRunner for Box<dyn Servling> {
    fn execute(&self, request: &LLMRequest) -> Result<LLMResponse> {
        (**self).execute(request)
    }

    fn planned_invocation(&self, request: &LLMRequest) -> Option<RunnerInvocation> {
        (**self).planned_invocation(request)
    }
}

/// A standardized request to a Servling.
#[derive(Debug, Clone)]
pub struct LLMRequest {
    pub prompt: String,
    pub working_dir: PathBuf,
    pub source_writable_roots: Vec<PathBuf>,
    pub runtime_writable_roots: Vec<PathBuf>,
    pub runtime_env: Vec<(String, String)>,
    pub runtime_profile: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub max_runtime_seconds: u32,
    pub stream_output: bool,
    /// Optional: If the prompt is already stored in a file.
    pub input_file: Option<PathBuf>,
    /// Optional: Override the directory used for temporary files (e.g., prompt
    /// staging files written by the backend before invoking the CLI).  When set,
    /// takes priority over `runtime_writable_roots` / `source_writable_roots`.
    /// Use `std::env::temp_dir()` to write probe files outside the agent's
    /// `--add-dir` accessible area so safety filters do not fire.
    pub temp_dir_override: Option<PathBuf>,
}

impl LLMRequest {
    pub fn writable_roots(&self) -> Vec<PathBuf> {
        let mut roots = Vec::new();
        for root in self
            .source_writable_roots
            .iter()
            .chain(self.runtime_writable_roots.iter())
        {
            if !roots.iter().any(|existing| existing == root) {
                roots.push(root.clone());
            }
        }
        if roots.is_empty() {
            roots.push(self.working_dir.clone());
        }
        roots
    }

    pub fn preferred_temp_dir(&self) -> PathBuf {
        if let Some(ref override_dir) = self.temp_dir_override {
            return override_dir.clone();
        }
        self.runtime_writable_roots
            .first()
            .cloned()
            .or_else(|| self.source_writable_roots.first().cloned())
            .unwrap_or_else(|| self.working_dir.clone())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum OutcomeClassification {
    Ok,
    Failed,
    Timeout,
    EnvironmentError,
    /// Rate limited by the AI provider (Claude, Codex, etc.)
    RateLimited,
}

impl OutcomeClassification {
    /// Returns true if this classification should trigger an agent fallback.
    pub fn should_fallback(&self) -> bool {
        matches!(self, Self::RateLimited)
    }
}

/// A standardized response from a Servling.
#[derive(Debug, Clone)]
pub struct LLMResponse {
    pub text: String,
    pub classification: OutcomeClassification,
    pub backend_name: Option<String>,
    pub exit_code: Option<i32>,
    pub token_usage: Option<TokenUsage>,
    pub elapsed_seconds: f64,
    pub stdout_path: Option<String>,
    pub stderr_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerInvocation {
    pub program: String,
    pub args: Vec<String>,
    pub working_dir: String,
    pub env: Vec<(String, String)>,
}

/// Helper to normalize a requested model name for a specific backend.
pub fn normalize_model(backend_name: &str, model: Option<String>) -> Option<String> {
    let model = model?;
    // Claude models pass through to Claude backend
    if backend_name == "claude" {
        return Some(model);
    }
    // Generic tiers are stripped for non-claude backends unless they match
    if is_claude_tier(&model) {
        return None;
    }
    Some(model)
}

pub fn normalize_reasoning_effort(reasoning_effort: Option<String>) -> Option<String> {
    reasoning_effort.and_then(|effort| {
        let normalized = effort.trim().to_lowercase();
        if normalized.is_empty() {
            None
        } else {
            Some(normalized)
        }
    })
}

pub fn backend_reasoning_cli_args(
    backend_name: &str,
    reasoning_effort: Option<&str>,
) -> Vec<String> {
    match (backend_name, reasoning_effort) {
        ("codex", Some(effort)) => vec![
            "-c".to_string(),
            format!("model_reasoning_effort=\"{effort}\""),
        ],
        _ => Vec::new(),
    }
}

fn is_claude_tier(model: &str) -> bool {
    let lower = model.to_lowercase();
    matches!(lower.as_str(), "haiku" | "sonnet" | "opus") || lower.contains("claude-")
}

/// Build a single Servling backend.
pub fn build_servling(name: &str, command: Option<String>) -> Result<Box<dyn Servling>> {
    crate::backend_registry::build_batch_backend(name, command)
}
