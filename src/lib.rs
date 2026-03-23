//! Core AI agent trait and CLI engine.

pub mod backend_registry;
pub mod backend_policy;
pub mod claude_agent;
pub mod cli_backend;
pub mod codex_agent;
pub mod codex_session;
pub mod coding_agent;
pub mod copilot_acp;
pub mod copilot_agent;
pub mod core;
pub mod runner;
pub mod session;
pub mod token_usage;

pub use crate::backend_registry::{
    all_backend_descriptors, build_batch_backend, build_session_backend_by_name,
    default_batch_backend_names, default_session_backend_names, find_backend_descriptor,
    BackendDescriptor,
};
pub use crate::backend_policy::{availability_for_request, record_outcome_for_request, BackendAvailability};
pub use crate::core::{
    build_servling, normalize_model, Backend, BackendMetadata, BatchCapabilities,
    BatchFallbackPolicy, LLMRequest, LLMResponse, OutcomeClassification, ProviderCapabilities,
    ProviderKind, RunnerInvocation, Servling, SessionAffinity, SessionCapabilities,
    SessionControlCapabilities, SessionEventCapabilities, SessionResumeKind, TransportKind,
    TurnRunner,
};
pub use claude_agent::ClaudeAgent;
pub use cli_backend::CliBackend;
pub use codex_agent::CodexAgent;
pub use codex_session::CodexSessionBackend;
pub use coding_agent::{
    agent_candidates, build_coding_agent, build_session_backend, describe_candidates,
    AgentCandidate, CodingAgent, CodingAgentBuilder,
};
pub use copilot_acp::CopilotAcpBackend;
pub use copilot_agent::CopilotAgent;
pub use runner::{run_cli_runner, CliRunnerConfig, CliRunnerOutcome};
pub use session::{
    ProviderSessionHandle, SessionBackend, SessionBackendBox, SessionEvent, SessionRuntimeStatus,
    SessionResumeRequest, SessionStartRequest, SessionStopReason, SessionTransportError,
    UserTurnRequest,
};
pub use token_usage::{
    EfficiencyRating, MissionTokenStats, MissionTypeStats, SessionTokenStats, TokenUsage,
};
