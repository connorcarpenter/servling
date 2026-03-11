//! Core AI agent trait and CLI engine.

pub mod claude_agent;
pub mod cli_backend;
pub mod codex_agent;
pub mod coding_agent;
pub mod copilot_agent;
pub mod core;
pub mod runner;
pub mod token_usage;

pub use crate::core::{
    build_servling, normalize_model, LLMRequest, LLMResponse, OutcomeClassification,
    RunnerInvocation, Servling,
};
pub use claude_agent::ClaudeAgent;
pub use cli_backend::CliBackend;
pub use codex_agent::CodexAgent;
pub use coding_agent::{
    agent_candidates, build_coding_agent, describe_candidates, AgentCandidate, CodingAgent,
    CodingAgentBuilder,
};
pub use copilot_agent::CopilotAgent;
pub use runner::{run_cli_runner, CliRunnerConfig, CliRunnerOutcome};
pub use token_usage::{
    EfficiencyRating, MissionTokenStats, MissionTypeStats, SessionTokenStats, TokenUsage,
};
