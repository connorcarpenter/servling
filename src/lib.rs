//! Core AI agent trait and CLI engine.

pub mod token_usage;
pub mod core;
pub mod cli_backend;
pub mod coding_agent;
pub mod runner;
pub mod claude_agent;
pub mod codex_agent;
pub mod copilot_agent;

pub use token_usage::{TokenUsage, MissionTokenStats, SessionTokenStats, MissionTypeStats, EfficiencyRating};
pub use crate::core::{Servling, LLMRequest, LLMResponse, RunnerInvocation, normalize_model, OutcomeClassification, build_servling};
pub use cli_backend::CliBackend;
pub use coding_agent::{CodingAgent, CodingAgentBuilder, AgentCandidate, agent_candidates, describe_candidates, build_coding_agent};
pub use runner::{run_cli_runner, CliRunnerConfig, CliRunnerOutcome};
pub use claude_agent::ClaudeAgent;
pub use codex_agent::CodexAgent;
pub use copilot_agent::CopilotAgent;
