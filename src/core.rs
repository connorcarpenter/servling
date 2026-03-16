//! Core Servling trait and shared data structures.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::claude_agent::ClaudeAgent;
use crate::codex_agent::CodexAgent;
use crate::copilot_agent::CopilotAgent;
use crate::token_usage::TokenUsage;

/// The core trait for any AI agent provider.
pub trait Servling: Send + Sync {
    /// Execute a raw prompt against the LLM and return a standardized response.
    fn execute(&self, request: &LLMRequest) -> Result<LLMResponse>;

    /// The display name of this agent.
    fn name(&self) -> &'static str;

    /// Optional: Describe how to invoke this as a CLI command.
    fn planned_invocation(&self, _request: &LLMRequest) -> Option<RunnerInvocation> {
        None
    }
}

/// Implement Servling for Boxed trait objects to allow delegation.
impl Servling for Box<dyn Servling> {
    fn execute(&self, request: &LLMRequest) -> Result<LLMResponse> {
        (**self).execute(request)
    }

    fn name(&self) -> &'static str {
        (**self).name()
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
    pub writable_roots: Vec<PathBuf>,
    pub model: Option<String>,
    pub max_runtime_seconds: u32,
    pub stream_output: bool,
    /// Optional: If the prompt is already stored in a file.
    pub input_file: Option<PathBuf>,
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

fn is_claude_tier(model: &str) -> bool {
    let lower = model.to_lowercase();
    matches!(lower.as_str(), "haiku" | "sonnet" | "opus") || lower.contains("claude-")
}

/// Build a single Servling backend.
pub fn build_servling(name: &str, command: Option<String>) -> Result<Box<dyn Servling>> {
    match name {
        "claude" => {
            ClaudeAgent::check_available()?;
            Ok(Box::new(ClaudeAgent::new(command, true)))
        }
        "codex" => {
            CodexAgent::check_available()?;
            Ok(Box::new(CodexAgent::new(command)))
        }
        "copilot" => {
            CopilotAgent::check_available()?;
            Ok(Box::new(CopilotAgent::new(command)))
        }
        other => anyhow::bail!("Unknown agent backend: {}", other),
    }
}
