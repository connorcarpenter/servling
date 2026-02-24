//! High-level agent orchestration with fallback logic.

use std::sync::Mutex;

use anyhow::{bail, Result};

use crate::core::{Servling, LLMRequest, LLMResponse, RunnerInvocation, OutcomeClassification, build_servling};

const AI_BACKENDS: [&str; 3] = ["claude", "copilot", "codex"];

#[derive(Debug, Clone)]
pub struct AgentCandidate {
    pub name: String,
    pub command: Option<String>,
}

/// Generate a prioritized list of agent candidates.
pub fn agent_candidates(preferred: &str, custom_command: Option<String>) -> Vec<AgentCandidate> {
    let preferred = preferred.to_lowercase();
    let mut candidates = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let mut push = |name: &str, cmd: Option<String>| {
        if seen.insert(name.to_string()) {
            candidates.push(AgentCandidate {
                name: name.to_string(),
                command: cmd,
            });
        }
    };

    if AI_BACKENDS.contains(&preferred.as_str()) {
        push(&preferred, custom_command);
        for name in AI_BACKENDS {
            push(name, None);
        }
    } else {
        push(&preferred, custom_command);
    }

    candidates
}

/// Format a candidate chain for display (e.g., "claude -> copilot -> codex").
pub fn describe_candidates(candidates: &[AgentCandidate]) -> String {
    candidates.iter()
        .map(|c| c.name.as_str())
        .collect::<Vec<_>>()
        .join(" -> ")
}

/// A high-level agent that orchestrates one or more backends with fallback logic.
pub struct CodingAgent {
    backends: Vec<Box<dyn Servling>>,
    current_index: Mutex<usize>,
}

impl CodingAgent {
    /// Start building a new CodingAgent.
    pub fn builder() -> CodingAgentBuilder {
        CodingAgentBuilder::default()
    }

    fn should_fallback(classification: OutcomeClassification) -> bool {
        classification.should_fallback()
    }
}

/// Builder for Configuring a CodingAgent.
#[derive(Default)]
pub struct CodingAgentBuilder {
    backends: Vec<Box<dyn Servling>>,
}

impl CodingAgentBuilder {
    /// Register a backend. Order of registration defines priority.
    pub fn register(mut self, backend: Box<dyn Servling>) -> Self {
        self.backends.push(backend);
        self
    }

    /// Convenience for registering multiple backends.
    pub fn with_backends(mut self, backends: Vec<Box<dyn Servling>>) -> Self {
        self.backends.extend(backends);
        self
    }

    pub fn build(self) -> Result<CodingAgent> {
        if self.backends.is_empty() {
            bail!("CodingAgent must have at least one backend");
        }
        Ok(CodingAgent {
            backends: self.backends,
            current_index: Mutex::new(0),
        })
    }
}

impl Servling for CodingAgent {
    fn name(&self) -> &'static str {
        let idx = *self.current_index.lock().unwrap();
        self.backends[idx].name()
    }

    fn execute(&self, request: &LLMRequest) -> Result<LLMResponse> {
        loop {
            let (idx, backend) = {
                let current = *self.current_index.lock().unwrap();
                if current >= self.backends.len() {
                    bail!("No backends available in CodingAgent");
                }
                (current, &self.backends[current])
            };

            match backend.execute(request) {
                Ok(resp) if CodingAgent::should_fallback(resp.classification) => {
                    let mut current = self.current_index.lock().unwrap();
                    if *current == idx {
                        *current += 1;
                        if *current >= self.backends.len() {
                            return Ok(resp);
                        }
                        log::warn!("Backend {} rate limited. Falling back to next.", backend.name());
                    }
                    continue;
                }
                Ok(resp) => return Ok(resp),
                Err(err) => {
                    let mut current = self.current_index.lock().unwrap();
                    if *current == idx {
                        *current += 1;
                        if *current >= self.backends.len() {
                            return Err(err);
                        }
                        log::warn!("Backend {} failed: {}. Falling back.", backend.name(), err);
                    }
                    continue;
                }
            }
        }
    }

    fn planned_invocation(&self, request: &LLMRequest) -> Option<RunnerInvocation> {
        let idx = *self.current_index.lock().unwrap();
        self.backends.get(idx).and_then(|b| b.planned_invocation(request))
    }
}

/// Build a CodingAgent (with fallbacks) from a list of candidates.
pub fn build_coding_agent(candidates: Vec<AgentCandidate>) -> Result<Box<dyn Servling>> {
    if candidates.len() == 1 {
        return build_servling(&candidates[0].name, candidates[0].command.clone());
    }

    let mut builder = CodingAgent::builder();
    let mut count = 0;
    
    for candidate in candidates {
        match build_servling(&candidate.name, candidate.command.clone()) {
            Ok(s) => {
                builder = builder.register(s);
                count += 1;
            }
            Err(e) => log::warn!("Agent candidate {} unavailable: {}", candidate.name, e),
        }
    }
    
    if count == 0 {
        anyhow::bail!("No agent candidates available");
    }
    
    Ok(Box::new(builder.build()?))
}
