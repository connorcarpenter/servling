//! High-level agent orchestration with fallback logic.

use std::sync::Mutex;

use anyhow::{bail, Result};

use crate::backend_policy::{availability_for_request, record_outcome_for_request};
use crate::backend_registry::{
    build_batch_backend, build_session_backend_by_name, default_batch_backend_names,
    find_backend_descriptor,
};
use crate::core::{
    Backend, BackendMetadata, LLMRequest, LLMResponse, OutcomeClassification, ProviderCapabilities,
    ProviderKind, RunnerInvocation, Servling, TransportKind, TurnRunner,
};
use crate::session::SessionBackendBox;

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

    if find_backend_descriptor(&preferred).is_some() {
        push(&preferred, custom_command);
        for name in default_batch_backend_names() {
            push(name, None);
        }
    } else {
        push(&preferred, custom_command);
    }

    candidates
}

/// Format a candidate chain for display (e.g., "claude -> copilot -> codex").
pub fn describe_candidates(candidates: &[AgentCandidate]) -> String {
    candidates
        .iter()
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
            || matches!(classification, OutcomeClassification::EnvironmentError)
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

impl Backend for CodingAgent {
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            name: "composite",
            provider_kind: ProviderKind::Composite,
            transport_kind: TransportKind::CompositeBatchFallback,
            capabilities: ProviderCapabilities::batch_fallback_chain(),
        }
    }
}

impl TurnRunner for CodingAgent {
    fn execute(&self, request: &LLMRequest) -> Result<LLMResponse> {
        let mut failures = Vec::new();

        loop {
            let (idx, backend) = {
                let current = *self.current_index.lock().unwrap();
                if current >= self.backends.len() {
                    bail!(format_failure_chain(&failures));
                }
                (current, &self.backends[current])
            };

            let availability = availability_for_request(request, backend.name());
            if !availability.allowed {
                let reason = availability
                    .reason
                    .unwrap_or_else(|| "backend blocked by policy".to_string());
                failures.push(format!("{}: skipped ({reason})", backend.name()));
                let mut current = self.current_index.lock().unwrap();
                if *current == idx {
                    *current += 1;
                    if *current >= self.backends.len() {
                        bail!(format_failure_chain(&failures));
                    }
                    log::warn!("Skipping backend {}: {}", backend.name(), reason);
                }
                continue;
            }

            match backend.execute(request) {
                Ok(resp) if CodingAgent::should_fallback(resp.classification) => {
                    record_outcome_for_request(request, backend.name(), resp.classification);
                    failures.push(format_response_failure(
                        backend.name(),
                        resp.classification,
                        &resp.text,
                    ));
                    let mut current = self.current_index.lock().unwrap();
                    if *current == idx {
                        *current += 1;
                        if *current >= self.backends.len() {
                            return Ok(resp);
                        }
                        log::warn!(
                            "Backend {} returned {:?}. Falling back to next.",
                            backend.name(),
                            resp.classification
                        );
                    }
                    continue;
                }
                Ok(mut resp) => {
                    record_outcome_for_request(request, backend.name(), resp.classification);
                    if resp.backend_name.is_none() {
                        resp.backend_name = Some(backend.name().to_string());
                    }
                    return Ok(resp);
                }
                Err(err) => {
                    failures.push(format!("{}: {}", backend.name(), err));
                    let mut current = self.current_index.lock().unwrap();
                    if *current == idx {
                        *current += 1;
                        if *current >= self.backends.len() {
                            bail!(format_failure_chain(&failures));
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
        self.backends
            .get(idx)
            .and_then(|b| b.planned_invocation(request))
    }
}

fn format_failure_chain(failures: &[String]) -> String {
    if failures.is_empty() {
        "No backends available in CodingAgent".to_string()
    } else {
        format!(
            "No backends available in CodingAgent: {}",
            failures.join(" | ")
        )
    }
}

fn format_response_failure(
    backend_name: &str,
    classification: OutcomeClassification,
    text: &str,
) -> String {
    let excerpt = select_failure_excerpt(text);
    if excerpt.is_empty() {
        format!("{backend_name}: {classification:?}")
    } else {
        format!("{backend_name}: {classification:?} ({excerpt})")
    }
}

fn select_failure_excerpt(text: &str) -> &str {
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>();
    if lines.is_empty() {
        return "";
    }

    lines
        .iter()
        .copied()
        .max_by_key(|line| {
            let lower = line.to_ascii_lowercase();
            let mut score = 0;
            if lower.contains("forkpty") {
                score += 8;
            }
            if lower.contains("eacces") || lower.contains("permission") || lower.contains("denied")
            {
                score += 7;
            }
            if lower.contains("error") {
                score += 6;
            }
            if lower.contains("failed") {
                score += 5;
            }
            if lower.contains('✗') {
                score += 1;
            }
            score
        })
        .filter(|line| {
            let lower = line.to_ascii_lowercase();
            lower.contains('✗')
                || lower.contains("error")
                || lower.contains("failed")
                || lower.contains("denied")
                || lower.contains("permission")
                || lower.contains("eacces")
                || lower.contains("forkpty")
        })
        .unwrap_or(lines[0])
}

/// Build a CodingAgent (with fallbacks) from a list of candidates.
pub fn build_coding_agent(candidates: Vec<AgentCandidate>) -> Result<Box<dyn Servling>> {
    if candidates.len() == 1 {
        return build_batch_backend(&candidates[0].name, candidates[0].command.clone());
    }

    let mut builder = CodingAgent::builder();
    let mut count = 0;

    for candidate in candidates {
        match build_batch_backend(&candidate.name, candidate.command.clone()) {
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

/// Build a single provider-pinned interactive session backend.
///
/// Selection happens once at session creation time and does not keep fallback
/// state around for the live session.
pub fn build_session_backend(candidates: Vec<AgentCandidate>) -> Result<SessionBackendBox> {
    for candidate in candidates {
        let is_session_candidate = find_backend_descriptor(&candidate.name)
            .map(|descriptor| descriptor.supports_session_lane())
            .unwrap_or(false);
        if is_session_candidate {
            return build_session_backend_by_name(&candidate.name, candidate.command);
        }
    }

    bail!("No interactive session backend available in candidate set")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BatchCapabilities, BatchFallbackPolicy};
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tempfile::tempdir;

    struct StubBackend {
        name: &'static str,
        classification: OutcomeClassification,
        text: &'static str,
        calls: Arc<AtomicUsize>,
    }

    impl Backend for StubBackend {
        fn metadata(&self) -> BackendMetadata {
            BackendMetadata {
                name: self.name,
                provider_kind: ProviderKind::Composite,
                transport_kind: TransportKind::CliBatch,
                capabilities: ProviderCapabilities {
                    batch: Some(BatchCapabilities {
                        fallback: BatchFallbackPolicy::None,
                    }),
                    session: None,
                },
            }
        }
    }

    impl TurnRunner for StubBackend {
        fn execute(&self, _request: &LLMRequest) -> Result<LLMResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(LLMResponse {
                text: self.text.to_string(),
                classification: self.classification,
                backend_name: Some(self.name.to_string()),
                exit_code: Some(if self.classification == OutcomeClassification::Ok {
                    0
                } else {
                    1
                }),
                token_usage: None,
                elapsed_seconds: 0.0,
                stdout_path: None,
                stderr_path: None,
            })
        }
    }

    #[test]
    fn coding_agent_falls_back_when_first_backend_is_rate_limited() {
        let temp = tempdir().expect("temp dir");
        let first_calls = Arc::new(AtomicUsize::new(0));
        let second_calls = Arc::new(AtomicUsize::new(0));

        let agent = CodingAgent::builder()
            .register(Box::new(StubBackend {
                name: "first",
                classification: OutcomeClassification::RateLimited,
                text: "rate limited",
                calls: first_calls.clone(),
            }))
            .register(Box::new(StubBackend {
                name: "second",
                classification: OutcomeClassification::Ok,
                text: "fallback success",
                calls: second_calls.clone(),
            }))
            .build()
            .expect("build coding agent");

        let request = LLMRequest {
            prompt: "hello".into(),
            working_dir: temp.path().to_path_buf(),
            source_writable_roots: vec![],
            runtime_writable_roots: vec![temp.path().to_path_buf()],
            runtime_env: vec![],
            runtime_profile: None,
            model: None,
            max_runtime_seconds: 30,
            stream_output: false,
            input_file: None,
        };

        let response = agent.execute(&request).expect("fallback response");
        assert_eq!(response.classification, OutcomeClassification::Ok);
        assert_eq!(response.text, "fallback success");
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn coding_agent_falls_back_when_first_backend_has_environment_error() {
        let temp = tempdir().expect("temp dir");
        let first_calls = Arc::new(AtomicUsize::new(0));
        let second_calls = Arc::new(AtomicUsize::new(0));

        let agent = CodingAgent::builder()
            .register(Box::new(StubBackend {
                name: "first",
                classification: OutcomeClassification::EnvironmentError,
                text: "auth missing",
                calls: first_calls.clone(),
            }))
            .register(Box::new(StubBackend {
                name: "second",
                classification: OutcomeClassification::Ok,
                text: "fallback success",
                calls: second_calls.clone(),
            }))
            .build()
            .expect("build coding agent");

        let request = LLMRequest {
            prompt: "hello".into(),
            working_dir: temp.path().to_path_buf(),
            source_writable_roots: vec![],
            runtime_writable_roots: vec![temp.path().to_path_buf()],
            runtime_env: vec![],
            runtime_profile: None,
            model: None,
            max_runtime_seconds: 30,
            stream_output: false,
            input_file: None,
        };

        let response = agent.execute(&request).expect("fallback response");
        assert_eq!(response.classification, OutcomeClassification::Ok);
        assert_eq!(response.text, "fallback success");
        assert_eq!(first_calls.load(Ordering::SeqCst), 1);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn coding_agent_skips_provider_disabled_by_policy_file() {
        let temp = tempdir().expect("temp dir");
        fs::write(
            temp.path().join("servling_backend_policy.json"),
            r#"{"backends":{"first":{"disabled":true,"reason":"operator reported exhausted"}}}"#,
        )
        .expect("write policy");

        let first_calls = Arc::new(AtomicUsize::new(0));
        let second_calls = Arc::new(AtomicUsize::new(0));

        let agent = CodingAgent::builder()
            .register(Box::new(StubBackend {
                name: "first",
                classification: OutcomeClassification::Ok,
                text: "should not run",
                calls: first_calls.clone(),
            }))
            .register(Box::new(StubBackend {
                name: "second",
                classification: OutcomeClassification::Ok,
                text: "fallback success",
                calls: second_calls.clone(),
            }))
            .build()
            .expect("build coding agent");

        let request = LLMRequest {
            prompt: "hello".into(),
            working_dir: temp.path().to_path_buf(),
            source_writable_roots: vec![],
            runtime_writable_roots: vec![temp.path().to_path_buf()],
            runtime_env: vec![],
            runtime_profile: None,
            model: None,
            max_runtime_seconds: 30,
            stream_output: false,
            input_file: None,
        };

        let response = agent.execute(&request).expect("response");
        assert_eq!(response.classification, OutcomeClassification::Ok);
        assert_eq!(response.text, "fallback success");
        assert_eq!(first_calls.load(Ordering::SeqCst), 0);
        assert_eq!(second_calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn coding_agent_reports_full_failure_chain_when_all_backends_fail() {
        let temp = tempdir().expect("temp dir");
        fs::write(
            temp.path().join("servling_backend_policy.json"),
            r#"{"backends":{"third":{"disabled":true,"reason":"known exhausted"}}}"#,
        )
        .expect("write policy");

        let agent = CodingAgent::builder()
            .register(Box::new(StubBackend {
                name: "first",
                classification: OutcomeClassification::EnvironmentError,
                text: "auth missing",
                calls: Arc::new(AtomicUsize::new(0)),
            }))
            .register(Box::new(StubBackend {
                name: "second",
                classification: OutcomeClassification::RateLimited,
                text: "rate limited",
                calls: Arc::new(AtomicUsize::new(0)),
            }))
            .register(Box::new(StubBackend {
                name: "third",
                classification: OutcomeClassification::Ok,
                text: "unused",
                calls: Arc::new(AtomicUsize::new(0)),
            }))
            .build()
            .expect("build coding agent");

        let request = LLMRequest {
            prompt: "hello".into(),
            working_dir: temp.path().to_path_buf(),
            source_writable_roots: vec![],
            runtime_writable_roots: vec![temp.path().to_path_buf()],
            runtime_env: vec![],
            runtime_profile: None,
            model: None,
            max_runtime_seconds: 30,
            stream_output: false,
            input_file: None,
        };

        let err = agent
            .execute(&request)
            .expect_err("all backends should fail");
        let message = err.to_string();
        assert!(message.contains("first: EnvironmentError"));
        assert!(message.contains("second: RateLimited"));
        assert!(message.contains("third: skipped (known exhausted)"));
    }

    #[test]
    fn format_response_failure_prefers_meaningful_error_line() {
        let text = "\
● Read prompt\n\
\n\
✗ Reproduce compile failure (shell)\n\
  └ <exited with error: forkpty(3) failed.>\n";
        let rendered =
            format_response_failure("copilot", OutcomeClassification::EnvironmentError, text);
        assert!(rendered.contains("forkpty(3) failed"));
    }
}
