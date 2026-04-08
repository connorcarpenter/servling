use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::backend_registry::{default_batch_backend_names, find_backend_descriptor};
use crate::core::{LLMRequest, OutcomeClassification};

const POLICY_FILE_NAME: &str = "servling_backend_policy.json";
const STATE_FILE_NAME: &str = "servling_backend_state.json";
const RATE_LIMIT_COOLDOWN_HOURS: i64 = 6;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendAvailability {
    pub allowed: bool,
    pub reason: Option<String>,
}

impl BackendAvailability {
    fn allowed() -> Self {
        Self {
            allowed: true,
            reason: None,
        }
    }

    fn blocked(reason: impl Into<String>) -> Self {
        Self {
            allowed: false,
            reason: Some(reason.into()),
        }
    }
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct BackendPolicyFile {
    #[serde(default)]
    preferred_backend: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    backends: BTreeMap<String, BackendPolicyEntry>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct BackendPolicyEntry {
    #[serde(default)]
    disabled: bool,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct BackendStateFile {
    #[serde(default)]
    backends: BTreeMap<String, BackendStateEntry>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct BackendStateEntry {
    #[serde(default)]
    cooldown_until: Option<DateTime<Utc>>,
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendSelection {
    pub preferred_backend: String,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
}

pub fn availability_for_request(request: &LLMRequest, backend: &str) -> BackendAvailability {
    if let Some(reason) = policy_disable_reason(request, backend) {
        return BackendAvailability::blocked(reason);
    }

    if let Some(reason) = active_cooldown_reason(request, backend, Utc::now()) {
        return BackendAvailability::blocked(reason);
    }

    BackendAvailability::allowed()
}

/// Read the required model from the `servling_backend_policy.json` in `dir`.
///
/// Returns `Err` with a clear message if the policy file does not exist or
/// does not contain a `"model"` field.  Model selection must be explicit and
/// project-controlled — no silent fallback to user-level configuration.
pub fn required_model_from_dir(dir: &Path) -> Result<String, String> {
    let selection = resolve_backend_selection_from_dir(dir)?;
    selection.model.ok_or_else(|| {
        format!(
            "No \"model\" field in {}. \
             Add an explicit model, e.g.: {{\"model\": \"claude-sonnet-4-5\", ...}}",
            dir.join(POLICY_FILE_NAME).display()
        )
    })
}

pub fn resolve_backend_selection_from_dir(dir: &Path) -> Result<BackendSelection, String> {
    let path = dir.join(POLICY_FILE_NAME);
    let policy = load_policy_file(&path);
    let model = policy.model.clone().ok_or_else(|| {
        format!(
            "No \"model\" field in {}. \
             Add an explicit model, e.g.: {{\"model\": \"claude-sonnet-4-5\", ...}}",
            path.display()
        )
    })?;
    let preferred_backend = select_preferred_backend(&policy, &model, &path)?;
    Ok(BackendSelection {
        model: crate::core::normalize_model(&preferred_backend, Some(model)),
        preferred_backend,
        reasoning_effort: crate::core::normalize_reasoning_effort(policy.reasoning_effort),
    })
}

pub fn record_outcome_for_request(
    request: &LLMRequest,
    backend: &str,
    classification: OutcomeClassification,
) {
    let state_path = state_path_for_request(request);
    let mut state = load_state_file(&state_path);

    match classification {
        OutcomeClassification::RateLimited => {
            state.backends.insert(
                backend.to_string(),
                BackendStateEntry {
                    cooldown_until: Some(Utc::now() + Duration::hours(RATE_LIMIT_COOLDOWN_HOURS)),
                    reason: Some("automatic rate-limit cooldown".to_string()),
                },
            );
            persist_state_file(&state_path, &state);
        }
        OutcomeClassification::Ok => {
            if state.backends.remove(backend).is_some() {
                persist_state_file(&state_path, &state);
            }
        }
        _ => {}
    }
}

fn policy_disable_reason(request: &LLMRequest, backend: &str) -> Option<String> {
    let policy_path = policy_path_for_request(request);
    let policy = load_policy_file(&policy_path);
    let entry = policy.backends.get(backend)?;
    if !entry.disabled {
        return None;
    }

    Some(
        entry
            .reason
            .clone()
            .unwrap_or_else(|| "disabled by backend policy".to_string()),
    )
}

fn active_cooldown_reason(
    request: &LLMRequest,
    backend: &str,
    now: DateTime<Utc>,
) -> Option<String> {
    let state_path = state_path_for_request(request);
    let state = load_state_file(&state_path);
    let entry = state.backends.get(backend)?;
    let cooldown_until = entry.cooldown_until?;
    if cooldown_until <= now {
        return None;
    }

    let reason = entry
        .reason
        .clone()
        .unwrap_or_else(|| "backend temporarily quarantined".to_string());
    Some(format!("{reason} until {}", cooldown_until.to_rfc3339()))
}

fn policy_path_for_request(request: &LLMRequest) -> PathBuf {
    request.preferred_temp_dir().join(POLICY_FILE_NAME)
}

fn state_path_for_request(request: &LLMRequest) -> PathBuf {
    request.preferred_temp_dir().join(STATE_FILE_NAME)
}

fn load_policy_file(path: &Path) -> BackendPolicyFile {
    match fs::read_to_string(path) {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => BackendPolicyFile::default(),
    }
}

fn load_state_file(path: &Path) -> BackendStateFile {
    match fs::read_to_string(path) {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => BackendStateFile::default(),
    }
}

fn persist_state_file(path: &Path, state: &BackendStateFile) {
    let Some(parent) = path.parent() else {
        return;
    };
    if fs::create_dir_all(parent).is_err() {
        return;
    }
    let Ok(serialized) = serde_json::to_string_pretty(state) else {
        return;
    };
    let _ = fs::write(path, serialized);
}

fn select_preferred_backend(
    policy: &BackendPolicyFile,
    model: &str,
    path: &Path,
) -> Result<String, String> {
    let enabled_backends = enabled_backends(policy);
    if enabled_backends.is_empty() {
        return Err(format!(
            "All backends are disabled in {}. Re-enable at least one backend.",
            path.display()
        ));
    }

    if let Some(preferred_backend) = policy.preferred_backend.as_deref() {
        if find_backend_descriptor(preferred_backend).is_none() {
            return Err(format!(
                "Unknown preferred_backend {:?} in {}",
                preferred_backend,
                path.display()
            ));
        }
        if !enabled_backends
            .iter()
            .any(|candidate| candidate == preferred_backend)
        {
            return Err(format!(
                "preferred_backend {:?} is disabled in {}",
                preferred_backend,
                path.display()
            ));
        }
        return Ok(preferred_backend.to_string());
    }

    if enabled_backends.len() == 1 {
        return Ok(enabled_backends[0].clone());
    }

    let inferred = infer_backend_from_model(model, &enabled_backends);
    if inferred.len() == 1 {
        return Ok(inferred[0].clone());
    }

    Err(format!(
        "Backend selection is ambiguous in {}. Enabled backends: {}. \
         Set \"preferred_backend\" explicitly or disable the unused backends.",
        path.display(),
        enabled_backends.join(", ")
    ))
}

fn enabled_backends(policy: &BackendPolicyFile) -> Vec<String> {
    default_batch_backend_names()
        .iter()
        .copied()
        .filter(|backend| {
            !policy
                .backends
                .get(*backend)
                .is_some_and(|entry| entry.disabled)
        })
        .map(str::to_string)
        .collect()
}

fn infer_backend_from_model(model: &str, enabled_backends: &[String]) -> Vec<String> {
    let lower = model.trim().to_lowercase();
    enabled_backends
        .iter()
        .filter(|backend| model_matches_backend(&lower, backend))
        .cloned()
        .collect()
}

fn model_matches_backend(model: &str, backend: &str) -> bool {
    match backend {
        "claude" => matches!(model, "haiku" | "sonnet" | "opus") || model.contains("claude"),
        "codex" => {
            model.contains("codex")
                || model.starts_with("gpt-")
                || model.starts_with("o1")
                || model.starts_with("o3")
                || model.starts_with("o4")
        }
        "copilot" => model.contains("copilot"),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::LLMRequest;
    use tempfile::tempdir;

    fn request_with_runtime_root(runtime_root: PathBuf) -> LLMRequest {
        LLMRequest {
            prompt: "hello".to_string(),
            working_dir: runtime_root.clone(),
            source_writable_roots: vec![],
            runtime_writable_roots: vec![runtime_root],
            runtime_env: vec![],
            runtime_profile: None,
            model: None,
            reasoning_effort: None,
            max_runtime_seconds: 30,
            stream_output: false,
            input_file: None,
            temp_dir_override: None,
        }
    }

    #[test]
    fn policy_file_can_disable_backend() {
        let temp = tempdir().expect("temp dir");
        let request = request_with_runtime_root(temp.path().to_path_buf());
        let policy_path = policy_path_for_request(&request);
        fs::write(
            policy_path,
            r#"{"backends":{"claude":{"disabled":true,"reason":"operator-reported exhausted"}}}"#,
        )
        .expect("write policy");

        let availability = availability_for_request(&request, "claude");
        assert!(!availability.allowed);
        assert_eq!(
            availability.reason.as_deref(),
            Some("operator-reported exhausted")
        );
    }

    #[test]
    fn rate_limited_backend_enters_cooldown() {
        let temp = tempdir().expect("temp dir");
        let request = request_with_runtime_root(temp.path().to_path_buf());

        record_outcome_for_request(&request, "claude", OutcomeClassification::RateLimited);

        let availability = availability_for_request(&request, "claude");
        assert!(!availability.allowed);
        assert!(availability
            .reason
            .as_deref()
            .unwrap_or_default()
            .contains("automatic rate-limit cooldown"));
    }

    #[test]
    fn successful_backend_clears_cooldown() {
        let temp = tempdir().expect("temp dir");
        let request = request_with_runtime_root(temp.path().to_path_buf());

        record_outcome_for_request(&request, "claude", OutcomeClassification::RateLimited);
        record_outcome_for_request(&request, "claude", OutcomeClassification::Ok);

        let availability = availability_for_request(&request, "claude");
        assert!(availability.allowed);
        assert!(availability.reason.is_none());
    }

    #[test]
    fn selection_uses_only_enabled_backend_when_unique() {
        let temp = tempdir().expect("temp dir");
        let policy_path = temp.path().join(POLICY_FILE_NAME);
        fs::write(
            &policy_path,
            r#"{
              "model": "gpt-5.3-codex",
              "reasoning_effort": "medium",
              "backends": {
                "claude": { "disabled": true },
                "copilot": { "disabled": true }
              }
            }"#,
        )
        .expect("write policy");

        let selection = resolve_backend_selection_from_dir(temp.path()).expect("selection");
        assert_eq!(selection.preferred_backend, "codex");
        assert_eq!(selection.model.as_deref(), Some("gpt-5.3-codex"));
        assert_eq!(selection.reasoning_effort.as_deref(), Some("medium"));
    }

    #[test]
    fn selection_respects_explicit_preferred_backend() {
        let temp = tempdir().expect("temp dir");
        let policy_path = temp.path().join(POLICY_FILE_NAME);
        fs::write(
            &policy_path,
            r#"{
              "preferred_backend": "codex",
              "model": "gpt-5.3-codex"
            }"#,
        )
        .expect("write policy");

        let selection = resolve_backend_selection_from_dir(temp.path()).expect("selection");
        assert_eq!(selection.preferred_backend, "codex");
    }

    #[test]
    fn selection_errors_when_multiple_backends_are_enabled_without_hint() {
        let temp = tempdir().expect("temp dir");
        let policy_path = temp.path().join(POLICY_FILE_NAME);
        fs::write(
            &policy_path,
            r#"{
              "model": "custom-model"
            }"#,
        )
        .expect("write policy");

        let err = resolve_backend_selection_from_dir(temp.path()).expect_err("ambiguous");
        assert!(err.contains("ambiguous"), "error: {err}");
    }
}
