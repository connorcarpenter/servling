use anyhow::{bail, Result};

use crate::claude_agent::ClaudeAgent;
use crate::codex_agent::CodexAgent;
use crate::codex_session::CodexSessionBackend;
use crate::copilot_acp::CopilotAcpBackend;
use crate::copilot_agent::CopilotAgent;
use crate::core::{ProviderKind, Servling};
use crate::cursor_agent::CursorAgent;
use crate::cursor_session::CursorSessionBackend;
use crate::session::SessionBackendBox;

type BatchBuilder = fn(Option<String>) -> Result<Box<dyn Servling>>;
type SessionBuilder = fn(Option<String>) -> Result<SessionBackendBox>;

#[derive(Debug, Clone, Copy)]
pub struct BackendDescriptor {
    pub name: &'static str,
    pub provider_kind: ProviderKind,
    batch_builder: Option<BatchBuilder>,
    session_builder: Option<SessionBuilder>,
}

impl BackendDescriptor {
    pub const fn supports_batch_lane(&self) -> bool {
        self.batch_builder.is_some()
    }

    pub const fn supports_session_lane(&self) -> bool {
        self.session_builder.is_some()
    }

    pub fn build_batch(&self, command: Option<String>) -> Result<Box<dyn Servling>> {
        match self.batch_builder {
            Some(build) => build(command),
            None => bail!("Backend {} does not support the batch lane", self.name),
        }
    }

    pub fn build_session(&self, command: Option<String>) -> Result<SessionBackendBox> {
        match self.session_builder {
            Some(build) => build(command),
            None => bail!("Backend {} does not support the session lane", self.name),
        }
    }
}

fn build_claude_batch(command: Option<String>) -> Result<Box<dyn Servling>> {
    ClaudeAgent::check_available()?;
    Ok(Box::new(ClaudeAgent::new(command, true)))
}

fn build_codex_batch(command: Option<String>) -> Result<Box<dyn Servling>> {
    CodexAgent::check_available()?;
    Ok(Box::new(CodexAgent::new(command)))
}

fn build_codex_session(command: Option<String>) -> Result<SessionBackendBox> {
    CodexSessionBackend::check_available()?;
    Ok(Box::new(CodexSessionBackend::new(command)))
}

fn build_copilot_batch(command: Option<String>) -> Result<Box<dyn Servling>> {
    CopilotAgent::check_available()?;
    Ok(Box::new(CopilotAgent::new(command)))
}

fn build_copilot_session(command: Option<String>) -> Result<SessionBackendBox> {
    CopilotAcpBackend::check_available()?;
    Ok(Box::new(CopilotAcpBackend::new(command)))
}

fn build_cursor_batch(command: Option<String>) -> Result<Box<dyn Servling>> {
    CursorAgent::check_available()?;
    Ok(Box::new(CursorAgent::new(command, true)))
}

fn build_cursor_session(command: Option<String>) -> Result<SessionBackendBox> {
    CursorSessionBackend::check_available()?;
    Ok(Box::new(CursorSessionBackend::new(command)))
}

const BACKENDS: &[BackendDescriptor] = &[
    BackendDescriptor {
        name: "claude",
        provider_kind: ProviderKind::Claude,
        batch_builder: Some(build_claude_batch),
        session_builder: None,
    },
    BackendDescriptor {
        name: "codex",
        provider_kind: ProviderKind::Codex,
        batch_builder: Some(build_codex_batch),
        session_builder: Some(build_codex_session),
    },
    BackendDescriptor {
        name: "copilot",
        provider_kind: ProviderKind::Copilot,
        batch_builder: Some(build_copilot_batch),
        session_builder: Some(build_copilot_session),
    },
    BackendDescriptor {
        name: "cursor",
        provider_kind: ProviderKind::Cursor,
        batch_builder: Some(build_cursor_batch),
        session_builder: Some(build_cursor_session),
    },
];

const DEFAULT_BATCH_BACKENDS: &[&str] = &["claude", "codex", "copilot", "cursor"];
const DEFAULT_SESSION_BACKENDS: &[&str] = &["codex", "copilot", "cursor"];

pub fn all_backend_descriptors() -> &'static [BackendDescriptor] {
    BACKENDS
}

pub fn find_backend_descriptor(name: &str) -> Option<&'static BackendDescriptor> {
    BACKENDS.iter().find(|descriptor| descriptor.name == name)
}

pub fn default_batch_backend_names() -> &'static [&'static str] {
    DEFAULT_BATCH_BACKENDS
}

pub fn default_session_backend_names() -> &'static [&'static str] {
    DEFAULT_SESSION_BACKENDS
}

pub fn build_batch_backend(name: &str, command: Option<String>) -> Result<Box<dyn Servling>> {
    let descriptor = find_backend_descriptor(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown agent backend: {name}"))?;
    descriptor.build_batch(command)
}

pub fn build_session_backend_by_name(
    name: &str,
    command: Option<String>,
) -> Result<SessionBackendBox> {
    let descriptor = find_backend_descriptor(name)
        .ok_or_else(|| anyhow::anyhow!("Unknown agent backend: {name}"))?;
    descriptor.build_session(command)
}

#[cfg(test)]
mod tests {
    use super::{
        all_backend_descriptors, default_batch_backend_names, default_session_backend_names,
        find_backend_descriptor,
    };

    #[test]
    fn backend_registry_exposes_expected_lanes() {
        let codex = find_backend_descriptor("codex").expect("codex descriptor");
        assert!(codex.supports_batch_lane());
        assert!(codex.supports_session_lane());

        let claude = find_backend_descriptor("claude").expect("claude descriptor");
        assert!(claude.supports_batch_lane());
        assert!(!claude.supports_session_lane());

        let cursor = find_backend_descriptor("cursor").expect("cursor descriptor");
        assert!(cursor.supports_batch_lane());
        assert!(cursor.supports_session_lane());
    }

    #[test]
    fn default_lane_orders_are_explicit() {
        assert_eq!(
            default_batch_backend_names(),
            &["claude", "codex", "copilot", "cursor"]
        );
        assert_eq!(
            default_session_backend_names(),
            &["codex", "copilot", "cursor"]
        );
        assert_eq!(all_backend_descriptors().len(), 4);
    }
}
