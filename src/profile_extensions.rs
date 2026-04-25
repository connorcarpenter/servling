//! Extensions for Profile Phases 1 + 4.
//!
//! Added to support `workroach_agent_servling` per
//! `brood/_AGENTS/ARCHIVE/SERVLING_COMPATIBILITY_PROFILE.md` and
//! `brood/_AGENTS/WORKROACH_AGENT_IMPL_PLAN.md` Phase 2 Track 2D.
//!
//! **Additive only.** Zero breakage to existing CLI-wrapper backends (claude,
//! codex, copilot, cursor). Those continue to operate on
//! `provider_session_ref`; `backend_session_id` is optional for them and
//! required for backends (like future `workroach_agent_servling`) that own
//! their own durable session identity.
//!
//! ## Profile Phase 1 — backend session identity
//!
//! `BackendSessionId` is a newtype for opaque durable session ids owned by
//! the backend runtime (as opposed to provider-native session refs).
//!
//! ## Profile Phase 4 — typed error families
//!
//! `SessionError` categorises failures (unsupported capability / invalid id /
//! invalid state / provider transport / persistence / user cancellation).
//! Existing `SessionTransportError` stays for now; new code that wants
//! typed categorisation returns `SessionError`.
//!
//! ## Profile Phases 2 + 3 (listings enrichment + unified open) are NOT yet
//! landed here — tracked as V1.5 follow-up. Phase 6 of the
//! `workroach_agent` campaign (implementing `workroach_agent_servling`) can
//! proceed using the additions here plus the existing v1 API surface.

use serde::{Deserialize, Serialize};

/// Opaque durable session identity owned by the backend.
///
/// For CLI-wrapper backends (current), this is typically unused. For backends
/// that own their own session lifecycle (future `workroach_agent`), this is
/// the sovereign id that survives process restarts.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct BackendSessionId(pub String);

impl BackendSessionId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for BackendSessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Typed error family per Profile §H.
///
/// Alongside existing `SessionTransportError` (string-based). Backends that
/// support typed errors return `Result<T, SessionError>`; legacy wrappers can
/// continue returning the anyhow/string shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionError {
    /// Backend does not support the requested capability.
    UnsupportedCapability { detail: String },
    /// Session id supplied was not found / malformed.
    InvalidSessionId { detail: String },
    /// State machine refused the operation in its current state.
    InvalidStateTransition { detail: String },
    /// Provider transport failure (HTTP, SSE, subprocess stream).
    ProviderTransport { detail: String },
    /// Persistence layer error (session store write failure, schema mismatch).
    PersistenceFailure { detail: String },
    /// User cancelled or aborted the operation.
    UserCancelled { detail: String },
    /// Catch-all.
    Other { detail: String },
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedCapability { detail } => write!(f, "unsupported capability: {detail}"),
            Self::InvalidSessionId { detail } => write!(f, "invalid session id: {detail}"),
            Self::InvalidStateTransition { detail } => write!(f, "invalid state transition: {detail}"),
            Self::ProviderTransport { detail } => write!(f, "provider transport error: {detail}"),
            Self::PersistenceFailure { detail } => write!(f, "persistence failure: {detail}"),
            Self::UserCancelled { detail } => write!(f, "user cancelled: {detail}"),
            Self::Other { detail } => write!(f, "other: {detail}"),
        }
    }
}

impl std::error::Error for SessionError {}

impl SessionError {
    pub fn category(&self) -> &'static str {
        match self {
            Self::UnsupportedCapability { .. } => "unsupported_capability",
            Self::InvalidSessionId { .. } => "invalid_session_id",
            Self::InvalidStateTransition { .. } => "invalid_state_transition",
            Self::ProviderTransport { .. } => "provider_transport",
            Self::PersistenceFailure { .. } => "persistence_failure",
            Self::UserCancelled { .. } => "user_cancelled",
            Self::Other { .. } => "other",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_session_id_round_trips() {
        let id = BackendSessionId::new("sess-123");
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, r#""sess-123""#);
        let back: BackendSessionId = serde_json::from_str(&json).unwrap();
        assert_eq!(back.as_str(), "sess-123");
    }

    #[test]
    fn session_error_category_codes() {
        assert_eq!(
            SessionError::UnsupportedCapability {
                detail: "fork".into()
            }
            .category(),
            "unsupported_capability"
        );
        assert_eq!(
            SessionError::InvalidSessionId {
                detail: "sess-?".into()
            }
            .category(),
            "invalid_session_id"
        );
        assert_eq!(
            SessionError::UserCancelled {
                detail: "Ctrl+C".into()
            }
            .category(),
            "user_cancelled"
        );
    }

    #[test]
    fn session_error_serializes_tagged() {
        let err = SessionError::ProviderTransport {
            detail: "503 upstream".into(),
        };
        let json = serde_json::to_string(&err).unwrap();
        assert!(json.contains(r#""kind":"provider_transport""#));
    }
}
