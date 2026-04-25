# Servling Compat Profile — Migration Status

Profile reference:
`brood/_AGENTS/ARCHIVE/SERVLING_COMPATIBILITY_PROFILE.md`

## Phase 1 — seam-level session identity — ✅ partial (additive)

**Landed:** `BackendSessionId` newtype in `profile_extensions::BackendSessionId`.
Re-exported at crate root.

**Not yet landed:** adding `backend_session_id: Option<BackendSessionId>` as
a field on `ProviderSessionHandle`. This would touch every existing backend
that constructs a handle. Deferred to V1.5 to avoid rippling changes through
claude/codex/copilot/cursor wrappers during Phase 2 of the workroach_agent
campaign.

**What this unblocks:** `workroach_agent_servling` can carry a
`BackendSessionId` in its own handle type when it lands in Phase 6 — it
doesn't need to patch every existing backend.

## Phase 2 — enriched session listings — 🟡 deferred to V1.5

Current `ProviderSessionListing` has `provider_session_ref`, `working_dir`,
`title`, `updated_at`. Profile §C asks for `backend_session_id`,
`provider_hint`, `model_hint`, `resumable`. Deferred.

## Phase 3 — unified open semantics — 🟡 deferred to V1.5

Profile §B asks for `SessionOpenRequest { New(...) | Resume { backend_session_id } }`.
Current `start_session` / `resume_session` stay. Deferred.

## Phase 4 — typed error families — ✅ additive

**Landed:** `SessionError` enum in `profile_extensions::SessionError` with
Profile §H's six categories (UnsupportedCapability, InvalidSessionId,
InvalidStateTransition, ProviderTransport, PersistenceFailure,
UserCancelled) + `Other` catch-all. Re-exported at crate root.

Existing `SessionTransportError` (string-shaped) remains for backward
compatibility. New code that wants typed categorisation returns `SessionError`.

## Phase 5 — `workroach_agent_servling` impl — scheduled for campaign Phase 6

The additive Phase 1 + 4 surface is sufficient for
`workroach_agent_servling` to land in campaign Phase 6 (per
`brood/_AGENTS/WORKROACH_AGENT_IMPL_PLAN.md` §5). Phases 2 + 3 can come
later without blocking Phase 6 integration.

## Why additive-only for now

The original plan (WORKROACH_AGENT_IMPL_PLAN.md §4 Phase 2 Track 2D)
envisioned Phases 1-4 as non-breaking additive shims landing in servling
during workroach_agent Phase 2. The "additive shim" interpretation chosen
here — add new types alongside existing ones, don't patch existing
structs — preserves that principle while minimizing ripple. Full migration
of `ProviderSessionHandle` + `ProviderSessionListing` + unified open is
V1.5 work, best done against real consumer needs (workroach_agent_servling
in production use) rather than speculatively.
