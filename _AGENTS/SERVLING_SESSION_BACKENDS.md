# Servling Session Backends

`servling` now exposes two distinct execution lanes:

- Batch lane: `TurnRunner` / `Servling`
- Session lane: `SessionBackend`

This split is intentional.

- Batch mode is still the home of one-shot CLI invocation and provider fallback.
- Session mode is for real interactive transports only.
- Session mode is optional per provider.
- Interactive sessions are provider-pinned once created.

## Current Interactive Backend

The first real session backend is GitHub Copilot CLI ACP mode:

- launch: `copilot --acp`
- transport: stdio JSON-RPC
- session creation: `initialize` + `session/new`
- resume path: `session/load`
- discovery path: `session/list`
- turn execution: `session/prompt`
- interrupt path: `session/cancel`
- streamed output: `session/update`

## Capability Truth

The capability model is intentionally small and product-facing:

- batch mode support
- interactive session support
- resume support
- live steering while running
- operator interrupt
- durable provider session reference
- structured event stream
- tool call event granularity
- batch fallback
- provider pinning for live sessions

## Boundary Reminder

- `servling` owns provider transport semantics.
- `orchlord` owns durable operator truth.
- `workroach` should eventually own live local process/workspace hosting.
- ACP protocol details should stop inside `servling`.
