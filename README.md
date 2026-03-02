# 🤖 servling

> **The lightweight core AI agent trait and CLI engine.**

`servling` is a Rust crate providing a standardized, resilient interface for AI agents that run as CLI tools. It's the "little servant" (servling) that handles the messy work of interacting with LLM-powered command-line interfaces.

Built originally as the core engine for high-reliability agent tasks, it manages **streaming**, **timeouts**, **token usage tracking**, and **automatic fallback logic**.

---

## ✨ Features

- **Standardized Trait**: A single `Servling` trait to rule them all. Whether it's Claude, Copilot, or Codex, the interface is the same.
- **Resilient Fallbacks**: Automatic "chain-of-command" logic. If `Claude` is rate-limited, `servling` can automatically fall back to `Copilot` or `Codex` without missing a beat.
- **Observability**: Built-in `stderr` parsing for real-time token usage, cost estimation (USD), and efficiency ratings.
- **Live Streaming**: Full support for real-time output streaming from underlying CLI processes.
- **Mission Control**: Handles standard mission/task structures, timeouts, and outcome classifications (Ok, Failed, Timeout, RateLimited).

---

## 🏗️ Supported Backends

| Backend | CLI Tool |
| :--- | :--- |
| **Claude** | [Claude Code](https://github.com/anthropics/claude-code) |
| **Copilot** | [GitHub Copilot CLI](https://github.com/github/gh-copilot) |
| **Codex** | OpenAI Codex / Generic CLI Wrappers |

---

## 🚀 Getting Started

Add `servling` to your `Cargo.toml`:

```toml
[dependencies]
servling = { path = "../servling" }
```

### Basic Usage

```rust
use servling::{Servling, LLMRequest, build_servling};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    // 1. Build a backend (or a chain of backends!)
    let agent = build_servling("claude", None)?;

    // 2. Prepare a request
    let request = LLMRequest {
        prompt: "Refactor this function for better performance.".to_string(),
        working_dir: PathBuf::from("."),
        model: Some("claude-3-5-sonnet".to_string()),
        max_runtime_seconds: 300,
        stream_output: true,
        input_file: None,
    };

    // 3. Execute!
    let response = agent.execute(&request)?;

    println!("Response: {}", response.text);
    if let Some(usage) = response.token_usage {
        println!("{}", usage.to_display_line());
    }

    Ok(())
}
```

---

## 📊 Token Usage & Efficiency

`servling` doesn't just run agents; it watches them. It parses CLI output to provide detailed stats:

```text
📊 Tokens: 1.2M in, 8.5k out (935.5k cached) | Model: sonnet | Premium: 3
```

It can even calculate the estimated cost of your session and rate your efficiency from **Excellent ✨** to **Critical ❌**.

---

## 🛠️ Internal Architecture

- **`core.rs`**: The `Servling` trait and fundamental request/response types.
- **`runner.rs`**: Low-level CLI execution, streaming, and timeout logic.
- **`coding_agent.rs`**: High-level orchestration and fallback chains.
- **`token_usage.rs`**: Regex-powered parsing for AI provider output formats.

---

License: MIT or Apache-2.0
