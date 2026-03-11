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

## ⚙️ Under the Hood: How it Works

`servling` doesn't just call a library; it orchestrates full-blown CLI processes. Here's what happens when you call `agent.execute()`:

1.  **Command Expansion**: It takes a template (e.g., `claude --print`) and dynamically injects parameters like `{input_file}`, `{mission_dir}`, and the requested `--model`.
2.  **Subprocess Management**: It spawns the CLI tool using `std::process::Command`, carefully piping `stdin`, `stdout`, and `stderr`.
3.  **IO Orchestration**:
    *   **Stdin**: If the agent needs a prompt, `servling` writes it directly to the process's standard input.
    *   **Stdout**: Captures the agent's response. If `stream_output` is enabled, it renders a cleaned-up version of the response (e.g., stripping JSON noise from Claude's output) in real-time.
    *   **Stderr**: This is the "metadata channel." `servling` scans this stream with optimized regex to extract token usage, premium request counts, and model names.
4.  **Resilience**: It monitors the process for timeouts and specific error patterns (like rate limits) to decide if it should trigger an automatic fallback to the next agent in your chain.

---

## 🛠️ Internal Architecture

- **`core.rs`**: The `Servling` trait and fundamental request/response types.
- **`runner.rs`**: Low-level CLI execution, streaming, and timeout logic.
- **`coding_agent.rs`**: High-level orchestration and fallback chains.
- **`token_usage.rs`**: Regex-powered parsing for AI provider output formats.