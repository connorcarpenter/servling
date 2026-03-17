//! Claude Code agent.

use crate::cli_backend::CliBackend;
use crate::core::{
    LLMRequest, LLMResponse, ProviderCapabilities, ProviderKind, RunnerInvocation, TransportKind,
    TurnRunner,
};
use anyhow::{bail, Result};
use std::process::{Command, Stdio};

pub struct ClaudeAgent {
    cli: CliBackend,
}

impl ClaudeAgent {
    pub fn new(command: Option<String>, stream_output: bool) -> Self {
        let template = command.unwrap_or_else(|| {
            if stream_output {
                "claude --print --settings {claude_settings} --output-format stream-json --include-partial-messages --verbose".to_string()
            } else {
                "claude --print --settings {claude_settings}".to_string()
            }
        });
        Self {
            cli: CliBackend {
                name: "claude",
                command_template: template,
            },
        }
    }

    pub fn check_available() -> Result<()> {
        let output = Command::new("claude")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        match output {
            Ok(status) if status.success() => Ok(()),
            Ok(_) => bail!("Claude CLI returned non-zero exit code"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                bail!("Claude CLI not found. Please install Claude Code CLI.")
            }
            Err(e) => bail!("Failed to check Claude CLI availability: {}", e),
        }
    }
}

impl TurnRunner for ClaudeAgent {
    fn name(&self) -> &'static str {
        self.cli.name
    }

    fn provider_kind(&self) -> ProviderKind {
        ProviderKind::Claude
    }

    fn transport_kind(&self) -> TransportKind {
        TransportKind::CliBatch
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::batch_only()
    }

    fn execute(&self, request: &LLMRequest) -> Result<LLMResponse> {
        self.cli.execute_with_expansion(request, true, None)
    }

    fn planned_invocation(&self, request: &LLMRequest) -> Option<RunnerInvocation> {
        let cmd = self.cli.expand_command(
            &self.cli.command_template,
            &request.working_dir,
            &request.writable_roots,
            request.input_file.as_deref(),
            None,
            request.model.as_deref(),
        );
        let parts: Vec<String> = cmd
            .split_whitespace()
            .map(|s: &str| s.to_string())
            .collect();
        if parts.is_empty() {
            return None;
        }

        let mission_dir = request
            .input_file
            .as_ref()
            .and_then(|p| p.parent())
            .unwrap_or(&request.working_dir);
        let mission_dir_abs =
            std::fs::canonicalize(mission_dir).unwrap_or_else(|_| mission_dir.to_path_buf());

        Some(RunnerInvocation {
            program: parts[0].clone(),
            args: parts[1..].to_vec(),
            working_dir: request.working_dir.display().to_string(),
            env: vec![(
                "TESAKI_MISSION_DIR".to_string(),
                mission_dir_abs.display().to_string(),
            )],
        })
    }
}
