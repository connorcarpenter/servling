//! Codex CLI agent.

use crate::cli_backend::CliBackend;
use crate::core::{LLMRequest, LLMResponse, RunnerInvocation, Servling};
use anyhow::{bail, Result};
use std::process::{Command, Stdio};

pub struct CodexAgent {
    cli: CliBackend,
}

impl CodexAgent {
    pub fn new(command: Option<String>) -> Self {
        let template = command.unwrap_or_else(|| {
            "codex -c approval_policy=\"never\" -c sandbox_mode=\"workspace-write\" exec -C {working_dir} {add_dir_args} -".to_string()
        });
        Self {
            cli: CliBackend {
                name: "codex",
                command_template: template,
            },
        }
    }

    pub fn check_available() -> Result<()> {
        let output = Command::new("codex")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        match output {
            Ok(status) if status.success() => Ok(()),
            Ok(_) => bail!("Codex CLI returned non-zero exit code"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                bail!("Codex CLI not found. Please install Codex CLI.")
            }
            Err(e) => bail!("Failed to check Codex CLI availability: {}", e),
        }
    }
}

impl Servling for CodexAgent {
    fn name(&self) -> &'static str {
        self.cli.name
    }

    fn execute(&self, request: &LLMRequest) -> Result<LLMResponse> {
        self.cli.execute_with_expansion(request, false, None)
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
