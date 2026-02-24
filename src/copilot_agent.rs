//! GitHub Copilot CLI agent.

use anyhow::{bail, Result};
use std::process::{Command, Stdio};
use crate::core::{Servling, LLMRequest, LLMResponse, RunnerInvocation};
use crate::cli_backend::CliBackend;

pub struct CopilotAgent {
    cli: CliBackend,
}

impl CopilotAgent {
    pub fn new(command: Option<String>) -> Self {
        let template = command.unwrap_or_else(|| {
            "copilot -p @{input_file} --allow-all --add-dir {working_dir}".to_string()
        });
        Self {
            cli: CliBackend {
                name: "copilot",
                command_template: template,
            },
        }
    }

    pub fn check_available() -> Result<()> {
        let output = Command::new("copilot")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        match output {
            Ok(status) if status.success() => Ok(()),
            Ok(_) => bail!("Copilot CLI returned non-zero exit code"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                bail!("Copilot CLI not found. Please install GitHub Copilot CLI.")
            }
            Err(e) => bail!("Failed to check Copilot CLI availability: {}", e),
        }
    }
}

impl Servling for CopilotAgent {
    fn name(&self) -> &'static str {
        "copilot"
    }

    fn execute(&self, request: &LLMRequest) -> Result<LLMResponse> {
        self.cli.execute_with_expansion(request, false, Some(expand_model_name))
    }

    fn planned_invocation(&self, request: &LLMRequest) -> Option<RunnerInvocation> {
        let cmd = self.cli.expand_command(
            &self.cli.command_template,
            &request.working_dir,
            request.input_file.as_deref(),
            None,
            request.model.as_deref().map(|m| expand_model_name(m)).as_deref()
        );
        let parts: Vec<String> = cmd.split_whitespace().map(|s: &str| s.to_string()).collect();
        if parts.is_empty() { return None; }

        let mission_dir = request.input_file.as_ref()
            .and_then(|p| p.parent())
            .unwrap_or(&request.working_dir);
        let mission_dir_abs = std::fs::canonicalize(mission_dir).unwrap_or_else(|_| mission_dir.to_path_buf());

        Some(RunnerInvocation {
            program: parts[0].clone(),
            args: parts[1..].to_vec(),
            working_dir: request.working_dir.display().to_string(),
            env: vec![("TESAKI_MISSION_DIR".to_string(), mission_dir_abs.display().to_string())],
        })
    }
}

fn expand_model_name(tier: &str) -> String {
    match tier.to_lowercase().as_str() {
        "opus" => "claude-opus-4.5".to_string(),
        "sonnet" => "claude-sonnet-4.5".to_string(),
        "haiku" => "claude-haiku-4.5".to_string(),
        _ => tier.to_string(),
    }
}
