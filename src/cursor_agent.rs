//! Cursor Agent CLI (`agent` on PATH).

use crate::cli_backend::CliBackend;
use crate::core::{
    Backend, BackendMetadata, LLMRequest, LLMResponse, ProviderCapabilities, ProviderKind,
    RunnerInvocation, TransportKind, TurnRunner,
};
use anyhow::{bail, Result};
use std::process::{Command, Stdio};

pub struct CursorAgent {
    cli: CliBackend,
}

impl CursorAgent {
    pub fn new(command: Option<String>, stream_output: bool) -> Self {
        let template = command.unwrap_or_else(|| {
            if stream_output {
                "agent {provider_args} --print --trust --workspace {working_dir} --force --output-format stream-json --stream-partial-output"
                    .to_string()
            } else {
                "agent {provider_args} --print --trust --workspace {working_dir} --force --output-format json"
                    .to_string()
            }
        });
        Self {
            cli: CliBackend {
                name: "cursor",
                command_template: template,
            },
        }
    }

    pub fn check_available() -> Result<()> {
        let output = Command::new("agent")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();

        match output {
            Ok(status) if status.success() => Ok(()),
            Ok(_) => bail!("Cursor Agent CLI returned non-zero exit code"),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                bail!("Cursor Agent CLI (`agent`) not found. Install Cursor CLI and ensure it is on PATH.")
            }
            Err(e) => bail!("Failed to check Cursor Agent CLI availability: {}", e),
        }
    }
}

impl Backend for CursorAgent {
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            name: self.cli.name,
            provider_kind: ProviderKind::Cursor,
            transport_kind: TransportKind::CliBatch,
            capabilities: ProviderCapabilities::batch_only(),
        }
    }
}

impl TurnRunner for CursorAgent {
    fn execute(&self, request: &LLMRequest) -> Result<LLMResponse> {
        self.cli
            .execute_with_expansion(request, true, None)
    }

    fn planned_invocation(&self, request: &LLMRequest) -> Option<RunnerInvocation> {
        let writable_roots = request.writable_roots();
        let cmd = self.cli.expand_command(
            &self.cli.command_template,
            &request.working_dir,
            &writable_roots,
            request.input_file.as_deref(),
            None,
            request.model.as_deref(),
            request.reasoning_effort.as_deref(),
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

        let mut env = vec![(
            "TESAKI_MISSION_DIR".to_string(),
            mission_dir_abs.display().to_string(),
        )];
        env.extend(request.runtime_env.iter().cloned());

        Some(RunnerInvocation {
            program: parts[0].clone(),
            args: parts[1..].to_vec(),
            working_dir: request.working_dir.display().to_string(),
            env,
        })
    }
}
