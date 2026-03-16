//! Unified CLI-based LLM backend.

use anyhow::Result;
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::core::{LLMRequest, LLMResponse};
use crate::runner::{run_cli_runner, CliRunnerConfig};

/// A unified workhorse for any CLI-based LLM agent.
pub struct CliBackend {
    pub name: &'static str,
    pub command_template: String,
}

impl CliBackend {
    pub fn expand_command(
        &self,
        base_cmd: &str,
        working_dir: &Path,
        writable_roots: &[PathBuf],
        input_path: Option<&Path>,
        output_path: Option<&Path>,
        model: Option<&str>,
    ) -> String {
        let mut cmd = base_cmd.to_string();

        if let Some(m) = model {
            cmd = format!("{} --model {}", cmd, m);
        }

        let mission_dir = input_path.and_then(|p| p.parent()).unwrap_or(working_dir);
        let writable_root = writable_roots
            .first()
            .map(PathBuf::as_path)
            .unwrap_or(working_dir);
        let add_dir_args = writable_roots
            .iter()
            .skip(1)
            .filter(|root| root.as_path() != working_dir)
            .map(|root| format!("--add-dir {}", root.display()))
            .collect::<Vec<_>>()
            .join(" ");
        let claude_settings = build_claude_settings(working_dir, writable_roots);

        cmd.replace(
            "{input_file}",
            &input_path
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        )
        .replace("{mission_dir}", &mission_dir.display().to_string())
        .replace("{working_dir}", &working_dir.display().to_string())
        .replace("{writable_root}", &writable_root.display().to_string())
        .replace("{add_dir_args}", &add_dir_args)
        .replace("{claude_settings}", &claude_settings)
        .replace(
            "{output_file}",
            &output_path
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
        )
    }

    pub fn prepare_temp_files(
        &self,
        prompt: &str,
        temp_dir: &Path,
    ) -> Result<(
        Option<tempfile::NamedTempFile>,
        Option<PathBuf>,
        Option<tempfile::NamedTempFile>,
        Option<PathBuf>,
    )> {
        let mut temp_input = None;
        let mut input_path = None;
        if self.command_template.contains("{input_file}")
            || self.command_template.contains("{mission_dir}")
        {
            let mut file = tempfile::NamedTempFile::new_in(temp_dir)?;
            file.write_all(prompt.as_bytes())?;
            input_path = Some(file.path().to_path_buf());
            temp_input = Some(file);
        }

        let mut temp_output = None;
        let mut output_path = None;
        if self.command_template.contains("{output_file}") {
            let file = tempfile::NamedTempFile::new_in(temp_dir)?;
            output_path = Some(file.path().to_path_buf());
            temp_output = Some(file);
        }

        Ok((temp_input, input_path, temp_output, output_path))
    }

    pub fn execute_with_expansion(
        &self,
        request: &LLMRequest,
        extract_error: bool,
        model_expander: Option<fn(&str) -> String>,
    ) -> Result<LLMResponse> {
        let temp_dir = request
            .writable_roots
            .first()
            .map(PathBuf::as_path)
            .unwrap_or(&request.working_dir);
        let (temp_input, input_path, temp_output, output_path) =
            self.prepare_temp_files(&request.prompt, temp_dir)?;

        let model = request.model.as_deref().map(|m| {
            if let Some(expander) = model_expander {
                expander(m)
            } else {
                m.to_string()
            }
        });

        let cmd = self.expand_command(
            &self.command_template,
            &request.working_dir,
            &request.writable_roots,
            input_path.as_deref().or(request.input_file.as_deref()),
            output_path.as_deref(),
            model.as_deref(),
        );

        let config = CliRunnerConfig {
            working_dir: request.working_dir.clone(),
            max_runtime_seconds: request.max_runtime_seconds,
            stream_output: request.stream_output,
        };

        let mission_dir = input_path
            .as_deref()
            .or(request.input_file.as_deref())
            .and_then(|p| p.parent())
            .unwrap_or(&request.working_dir);

        let outcome = run_cli_runner(
            &cmd,
            mission_dir,
            &config,
            extract_error,
            if temp_input.is_some() || request.input_file.is_some() {
                None
            } else {
                Some(request.prompt.clone())
            },
            input_path.as_deref().or(request.input_file.as_deref()),
            output_path.as_deref(),
        )?;

        let text = if let Some(out_p) = output_path {
            std::fs::read_to_string(out_p).unwrap_or_default()
        } else {
            outcome
                .stdout_path
                .as_ref()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .unwrap_or_default()
        };

        drop(temp_input);
        drop(temp_output);

        Ok(LLMResponse {
            text,
            exit_code: outcome.exit_code,
            classification: outcome.classification,
            stdout_path: outcome.stdout_path,
            stderr_path: outcome.stderr_path,
            token_usage: outcome.token_usage,
            elapsed_seconds: outcome.elapsed_seconds,
        })
    }
}

fn build_claude_settings(working_dir: &Path, writable_roots: &[PathBuf]) -> String {
    let additional_directories = writable_roots
        .iter()
        .filter(|root| root.as_path() != working_dir)
        .map(|root| claude_path(root))
        .collect::<Vec<_>>();
    let allow_write = writable_roots
        .iter()
        .map(|root| claude_path(root))
        .collect::<Vec<_>>();

    json!({
        "sandbox": {
            "enabled": true,
            "autoAllowBashIfSandboxed": true,
            "allowUnsandboxedCommands": false,
            "filesystem": {
                "allowWrite": allow_write,
            }
        },
        "permissions": {
            "defaultMode": "acceptEdits",
            "additionalDirectories": additional_directories,
        }
    })
    .to_string()
}

fn claude_path(path: &Path) -> String {
    let rendered = path.display().to_string();
    if rendered.starts_with('/') {
        format!("//{}", rendered.trim_start_matches('/'))
    } else {
        rendered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_command_uses_writable_root_placeholders() {
        let backend = CliBackend {
            name: "test",
            command_template: "tool --root {writable_root} {add_dir_args}".to_string(),
        };
        let working_dir = PathBuf::from("/repo/candidate");
        let writable_roots = vec![
            PathBuf::from("/repo/candidate"),
            PathBuf::from("/repo/generated"),
        ];

        let expanded = backend.expand_command(
            &backend.command_template,
            &working_dir,
            &writable_roots,
            None,
            None,
            None,
        );

        assert_eq!(
            expanded,
            "tool --root /repo/candidate --add-dir /repo/generated"
        );
    }

    #[test]
    fn expand_command_embeds_claude_settings() {
        let backend = CliBackend {
            name: "claude",
            command_template: "claude --settings {claude_settings}".to_string(),
        };
        let working_dir = PathBuf::from("/repo/candidate");
        let writable_roots = vec![
            PathBuf::from("/repo/candidate"),
            PathBuf::from("/repo/generated"),
        ];

        let expanded = backend.expand_command(
            &backend.command_template,
            &working_dir,
            &writable_roots,
            None,
            None,
            None,
        );

        assert!(expanded.contains("\"enabled\":true"));
        assert!(expanded.contains("\"defaultMode\":\"acceptEdits\""));
        assert!(expanded.contains("\"allowWrite\":[\"//repo/candidate\",\"//repo/generated\"]"));
        assert!(expanded.contains("\"additionalDirectories\":[\"//repo/generated\"]"));
    }
}
