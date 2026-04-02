//! GitHub Copilot interactive session backend via ACP over stdio.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use agent_client_protocol::{self as acp, Agent as _};
use anyhow::{anyhow, bail, Context, Result};
use futures::{AsyncRead, AsyncWrite};
use tokio::runtime::Builder as RuntimeBuilder;
use tokio::sync::{mpsc as tokio_mpsc, oneshot};
use tokio_util::compat::{TokioAsyncReadCompatExt as _, TokioAsyncWriteCompatExt as _};

use crate::core::{
    normalize_model, Backend, BackendMetadata, ProviderCapabilities, ProviderKind, TransportKind,
};
use crate::session::{
    InteractiveSession, ProviderSessionHandle, ProviderSessionListing, SessionBackend,
    SessionContentKind, SessionEvent, SessionResumeRequest, SessionRuntimeStatus,
    SessionStartRequest, SessionStopReason, SessionTransportError, UserTurnRequest,
};

pub struct CopilotAcpBackend {
    command: Option<String>,
    io_factory: Arc<dyn AcpIoFactory>,
}

impl CopilotAcpBackend {
    pub fn new(command: Option<String>) -> Self {
        Self {
            command,
            io_factory: Arc::new(ProcessAcpIoFactory),
        }
    }

    pub fn check_available() -> Result<()> {
        crate::copilot_agent::CopilotAgent::check_available()
    }

    #[cfg(test)]
    fn with_io_factory(command: Option<String>, io_factory: Arc<dyn AcpIoFactory>) -> Self {
        Self {
            command,
            io_factory,
        }
    }

    fn launch_spec(
        &self,
        working_dir: &Path,
        writable_roots: &[PathBuf],
        model: Option<&str>,
    ) -> Result<(String, Vec<String>)> {
        let mut parts: Vec<String> = self
            .command
            .clone()
            .unwrap_or_else(|| "copilot".to_string())
            .split_whitespace()
            .map(|part| part.to_string())
            .collect();

        if parts.is_empty() {
            bail!("Copilot ACP command is empty");
        }

        let program = parts.remove(0);
        let mut args = parts;
        if !args.iter().any(|arg| arg == "--acp") {
            args.push("--acp".to_string());
        }
        if !args
            .iter()
            .any(|arg| arg == "--allow-all-tools" || arg == "--allow-all" || arg == "--yolo")
        {
            args.push("--allow-all-tools".to_string());
        }
        if !args.iter().any(|arg| arg == "--no-ask-user") {
            args.push("--no-ask-user".to_string());
        }

        let roots = if writable_roots.is_empty() {
            vec![working_dir.to_path_buf()]
        } else {
            writable_roots.to_vec()
        };
        for root in roots {
            args.push("--add-dir".to_string());
            args.push(root.display().to_string());
        }

        if !args.iter().any(|arg| arg == "--model") {
            if let Some(model) = model.and_then(normalize_copilot_model) {
                args.push("--model".to_string());
                args.push(model);
            }
        }

        Ok((program, args))
    }

    fn spawn_session(
        &self,
        working_dir: &Path,
        writable_roots: &[PathBuf],
        model: Option<&str>,
        load_session_ref: Option<&str>,
    ) -> Result<Box<dyn InteractiveSession>> {
        let (program, args) = self.launch_spec(working_dir, writable_roots, model)?;
        let (event_tx, event_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let (command_tx, command_rx) = tokio_mpsc::unbounded_channel();
        let handle_state = Arc::new(Mutex::new(ProviderSessionHandle::new(
            ProviderKind::Copilot,
            TransportKind::CliJsonRpc,
            load_session_ref.map(str::to_string),
            ProviderCapabilities::session_jsonrpc(),
            SessionRuntimeStatus::Starting,
        )));

        let initial_working_dir = working_dir.to_path_buf();
        let state_for_thread = handle_state.clone();
        let load_session_ref = load_session_ref.map(str::to_string);
        let io_factory = self.io_factory.clone();

        thread::spawn(move || {
            if let Err(err) = run_worker(
                program,
                args,
                initial_working_dir,
                io_factory,
                event_tx,
                ready_tx,
                command_rx,
                state_for_thread,
                load_session_ref,
            ) {
                let _ = err;
            }
        });

        let mut ready_handle = ready_rx
            .recv()
            .context("Copilot ACP worker did not report readiness")??;
        ready_handle.status = SessionRuntimeStatus::Ready;
        *handle_state.lock().unwrap() = ready_handle.clone();

        Ok(Box::new(CopilotAcpSession {
            handle_state,
            event_rx: Mutex::new(event_rx),
            command_tx,
        }))
    }
}

impl Backend for CopilotAcpBackend {
    fn metadata(&self) -> BackendMetadata {
        BackendMetadata {
            name: "copilot",
            provider_kind: ProviderKind::Copilot,
            transport_kind: TransportKind::CliJsonRpc,
            capabilities: ProviderCapabilities::session_jsonrpc(),
        }
    }
}

impl SessionBackend for CopilotAcpBackend {
    fn start_session(&self, request: &SessionStartRequest) -> Result<Box<dyn InteractiveSession>> {
        self.spawn_session(
            &request.working_dir,
            &request.writable_roots,
            request.model.as_deref(),
            None,
        )
    }

    fn resume_session(
        &self,
        request: &SessionResumeRequest,
    ) -> Result<Box<dyn InteractiveSession>> {
        self.spawn_session(
            &request.working_dir,
            &request.writable_roots,
            request.model.as_deref(),
            Some(&request.provider_session_ref),
        )
    }

    fn list_sessions(&self) -> Result<Vec<ProviderSessionListing>> {
        let runtime = RuntimeBuilder::new_current_thread()
            .enable_all()
            .build()
            .context("failed to build temporary Copilot ACP runtime")?;
        let local_set = tokio::task::LocalSet::new();
        let working_dir = std::env::current_dir().context("failed to get current directory")?;
        let (program, args) = self.launch_spec(&working_dir, &[], None)?;

        runtime.block_on(local_set.run_until(async move {
            let connection = self
                .io_factory
                .connect(&program, &args)
                .await
                .with_context(|| format!("failed to launch `{program}` for ACP listing"))?;

            let (conn, io_task) = acp::ClientSideConnection::new(
                NullClient,
                connection.outgoing,
                connection.incoming,
                |fut| {
                    tokio::task::spawn_local(fut);
                },
            );
            let io_handle = tokio::task::spawn_local(async move { io_task.await });

            conn.initialize(acp::InitializeRequest::new(acp::ProtocolVersion::V1))
                .await
                .context("Copilot ACP initialize failed during list")?;

            let response = conn
                .list_sessions(acp::ListSessionsRequest::new())
                .await
                .context("Copilot ACP list_sessions failed")?;

            io_handle.abort();
            connection.guard.shutdown();

            Ok(response
                .sessions
                .into_iter()
                .map(|session| ProviderSessionListing {
                    provider_session_ref: session.session_id.0.to_string(),
                    working_dir: session.cwd,
                    title: session.title,
                    updated_at: session.updated_at,
                })
                .collect())
        }))
    }
}

struct CopilotAcpSession {
    handle_state: Arc<Mutex<ProviderSessionHandle>>,
    event_rx: Mutex<mpsc::Receiver<SessionEvent>>,
    command_tx: tokio_mpsc::UnboundedSender<WorkerCommand>,
}

impl Drop for CopilotAcpSession {
    fn drop(&mut self) {
        let _ = self.command_tx.send(WorkerCommand::Shutdown);
    }
}

impl InteractiveSession for CopilotAcpSession {
    fn handle(&self) -> ProviderSessionHandle {
        self.handle_state.lock().unwrap().clone()
    }

    fn status(&self) -> SessionRuntimeStatus {
        self.handle_state.lock().unwrap().status.clone()
    }

    fn send_user_turn(&self, request: &UserTurnRequest) -> Result<SessionStopReason> {
        let status = self.status();
        if matches!(
            status,
            SessionRuntimeStatus::Running | SessionRuntimeStatus::Interrupting
        ) {
            bail!("Copilot ACP session is busy and does not support enqueue while running");
        }

        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(WorkerCommand::SendTurn {
                message: request.message.clone(),
                reply: reply_tx,
            })
            .map_err(|_| anyhow!("Copilot ACP session worker is unavailable"))?;

        reply_rx
            .blocking_recv()
            .map_err(|_| anyhow!("Copilot ACP session worker dropped turn response"))?
    }

    fn interrupt(&self) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.command_tx
            .send(WorkerCommand::Interrupt { reply: reply_tx })
            .map_err(|_| anyhow!("Copilot ACP session worker is unavailable"))?;

        reply_rx
            .blocking_recv()
            .map_err(|_| anyhow!("Copilot ACP session worker dropped interrupt response"))?
    }

    fn next_event(&self, timeout: Duration) -> Result<Option<SessionEvent>> {
        let receiver = self.event_rx.lock().unwrap();
        match receiver.recv_timeout(timeout) {
            Ok(event) => Ok(Some(event)),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => Ok(None),
        }
    }
}

enum WorkerCommand {
    SendTurn {
        message: String,
        reply: oneshot::Sender<Result<SessionStopReason>>,
    },
    Interrupt {
        reply: oneshot::Sender<Result<()>>,
    },
    Shutdown,
}

struct TurnTaskResult {
    reply: oneshot::Sender<Result<SessionStopReason>>,
    result: Result<SessionStopReason>,
}

fn run_worker(
    program: String,
    args: Vec<String>,
    working_dir: PathBuf,
    io_factory: Arc<dyn AcpIoFactory>,
    event_tx: mpsc::Sender<SessionEvent>,
    ready_tx: mpsc::SyncSender<Result<ProviderSessionHandle>>,
    mut command_rx: tokio_mpsc::UnboundedReceiver<WorkerCommand>,
    handle_state: Arc<Mutex<ProviderSessionHandle>>,
    load_session_ref: Option<String>,
) -> Result<()> {
    let runtime = RuntimeBuilder::new_current_thread()
        .enable_all()
        .build()
        .context("failed to build Copilot ACP runtime")?;
    let local_set = tokio::task::LocalSet::new();

    runtime.block_on(local_set.run_until(async move {
        let connection = io_factory
            .connect(&program, &args)
            .await
            .with_context(|| format!("failed to launch `{program}` for ACP session"))?;

        let client = SessionEventClient {
            event_tx: event_tx.clone(),
        };
        let (conn, io_task) = acp::ClientSideConnection::new(
            client,
            connection.outgoing,
            connection.incoming,
            |fut| {
                tokio::task::spawn_local(fut);
            },
        );
        let conn = Rc::new(conn);
        let mut io_handle = tokio::task::spawn_local(async move { io_task.await });
        let (turn_result_tx, mut turn_result_rx) = tokio_mpsc::unbounded_channel();

        conn.initialize(
            acp::InitializeRequest::new(acp::ProtocolVersion::V1).client_info(
                acp::Implementation::new("servling", env!("CARGO_PKG_VERSION"))
                    .title("servling ACP client"),
            ),
        )
        .await
        .context("Copilot ACP initialize failed")?;

        let session_id = if let Some(existing_session_id) = load_session_ref.clone() {
            conn.load_session(acp::LoadSessionRequest::new(
                existing_session_id.clone(),
                working_dir.clone(),
            ))
            .await
            .context("Copilot ACP load_session failed")?;
            acp::SessionId::new(existing_session_id)
        } else {
            conn.new_session(acp::NewSessionRequest::new(working_dir.clone()))
                .await
                .context("Copilot ACP new_session failed")?
                .session_id
        };

        let started_handle = ProviderSessionHandle::new(
            ProviderKind::Copilot,
            TransportKind::CliJsonRpc,
            Some(session_id.0.to_string()),
            ProviderCapabilities::session_jsonrpc(),
            SessionRuntimeStatus::Ready,
        );
        *handle_state.lock().unwrap() = started_handle.clone();
        let _ = event_tx.send(SessionEvent::SessionStarted {
            provider_session_ref: started_handle.provider_session_ref.clone(),
        });
        let _ = event_tx.send(SessionEvent::StatusChanged {
            status: SessionRuntimeStatus::Ready,
        });
        let _ = ready_tx.send(Ok(started_handle));

        let mut turn_inflight = false;
        loop {
            tokio::select! {
                maybe_command = command_rx.recv() => {
                    match maybe_command {
                        Some(WorkerCommand::SendTurn { message, reply }) => {
                            if turn_inflight {
                                let _ = reply.send(Err(anyhow!(
                                    "Copilot ACP session is busy and does not support enqueue while running"
                                )));
                                continue;
                            }

                            update_status(&handle_state, &event_tx, SessionRuntimeStatus::Running);
                            turn_inflight = true;
                            let conn = conn.clone();
                            let session_id = session_id.clone();
                            let turn_result_tx = turn_result_tx.clone();
                            tokio::task::spawn_local(async move {
                                let result = conn
                                    .prompt(acp::PromptRequest::new(session_id, vec![message.into()]))
                                    .await
                                    .map(|prompt| map_stop_reason(prompt.stop_reason))
                                    .map_err(|err| anyhow!(err.to_string()));
                                let _ = turn_result_tx.send(TurnTaskResult { reply, result });
                            });
                        }
                        Some(WorkerCommand::Interrupt { reply }) => {
                            update_status(&handle_state, &event_tx, SessionRuntimeStatus::Interrupting);
                            let result = conn.cancel(acp::CancelNotification::new(session_id.clone())).await;
                            if let Err(err) = &result {
                                update_status(&handle_state, &event_tx, SessionRuntimeStatus::Failed);
                                let _ = event_tx.send(SessionEvent::Error {
                                    error: SessionTransportError::new(err.to_string()),
                                });
                            }
                            let _ = reply.send(result.map_err(|err| anyhow!(err.to_string())));
                        }
                        Some(WorkerCommand::Shutdown) | None => {
                            break;
                        }
                    }
                }
                Some(turn_result) = turn_result_rx.recv() => {
                    turn_inflight = false;
                    match &turn_result.result {
                        Ok(stop_reason) => {
                            let _ = event_tx.send(SessionEvent::TurnCompleted {
                                stop_reason: stop_reason.clone(),
                            });
                            update_status(&handle_state, &event_tx, SessionRuntimeStatus::Ready);
                        }
                        Err(err) => {
                            update_status(&handle_state, &event_tx, SessionRuntimeStatus::Failed);
                            let _ = event_tx.send(SessionEvent::Error {
                                error: SessionTransportError::new(err.to_string()),
                            });
                        }
                    }
                    let _ = turn_result.reply.send(turn_result.result);
                }
                io_result = &mut io_handle => {
                    match io_result {
                        Ok(Ok(())) => update_status(&handle_state, &event_tx, SessionRuntimeStatus::Ended),
                        Ok(Err(err)) => {
                            update_status(&handle_state, &event_tx, SessionRuntimeStatus::Failed);
                            let _ = event_tx.send(SessionEvent::Error {
                                error: SessionTransportError::new(err.to_string()),
                            });
                        }
                        Err(err) => {
                            update_status(&handle_state, &event_tx, SessionRuntimeStatus::Failed);
                            let _ = event_tx.send(SessionEvent::Error {
                                error: SessionTransportError::new(err.to_string()),
                            });
                        }
                    }
                    break;
                }
            }
        }

        connection.guard.shutdown();
        let _ = event_tx.send(SessionEvent::SessionEnded);
        Ok(())
    }))
}

fn update_status(
    handle_state: &Arc<Mutex<ProviderSessionHandle>>,
    event_tx: &mpsc::Sender<SessionEvent>,
    status: SessionRuntimeStatus,
) {
    handle_state.lock().unwrap().status = status.clone();
    let _ = event_tx.send(SessionEvent::StatusChanged { status });
}

fn normalize_copilot_model(model: &str) -> Option<String> {
    normalize_model("copilot", Some(model.to_string())).map(|normalized| {
        match normalized.to_lowercase().as_str() {
            "opus" => "claude-opus-4.5".to_string(),
            "sonnet" => "claude-sonnet-4.5".to_string(),
            "haiku" => "claude-haiku-4.5".to_string(),
            _ => normalized,
        }
    })
}

fn map_stop_reason(reason: acp::StopReason) -> SessionStopReason {
    match reason {
        acp::StopReason::EndTurn => SessionStopReason::EndTurn,
        acp::StopReason::MaxTokens => SessionStopReason::MaxTokens,
        acp::StopReason::MaxTurnRequests => SessionStopReason::MaxTurnRequests,
        acp::StopReason::Refusal => SessionStopReason::Refusal,
        acp::StopReason::Cancelled => SessionStopReason::Cancelled,
        other => SessionStopReason::Unknown(format!("{other:?}")),
    }
}

struct NullClient;

#[async_trait::async_trait(?Send)]
impl acp::Client for NullClient {
    async fn request_permission(
        &self,
        _args: acp::RequestPermissionRequest,
    ) -> acp::Result<acp::RequestPermissionResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn write_text_file(
        &self,
        _args: acp::WriteTextFileRequest,
    ) -> acp::Result<acp::WriteTextFileResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn read_text_file(
        &self,
        _args: acp::ReadTextFileRequest,
    ) -> acp::Result<acp::ReadTextFileResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn create_terminal(
        &self,
        _args: acp::CreateTerminalRequest,
    ) -> acp::Result<acp::CreateTerminalResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn terminal_output(
        &self,
        _args: acp::TerminalOutputRequest,
    ) -> acp::Result<acp::TerminalOutputResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn release_terminal(
        &self,
        _args: acp::ReleaseTerminalRequest,
    ) -> acp::Result<acp::ReleaseTerminalResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn wait_for_terminal_exit(
        &self,
        _args: acp::WaitForTerminalExitRequest,
    ) -> acp::Result<acp::WaitForTerminalExitResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn kill_terminal(
        &self,
        _args: acp::KillTerminalRequest,
    ) -> acp::Result<acp::KillTerminalResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn session_notification(&self, _args: acp::SessionNotification) -> acp::Result<()> {
        Ok(())
    }

    async fn ext_method(&self, _args: acp::ExtRequest) -> acp::Result<acp::ExtResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn ext_notification(&self, _args: acp::ExtNotification) -> acp::Result<()> {
        Err(acp::Error::method_not_found())
    }
}

struct SessionEventClient {
    event_tx: mpsc::Sender<SessionEvent>,
}

#[async_trait::async_trait(?Send)]
impl acp::Client for SessionEventClient {
    async fn request_permission(
        &self,
        args: acp::RequestPermissionRequest,
    ) -> acp::Result<acp::RequestPermissionResponse> {
        let _ = self.event_tx.send(SessionEvent::Warning {
            message: format!(
                "Copilot ACP requested permission for {}",
                args.tool_call.tool_call_id.0
            ),
        });

        let selected = args
            .options
            .iter()
            .find(|option| {
                matches!(
                    option.kind,
                    acp::PermissionOptionKind::AllowAlways | acp::PermissionOptionKind::AllowOnce
                )
            })
            .or_else(|| args.options.first())
            .ok_or_else(acp::Error::invalid_params)?;

        Ok(acp::RequestPermissionResponse::new(
            acp::RequestPermissionOutcome::Selected(acp::SelectedPermissionOutcome::new(
                selected.option_id.clone(),
            )),
        ))
    }

    async fn write_text_file(
        &self,
        args: acp::WriteTextFileRequest,
    ) -> acp::Result<acp::WriteTextFileResponse> {
        std::fs::write(&args.path, &args.content).map_err(to_acp_internal_error)?;
        Ok(acp::WriteTextFileResponse::new())
    }

    async fn read_text_file(
        &self,
        args: acp::ReadTextFileRequest,
    ) -> acp::Result<acp::ReadTextFileResponse> {
        let content = std::fs::read_to_string(&args.path).map_err(to_acp_internal_error)?;
        Ok(acp::ReadTextFileResponse::new(content))
    }

    async fn create_terminal(
        &self,
        _args: acp::CreateTerminalRequest,
    ) -> acp::Result<acp::CreateTerminalResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn terminal_output(
        &self,
        _args: acp::TerminalOutputRequest,
    ) -> acp::Result<acp::TerminalOutputResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn release_terminal(
        &self,
        _args: acp::ReleaseTerminalRequest,
    ) -> acp::Result<acp::ReleaseTerminalResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn wait_for_terminal_exit(
        &self,
        _args: acp::WaitForTerminalExitRequest,
    ) -> acp::Result<acp::WaitForTerminalExitResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn kill_terminal(
        &self,
        _args: acp::KillTerminalRequest,
    ) -> acp::Result<acp::KillTerminalResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn session_notification(&self, args: acp::SessionNotification) -> acp::Result<()> {
        if let Some(event) = map_session_notification(args) {
            let _ = self.event_tx.send(event);
        }
        Ok(())
    }

    async fn ext_method(&self, _args: acp::ExtRequest) -> acp::Result<acp::ExtResponse> {
        Err(acp::Error::method_not_found())
    }

    async fn ext_notification(&self, _args: acp::ExtNotification) -> acp::Result<()> {
        Err(acp::Error::method_not_found())
    }
}

fn to_acp_internal_error(error: std::io::Error) -> acp::Error {
    acp::Error::internal_error().data(error.to_string())
}

fn map_session_notification(notification: acp::SessionNotification) -> Option<SessionEvent> {
    match notification.update {
        acp::SessionUpdate::UserMessageChunk(chunk) => Some(SessionEvent::ContentChunk {
            kind: SessionContentKind::User,
            text: content_block_to_text(chunk.content),
        }),
        acp::SessionUpdate::AgentMessageChunk(chunk) => Some(SessionEvent::ContentChunk {
            kind: SessionContentKind::Assistant,
            text: content_block_to_text(chunk.content),
        }),
        acp::SessionUpdate::AgentThoughtChunk(chunk) => Some(SessionEvent::ContentChunk {
            kind: SessionContentKind::Thought,
            text: content_block_to_text(chunk.content),
        }),
        acp::SessionUpdate::ToolCall(tool_call) => Some(SessionEvent::ToolCall {
            tool_name: tool_call.title,
            call_id: Some(tool_call.tool_call_id.0.to_string()),
        }),
        acp::SessionUpdate::ToolCallUpdate(update) => Some(SessionEvent::ToolCallUpdate {
            call_id: Some(update.tool_call_id.0.to_string()),
            state: update
                .fields
                .status
                .map(|status| format!("{status:?}"))
                .unwrap_or_else(|| "updated".to_string()),
            detail: update
                .fields
                .title
                .or_else(|| {
                    update
                        .fields
                        .content
                        .as_ref()
                        .and_then(|content| content.first())
                        .map(|entry| format!("{entry:?}"))
                })
                .unwrap_or_else(|| "tool call updated".to_string()),
        }),
        acp::SessionUpdate::Plan(_) => Some(SessionEvent::Warning {
            message: "Copilot ACP emitted a plan update".to_string(),
        }),
        acp::SessionUpdate::AvailableCommandsUpdate(_) => Some(SessionEvent::Warning {
            message: "Copilot ACP available commands changed".to_string(),
        }),
        acp::SessionUpdate::CurrentModeUpdate(update) => Some(SessionEvent::Warning {
            message: format!("Copilot ACP mode changed to {}", update.current_mode_id.0),
        }),
        acp::SessionUpdate::ConfigOptionUpdate(_) => Some(SessionEvent::Warning {
            message: "Copilot ACP configuration options changed".to_string(),
        }),
        acp::SessionUpdate::SessionInfoUpdate(update) => Some(SessionEvent::Warning {
            message: format!(
                "Copilot ACP session info updated{}",
                update
                    .title
                    .value()
                    .map(|title| format!(": {title}"))
                    .unwrap_or_default()
            ),
        }),
        _ => None,
    }
}

fn content_block_to_text(content: acp::ContentBlock) -> String {
    match content {
        acp::ContentBlock::Text(text) => text.text,
        acp::ContentBlock::Image(_) => "<image>".to_string(),
        acp::ContentBlock::Audio(_) => "<audio>".to_string(),
        acp::ContentBlock::ResourceLink(link) => link.uri,
        acp::ContentBlock::Resource(_) => "<resource>".to_string(),
        other => format!("{other:?}"),
    }
}

struct AcpConnection {
    outgoing: Box<dyn AsyncWrite + Unpin + Send>,
    incoming: Box<dyn AsyncRead + Unpin + Send>,
    guard: Box<dyn AcpConnectionGuard>,
}

trait AcpConnectionGuard: Send {
    fn shutdown(self: Box<Self>);
}

#[async_trait::async_trait]
trait AcpIoFactory: Send + Sync {
    async fn connect(&self, program: &str, args: &[String]) -> Result<AcpConnection>;
}

struct ProcessAcpIoFactory;

#[async_trait::async_trait]
impl AcpIoFactory for ProcessAcpIoFactory {
    async fn connect(&self, program: &str, args: &[String]) -> Result<AcpConnection> {
        let mut child = tokio::process::Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("failed to launch `{program}` for ACP transport"))?;

        let stdin = child
            .stdin
            .take()
            .context("Copilot ACP stdin was unavailable")?;
        let stdout = child
            .stdout
            .take()
            .context("Copilot ACP stdout was unavailable")?;

        Ok(AcpConnection {
            outgoing: Box::new(stdin.compat_write()),
            incoming: Box::new(stdout.compat()),
            guard: Box::new(ProcessAcpConnectionGuard { child }),
        })
    }
}

struct ProcessAcpConnectionGuard {
    child: tokio::process::Child,
}

impl AcpConnectionGuard for ProcessAcpConnectionGuard {
    fn shutdown(mut self: Box<Self>) {
        let _ = self.child.start_kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        LLMRequest, LLMResponse, OutcomeClassification, ProviderCapabilities, ProviderKind,
        TurnRunner,
    };
    use agent_client_protocol::Client as _;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tokio::sync::mpsc as tokio_broadcast;

    struct FakeAcpIoFactory;

    #[async_trait::async_trait]
    impl AcpIoFactory for FakeAcpIoFactory {
        async fn connect(&self, _program: &str, _args: &[String]) -> Result<AcpConnection> {
            let (client_side, server_side) = tokio::io::duplex(16 * 1024);
            let (client_read, client_write) = tokio::io::split(client_side);
            let (server_read, server_write) = tokio::io::split(server_side);

            thread::spawn(move || {
                let runtime = RuntimeBuilder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("runtime");
                let local_set = tokio::task::LocalSet::new();
                runtime.block_on(local_set.run_until(async move {
                    let (notification_tx, mut notification_rx) =
                        tokio_broadcast::unbounded_channel();
                    let fake_agent = FakeAgent::new(notification_tx);
                    let (conn, io_task) = acp::AgentSideConnection::new(
                        fake_agent,
                        server_write.compat_write(),
                        server_read.compat(),
                        |fut| {
                            tokio::task::spawn_local(fut);
                        },
                    );

                    tokio::task::spawn_local(async move {
                        while let Some(notification) = notification_rx.recv().await {
                            let _ = conn.session_notification(notification).await;
                        }
                    });

                    let _ = io_task.await;
                }));
            });

            Ok(AcpConnection {
                outgoing: Box::new(client_write.compat_write()),
                incoming: Box::new(client_read.compat()),
                guard: Box::new(NoopGuard),
            })
        }
    }

    struct NoopGuard;

    impl AcpConnectionGuard for NoopGuard {
        fn shutdown(self: Box<Self>) {}
    }

    struct FakeAgent {
        notification_tx: tokio_broadcast::UnboundedSender<acp::SessionNotification>,
        cancelled: Arc<AtomicBool>,
        next_session: Cell<u64>,
    }

    impl FakeAgent {
        fn new(
            notification_tx: tokio_broadcast::UnboundedSender<acp::SessionNotification>,
        ) -> Self {
            Self {
                notification_tx,
                cancelled: Arc::new(AtomicBool::new(false)),
                next_session: Cell::new(1),
            }
        }

        fn session_id(&self) -> String {
            format!("fake-session-{}", self.next_session.get())
        }
    }

    #[async_trait::async_trait(?Send)]
    impl acp::Agent for FakeAgent {
        async fn initialize(
            &self,
            _args: acp::InitializeRequest,
        ) -> acp::Result<acp::InitializeResponse> {
            Ok(acp::InitializeResponse::new(acp::ProtocolVersion::V1)
                .agent_capabilities(
                    acp::AgentCapabilities::new()
                        .load_session(true)
                        .prompt_capabilities(
                            acp::PromptCapabilities::new()
                                .image(true)
                                .audio(false)
                                .embedded_context(true),
                        )
                        .session_capabilities(
                            acp::SessionCapabilities::new()
                                .list(acp::SessionListCapabilities::new()),
                        ),
                )
                .agent_info(acp::Implementation::new("Fake Copilot", "test").title("Fake Copilot")))
        }

        async fn authenticate(
            &self,
            _args: acp::AuthenticateRequest,
        ) -> acp::Result<acp::AuthenticateResponse> {
            Ok(acp::AuthenticateResponse::default())
        }

        async fn new_session(
            &self,
            _args: acp::NewSessionRequest,
        ) -> acp::Result<acp::NewSessionResponse> {
            let session_id = self.session_id();
            self.next_session.set(self.next_session.get() + 1);
            Ok(acp::NewSessionResponse::new(session_id))
        }

        async fn load_session(
            &self,
            args: acp::LoadSessionRequest,
        ) -> acp::Result<acp::LoadSessionResponse> {
            let _ = self.notification_tx.send(acp::SessionNotification::new(
                args.session_id.clone(),
                acp::SessionUpdate::UserMessageChunk(acp::ContentChunk::new(
                    "previous user turn".to_string().into(),
                )),
            ));
            let _ = self.notification_tx.send(acp::SessionNotification::new(
                args.session_id,
                acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                    "previous assistant turn".to_string().into(),
                )),
            ));
            Ok(acp::LoadSessionResponse::new())
        }

        async fn prompt(&self, args: acp::PromptRequest) -> acp::Result<acp::PromptResponse> {
            let prompt_text = args
                .prompt
                .iter()
                .filter_map(|block| match block {
                    acp::ContentBlock::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" ");

            if prompt_text.contains("cancel test") {
                let _ = self.notification_tx.send(acp::SessionNotification::new(
                    args.session_id.clone(),
                    acp::SessionUpdate::AgentThoughtChunk(acp::ContentChunk::new(
                        "thinking".to_string().into(),
                    )),
                ));
                for _ in 0..100 {
                    if self.cancelled.load(Ordering::SeqCst) {
                        let _ = self.notification_tx.send(acp::SessionNotification::new(
                            args.session_id,
                            acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                                "Info: Operation cancelled by user".to_string().into(),
                            )),
                        ));
                        self.cancelled.store(false, Ordering::SeqCst);
                        return Ok(acp::PromptResponse::new(acp::StopReason::EndTurn));
                    }
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                return Ok(acp::PromptResponse::new(acp::StopReason::EndTurn));
            }

            let _ = self.notification_tx.send(acp::SessionNotification::new(
                args.session_id.clone(),
                acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                    "fake".to_string().into(),
                )),
            ));
            let _ = self.notification_tx.send(acp::SessionNotification::new(
                args.session_id,
                acp::SessionUpdate::AgentMessageChunk(acp::ContentChunk::new(
                    " response".to_string().into(),
                )),
            ));
            Ok(acp::PromptResponse::new(acp::StopReason::EndTurn))
        }

        async fn cancel(&self, _args: acp::CancelNotification) -> acp::Result<()> {
            self.cancelled.store(true, Ordering::SeqCst);
            Ok(())
        }

        async fn set_session_mode(
            &self,
            _args: acp::SetSessionModeRequest,
        ) -> acp::Result<acp::SetSessionModeResponse> {
            Ok(acp::SetSessionModeResponse::default())
        }

        async fn list_sessions(
            &self,
            _args: acp::ListSessionsRequest,
        ) -> acp::Result<acp::ListSessionsResponse> {
            Ok(acp::ListSessionsResponse::new(vec![acp::SessionInfo::new(
                "listed-session",
                "/tmp/fake",
            )
            .title("Listed Session")
            .updated_at("2026-03-17T00:00:00Z")]))
        }

        async fn set_session_config_option(
            &self,
            _args: acp::SetSessionConfigOptionRequest,
        ) -> acp::Result<acp::SetSessionConfigOptionResponse> {
            Ok(acp::SetSessionConfigOptionResponse::new(vec![]))
        }

        async fn ext_method(&self, _args: acp::ExtRequest) -> acp::Result<acp::ExtResponse> {
            Err(acp::Error::method_not_found())
        }

        async fn ext_notification(&self, _args: acp::ExtNotification) -> acp::Result<()> {
            Err(acp::Error::method_not_found())
        }
    }

    #[test]
    fn copilot_acp_capabilities_are_honest() {
        let backend = CopilotAcpBackend::with_io_factory(None, Arc::new(FakeAcpIoFactory));
        let caps = backend.capabilities();

        assert!(!caps.supports_batch_mode());
        assert!(caps.supports_interactive_session_mode());
        assert!(caps.supports_resume());
        assert!(caps.supports_operator_interrupt());
        assert!(caps.supports_durable_provider_session_ref());
        assert!(caps.supports_structured_event_stream());
        assert!(!caps.supports_live_steering_while_running());
        assert!(!caps.supports_batch_fallback());
        assert!(caps.session_provider_pinned());
    }

    #[test]
    fn copilot_acp_session_happy_path_is_streamed() {
        let backend = CopilotAcpBackend::with_io_factory(None, Arc::new(FakeAcpIoFactory));
        let session = backend
            .start_session(&SessionStartRequest::new("/tmp/fake"))
            .expect("session starts");

        let handle = session.handle();
        assert_eq!(handle.provider_kind, ProviderKind::Copilot);
        assert_eq!(handle.transport_kind, TransportKind::CliJsonRpc);
        assert_eq!(
            handle.provider_session_ref.as_deref(),
            Some("fake-session-1")
        );

        let started = session
            .next_event(Duration::from_secs(1))
            .expect("event read")
            .expect("session started");
        assert!(matches!(started, SessionEvent::SessionStarted { .. }));

        let stop_reason = session
            .send_user_turn(&UserTurnRequest::new("hello"))
            .expect("turn succeeds");
        assert_eq!(stop_reason, SessionStopReason::EndTurn);

        let mut chunks = Vec::new();
        for _ in 0..4 {
            if let Some(event) = session
                .next_event(Duration::from_secs(1))
                .expect("event read")
            {
                if let SessionEvent::ContentChunk {
                    kind: SessionContentKind::Assistant,
                    text,
                } = event
                {
                    chunks.push(text);
                }
            }
        }
        assert_eq!(chunks.concat(), "fake response");
        assert_eq!(session.status(), SessionRuntimeStatus::Ready);
    }

    #[test]
    fn copilot_acp_resume_and_list_sessions_work() {
        let backend = CopilotAcpBackend::with_io_factory(None, Arc::new(FakeAcpIoFactory));
        let listed = backend.list_sessions().expect("sessions listed");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].provider_session_ref, "listed-session");
        assert_eq!(listed[0].title.as_deref(), Some("Listed Session"));

        let session = backend
            .resume_session(&SessionResumeRequest::new("loaded-session", "/tmp/fake"))
            .expect("resume succeeds");

        let mut saw_user = false;
        let mut saw_assistant = false;
        for _ in 0..6 {
            if let Some(SessionEvent::ContentChunk { kind, .. }) =
                session.next_event(Duration::from_secs(1)).expect("event")
            {
                saw_user |= kind == SessionContentKind::User;
                saw_assistant |= kind == SessionContentKind::Assistant;
            }
        }

        assert!(saw_user);
        assert!(saw_assistant);
    }

    #[test]
    fn copilot_acp_interrupt_is_provider_pinned_and_not_enqueueable() {
        let backend = CopilotAcpBackend::with_io_factory(None, Arc::new(FakeAcpIoFactory));
        let session: Arc<dyn InteractiveSession> = Arc::from(
            backend
                .start_session(&SessionStartRequest::new("/tmp/fake"))
                .expect("session starts"),
        );

        let _ = session.next_event(Duration::from_secs(1)).expect("started");
        let session_for_turn = session.clone();
        let join = thread::spawn(move || {
            session_for_turn
                .send_user_turn(&UserTurnRequest::new("cancel test"))
                .expect("turn returns")
        });

        loop {
            let event = session
                .next_event(Duration::from_secs(1))
                .expect("event read")
                .expect("event");
            match event {
                SessionEvent::StatusChanged {
                    status: SessionRuntimeStatus::Running,
                } => break,
                _ => {}
            }
        }

        let second_turn_error = session.send_user_turn(&UserTurnRequest::new("second turn"));
        assert!(second_turn_error.is_err());

        session.interrupt().expect("interrupt works");
        let stop_reason = join.join().expect("join");
        assert_eq!(stop_reason, SessionStopReason::EndTurn);

        let mut saw_cancel_message = false;
        let mut seen_events = Vec::new();
        for _ in 0..10 {
            if let Some(event) = session.next_event(Duration::from_secs(1)).expect("event") {
                seen_events.push(format!("{event:?}"));
                match event {
                    SessionEvent::ContentChunk {
                        kind: SessionContentKind::Assistant,
                        text,
                    } if text.contains("cancelled") => {
                        saw_cancel_message = true;
                    }
                    _ => {}
                }
            }
        }

        let handle = session.handle();
        assert!(
            saw_cancel_message,
            "events after interrupt: {seen_events:?}"
        );
        assert_eq!(handle.provider_kind, ProviderKind::Copilot);
        assert_eq!(handle.transport_kind, TransportKind::CliJsonRpc);
    }

    struct StubRunner {
        name: &'static str,
        responses: Mutex<Vec<Result<OutcomeClassification, &'static str>>>,
    }

    impl Backend for StubRunner {
        fn metadata(&self) -> BackendMetadata {
            BackendMetadata {
                name: self.name,
                provider_kind: ProviderKind::Composite,
                transport_kind: TransportKind::CliBatch,
                capabilities: ProviderCapabilities::batch_only(),
            }
        }
    }

    impl TurnRunner for StubRunner {
        fn execute(&self, _request: &LLMRequest) -> Result<LLMResponse> {
            match self.responses.lock().unwrap().remove(0) {
                Ok(classification) => Ok(LLMResponse {
                    text: self.name.to_string(),
                    classification,
                    backend_name: Some(self.name.to_string()),
                    exit_code: Some(0),
                    token_usage: None,
                    elapsed_seconds: 0.0,
                    stdout_path: None,
                    stderr_path: None,
                }),
                Err(message) => Err(anyhow!(message)),
            }
        }
    }

    #[test]
    fn batch_fallback_lane_is_preserved() {
        let agent = crate::coding_agent::CodingAgent::builder()
            .register(Box::new(StubRunner {
                name: "first",
                responses: Mutex::new(vec![Ok(OutcomeClassification::RateLimited)]),
            }))
            .register(Box::new(StubRunner {
                name: "second",
                responses: Mutex::new(vec![Ok(OutcomeClassification::Ok)]),
            }))
            .build()
            .expect("agent builds");

        let response = agent
            .execute(&LLMRequest {
                prompt: "hi".to_string(),
                working_dir: PathBuf::from("."),
                source_writable_roots: vec![PathBuf::from(".")],
                runtime_writable_roots: Vec::new(),
                runtime_env: Vec::new(),
                runtime_profile: None,
                model: None,
                max_runtime_seconds: 5,
                stream_output: false,
                input_file: None,
                temp_dir_override: None,
            })
            .expect("batch fallback succeeds");

        assert_eq!(response.text, "second");
        assert!(agent.capabilities().supports_batch_fallback());
    }

    // -----------------------------------------------------------------------
    // Real-provider probe tests (opt-in, require live Copilot CLI + auth)
    //
    // Run:  cargo test -p servling -- real_provider --ignored --nocapture
    // -----------------------------------------------------------------------

    /// Basic turn: starts a real session, sends one short non-destructive turn,
    /// drains all events, and asserts session returns to Ready.
    #[test]
    #[ignore = "requires live copilot CLI + GitHub auth; run: cargo test -p servling -- --ignored --nocapture"]
    fn real_provider_probe_basic_turn() {
        use std::fs;

        let workspace = "/tmp/brood-probe-workspace";
        fs::create_dir_all(workspace).expect("tmp workspace");

        let backend = CopilotAcpBackend::with_io_factory(None, Arc::new(ProcessAcpIoFactory));
        let session = backend
            .start_session(&SessionStartRequest::new(workspace))
            .expect("real session starts");

        // Drain startup events (expect SessionStarted within 20 s)
        let mut startup_events: Vec<String> = Vec::new();
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            assert!(
                std::time::Instant::now() < deadline,
                "session did not start within 20 s; events so far: {startup_events:?}"
            );
            if let Some(event) = session
                .next_event(Duration::from_secs(2))
                .expect("startup event read")
            {
                let label = format!("{event:?}");
                eprintln!("[probe/startup] {label}");
                startup_events.push(label);
                if matches!(event, SessionEvent::SessionStarted { .. }) {
                    break;
                }
            }
        }
        eprintln!(
            "[probe] session started; total startup events: {}",
            startup_events.len()
        );

        // Send a simple non-destructive turn that needs no tool use
        let stop_reason = session
            .send_user_turn(&UserTurnRequest::new(
                "Reply with exactly: PROBE_OK — no tools, no preamble, nothing else.",
            ))
            .expect("turn completes");

        eprintln!("[probe] stop_reason: {stop_reason:?}");
        eprintln!("[probe] post-turn status: {:?}", session.status());

        // Drain remaining queued events from the turn
        let mut turn_events: Vec<String> = Vec::new();
        loop {
            match session.next_event(Duration::from_millis(300)) {
                Ok(Some(event)) => {
                    let label = format!("{event:?}");
                    eprintln!("[probe/turn] {label}");
                    turn_events.push(label);
                }
                _ => break,
            }
        }
        eprintln!("[probe] turn events drained: {} events", turn_events.len());

        // Assertions
        assert_eq!(
            session.status(),
            SessionRuntimeStatus::Ready,
            "status must return to Ready after turn"
        );
        let ok = matches!(
            stop_reason,
            SessionStopReason::EndTurn
                | SessionStopReason::MaxTokens
                | SessionStopReason::Unknown(_)
        );
        assert!(
            ok,
            "stop_reason must be a recognized terminal reason; got: {stop_reason:?}"
        );
    }

    /// Interrupt probe: starts a real session, sends a long-running turn,
    /// interrupts after a short delay, and records what the provider yields.
    #[test]
    #[ignore = "requires live copilot CLI + GitHub auth; run: cargo test -p servling -- --ignored --nocapture"]
    fn real_provider_probe_interrupt() {
        use std::fs;

        let workspace = "/tmp/brood-probe-workspace";
        fs::create_dir_all(workspace).expect("tmp workspace");

        let backend = CopilotAcpBackend::with_io_factory(None, Arc::new(ProcessAcpIoFactory));
        let session: Arc<dyn InteractiveSession> = Arc::from(
            backend
                .start_session(&SessionStartRequest::new(workspace))
                .expect("real session starts"),
        );

        // Wait for SessionStarted
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        loop {
            assert!(
                std::time::Instant::now() < deadline,
                "session did not start within 20 s"
            );
            if let Some(event) = session
                .next_event(Duration::from_secs(2))
                .expect("startup event read")
            {
                eprintln!("[probe/interrupt/startup] {event:?}");
                if matches!(event, SessionEvent::SessionStarted { .. }) {
                    break;
                }
            }
        }

        // Spawn turn in separate thread (blocks until done)
        let session_for_turn = session.clone();
        let join = thread::spawn(move || {
            session_for_turn.send_user_turn(&UserTurnRequest::new(
                "Count slowly from 1 to 500, one number per line, no other text.",
            ))
        });

        // Wait for Running status before interrupting
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            assert!(
                std::time::Instant::now() < deadline,
                "never saw Running status"
            );
            if let Some(event) = session.next_event(Duration::from_secs(2)).expect("event") {
                eprintln!("[probe/interrupt/pre] {event:?}");
                if matches!(
                    event,
                    SessionEvent::StatusChanged {
                        status: SessionRuntimeStatus::Running
                    }
                ) {
                    break;
                }
            }
        }
        eprintln!("[probe/interrupt] Running confirmed; interrupting...");
        session.interrupt().expect("interrupt issued without error");

        // Collect post-interrupt events until turn thread finishes
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut post_events: Vec<String> = Vec::new();
        loop {
            assert!(
                std::time::Instant::now() < deadline,
                "turn thread did not finish within 30 s post-interrupt"
            );
            if join.is_finished() {
                // Drain a bit more to capture TurnCompleted + StatusChanged
                loop {
                    match session.next_event(Duration::from_millis(200)) {
                        Ok(Some(event)) => {
                            eprintln!("[probe/interrupt/post] {event:?}");
                            post_events.push(format!("{event:?}"));
                        }
                        _ => break,
                    }
                }
                break;
            }
            if let Some(event) = session.next_event(Duration::from_secs(1)).expect("event") {
                eprintln!("[probe/interrupt/post] {event:?}");
                post_events.push(format!("{event:?}"));
            }
        }

        let stop_reason = join.join().expect("thread join").expect("turn returned Ok");
        eprintln!("[probe/interrupt] final stop_reason: {stop_reason:?}");
        eprintln!(
            "[probe/interrupt] post-interrupt events ({}):",
            post_events.len()
        );
        for e in &post_events {
            eprintln!("  {e}");
        }
        eprintln!(
            "[probe/interrupt] final session status: {:?}",
            session.status()
        );

        // After interrupt + turn completion the session must be usable again
        assert!(
            matches!(
                session.status(),
                SessionRuntimeStatus::Ready | SessionRuntimeStatus::Ended
            ),
            "session should be Ready or Ended after interrupt, got: {:?}",
            session.status()
        );
    }
}
