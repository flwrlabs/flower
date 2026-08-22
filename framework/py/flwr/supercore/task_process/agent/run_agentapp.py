# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower AgentApp process."""


import os
import ssl
from logging import DEBUG, ERROR
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile

import httpx

from flwr.agentapp import AgentApp, LoadAgentAppError
from flwr.app import Context
from flwr.app.exception import AppExitException
from flwr.cli.config_utils import get_fab_metadata
from flwr.cli.install import install_from_fab
from flwr.cli.utils import get_sha256_hash
from flwr.common.config import (
    get_fused_config_from_dir,
    get_project_config,
    get_project_dir,
)
from flwr.common.constant import RUNTIME_DEPENDENCY_INSTALL, SubStatus
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_from_proto,
    fab_to_proto,
    run_from_proto,
    user_config_to_proto,
)
from flwr.proto.control_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushTaskOutputRequest,
)
from flwr.supercore import log
from flwr.supercore.app_utils import start_parent_process_monitor
from flwr.supercore.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.supercore.heartbeat import HeartbeatSender, make_task_heartbeat_fn_http
from flwr.supercore.logger import flush_logs, start_log_uploader, stop_log_uploader
from flwr.supercore.object_ref import load_app
from flwr.supercore.superexec.dependency_installer import (
    RuntimeDependencyInstallationError,
    cleanup_app_runtime_environment,
    install_app_dependencies,
)
from flwr.supercore.telemetry import EventType, event
from flwr.supercore.typing import JSONObject
from flwr.superlink.grid import HttpGrid

from .context_items import append_items
from .session import RuntimeAgentConnectors, RuntimeAgentResponses, RuntimeAgentSession

_AGENT_INPUT_KEY = "agent.input"
_RUNTIME_API_KEY_ENV = "FLWR_RUNTIME_API_KEY"
_RUNTIME_BASE_URL_ENV = "FLWR_RUNTIME_BASE_URL"
_SSL_CERT_FILE_ENV = "SSL_CERT_FILE"
_SSL_CERT_DIR_ENV = "SSL_CERT_DIR"


def run_agentapp(  # pylint: disable=R0912, R0913, R0914, R0915, R0917, W0212
    runtime_api_address: str,
    log_queue: Queue[str | None],
    token: str,
    insecure: bool,
    certificates: bytes | None = None,
    parent_pid: int | None = None,
    runtime_dependency_install: bool = RUNTIME_DEPENDENCY_INSTALL,
) -> None:
    """Run Flower AgentApp process."""
    # Monitor the main process in case of SIGKILL
    if parent_pid is not None:
        start_parent_process_monitor(parent_pid)

    # Initialize the Runtime API connection.
    grid = HttpGrid(
        runtime_api_address=runtime_api_address,
        insecure=insecure,
        root_certificates=certificates,
        token=token,
    )

    log_uploader = None
    hash_run_id = None
    sub_status = SubStatus.FAILED
    details = "Task failed with unknown error."
    heartbeat_sender = None
    context: Context | None = None
    runtime_env_dir: Path | None = None
    runtime_root_certificates_path: Path | None = None
    exit_code = ExitCode.SUCCESS

    def on_exit() -> None:
        log(DEBUG, "[flwr-agentapp] Will push AgentApp task output")

        grid._retry_invoker.max_tries = 1

        if log_uploader:
            flush_logs(log_queue)

        pushoutput_req = PushTaskOutputRequest(
            context=context_to_proto(context) if context else None,
            sub_status=sub_status,
            details=details,
        )
        try:
            grid._runtime_client.PushTaskOutput(pushoutput_req)
        except httpx.HTTPError as err:
            log(ERROR, "Failed to push task output: %s", str(err))

        if log_uploader:
            stop_log_uploader(log_queue, log_uploader)

        if heartbeat_sender and heartbeat_sender.is_running:
            heartbeat_sender.stop()

        grid.close()

        cleanup_app_runtime_environment(runtime_env_dir)
        if runtime_root_certificates_path is not None:
            runtime_root_certificates_path.unlink(missing_ok=True)

    register_signal_handlers(
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
        exit_message="Task stopped by user.",
        exit_handlers=[on_exit],
    )

    try:
        heartbeat_sender = HeartbeatSender(
            make_task_heartbeat_fn_http(grid._runtime_client)
        )
        heartbeat_sender.start()

        log(DEBUG, "[flwr-agentapp] Pull task input")
        res: PullTaskInputResponse = grid._runtime_client.PullTaskInput(
            PullTaskInputRequest()
        )

        context = context_from_proto(res.context)
        run = run_from_proto(res.run)
        fab = fab_from_proto(res.fab)
        task_id = res.task_id

        hash_run_id = get_sha256_hash(run.run_id)

        grid.set_run(run)

        log_uploader = start_log_uploader(
            log_queue=log_queue,
            node_id=0,
            run_id=run.run_id,
            client=grid._runtime_client,
        )

        log(DEBUG, "[flwr-agentapp] Start FAB installation.")
        install_from_fab(fab.content, skip_prompt=True)

        fab_id, fab_version = get_fab_metadata(fab.content)

        app_path = str(get_project_dir(fab_id, fab_version, fab.hash_str))

        if runtime_dependency_install:
            log(DEBUG, "[flwr-agentapp] Installing app dependencies.")
            runtime_env_dir = install_app_dependencies(
                app_path,
                launch_id=token,
                run_id=run.run_id,
                index_context={
                    "component": "agentapp",
                    "project_dir": app_path,
                    "run_id": run.run_id,
                    "launch_id": token,
                    "fab_id": run.fab_id,
                    "fab_version": run.fab_version,
                    "fab_hash": fab.hash_str,
                },
            )
        else:
            log(
                DEBUG,
                "[flwr-agentapp] Runtime dependency installation is disabled.",
            )

        config = get_project_config(app_path)

        agent_app_attr = config["tool"]["flwr"]["app"]["components"]["agentapp"]
        context.run_config = get_fused_config_from_dir(
            Path(app_path), run.override_config
        )

        agent_input = context.run_config.get(_AGENT_INPUT_KEY)
        if agent_input is not None:
            if not isinstance(agent_input, str):
                raise ValueError("context.run_config['agent.input'] must be a string.")
            if agent_input:
                item: JSONObject = {
                    "type": "message",
                    "role": "user",
                    "content": agent_input,
                }
                append_items(context, [item])

        log(
            DEBUG,
            "[flwr-agentapp] Will load AgentApp `%s` in %s",
            agent_app_attr,
            app_path,
        )

        event(
            EventType.FLWR_AGENTAPP_RUN_ENTER,
            event_details={"run-id-hash": hash_run_id},
        )

        responses = RuntimeAgentResponses(
            stub=grid._runtime_client,
            run_id=context.run_id,
            task_id=task_id,
            context=context,
            start_run_request=StartRunRequest(
                fab=fab_to_proto(fab),
                override_config=user_config_to_proto(run.override_config),
                override_federation_config=res.federation_config,
                federation=run.federation_id,
                series_id=run.series_id,
            ),
        )
        connectors = RuntimeAgentConnectors(responses)
        agent = RuntimeAgentSession(responses=responses, connectors=connectors)

        runtime_root_certificates_path = _set_runtime_environment(
            runtime_api_address, token, insecure, certificates
        )

        # Load and run the AgentApp
        agent_app = load_app(agent_app_attr, LoadAgentAppError, app_path)
        if not isinstance(agent_app, AgentApp):
            raise LoadAgentAppError(
                f"Attribute '{agent_app_attr}' is not of type '{AgentApp.__name__}'.",
            ) from None
        agent_app(agent=agent, context=context)

        # Set sub_status and details for successful completion
        sub_status = SubStatus.COMPLETED
        details = ""

    except Exception as ex:  # pylint: disable=broad-exception-caught
        log(ERROR, "AgentApp raised an exception", exc_info=ex)

        sub_status = SubStatus.FAILED
        details = f"AgentApp failed with exception: {str(ex)}"

        exit_code = ExitCode.TASK_PROC_EXCEPTION
        if isinstance(ex, AppExitException):
            exit_code = ex.exit_code
        elif isinstance(ex, ImportError):
            exit_code = ExitCode.COMMON_APP_IMPORT_ERROR
        elif isinstance(ex, RuntimeDependencyInstallationError):
            exit_code = ExitCode.COMMON_RUNTIME_DEPENDENCY_INSTALLATION_ERROR

    flwr_exit(
        code=exit_code,
        event_type=EventType.FLWR_AGENTAPP_RUN_LEAVE,
        event_details={
            "run-id-hash": hash_run_id,
            "success": exit_code == ExitCode.SUCCESS,
        },
    )


def _set_runtime_environment(
    runtime_api_address: str,
    token: str,
    insecure: bool,
    certificates: bytes | None = None,
) -> Path | None:
    """Expose the Open Responses-compatible Runtime endpoint to the AgentApp."""
    scheme = "http" if insecure else "https"
    address = runtime_api_address.rstrip("/")
    os.environ[_RUNTIME_BASE_URL_ENV] = f"{scheme}://{address}/v1/runtime"
    os.environ[_RUNTIME_API_KEY_ENV] = token

    if certificates is None:
        return None

    # OpenAI/httpx reads custom CAs from SSL_CERT_FILE, which requires a path.
    # Extend the public and inherited roots with the Runtime CA for this process.
    public_ca_context = httpx.create_ssl_context(trust_env=False)
    trusted_certificates = [
        *public_ca_context.get_ca_certs(binary_form=True),
        *_load_inherited_ca_certificates(),
    ]
    with NamedTemporaryFile(
        mode="wb", prefix="flwr-runtime-ca-", suffix=".pem", delete=False
    ) as certificate_file:
        for trusted_certificate in dict.fromkeys(trusted_certificates):
            certificate_file.write(
                ssl.DER_cert_to_PEM_cert(trusted_certificate).encode("ascii")
            )
        certificate_file.write(b"\n")
        certificate_file.write(certificates)
    certificate_path = Path(certificate_file.name)
    os.environ[_SSL_CERT_FILE_ENV] = str(certificate_path)
    return certificate_path


def _load_inherited_ca_certificates() -> list[bytes]:
    """Load roots configured through the standard OpenSSL environment."""
    ca_file = os.environ.get(_SSL_CERT_FILE_ENV)
    ca_dir = os.environ.get(_SSL_CERT_DIR_ENV)
    if not ca_file and not ca_dir:
        return []

    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if ca_file:
        context.load_verify_locations(cafile=ca_file)
    if ca_dir:
        for directory in ca_dir.split(os.pathsep):
            if not directory:
                continue
            try:
                ca_paths = sorted(Path(directory).iterdir())
            except OSError:
                continue
            for ca_path in ca_paths:
                if not ca_path.is_file():
                    continue
                try:
                    context.load_verify_locations(cafile=ca_path)
                except OSError:
                    continue
    return context.get_ca_certs(binary_form=True)
