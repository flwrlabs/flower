# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""`flower-superlink` command."""

# pylint: disable=too-many-lines

import argparse
import os
import subprocess
import sys
import threading
from logging import INFO, WARN
from pathlib import Path
from time import sleep

import uvicorn

from flwr.common.args import (
    add_args_removed_health_server,
    add_args_runtime_dependency_install,
    try_obtain_server_certificates,
)
from flwr.common.constant import (
    CONTROL_API_DEFAULT_SERVER_ADDRESS,
    FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION,
    ISOLATION_MODE_PROCESS,
    ISOLATION_MODE_SUBPROCESS,
)
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore import log
from flwr.supercore.address import parse_address, resolve_bind_address
from flwr.supercore.auth import (
    add_superexec_auth_secret_args,
    load_superexec_auth_secret,
)
from flwr.supercore.constant import (
    FLWR_IN_MEMORY_DB_NAME,
    SUPERLINK_UVICORN_DEFAULT_PORT,
    UVICORN_DEFAULT_HOST,
)
from flwr.supercore.exit import ExitCode, flwr_exit
from flwr.supercore.http_logging import get_uvicorn_log_config
from flwr.supercore.logger import configure_superlink_log_file, console_handler
from flwr.supercore.telemetry import EventType, event
from flwr.supercore.tls import get_client_tls_args
from flwr.supercore.update_check import warn_if_flwr_update_available
from flwr.supercore.utils import get_popen_detach_kwargs
from flwr.supercore.version import package_version
from flwr.superlink.artifact_provider import ArtifactProvider
from flwr.superlink.config_loader import (
    SuperLinkLifespanConfig,
    load_control_authn_plugin,
    load_control_event_log_plugin,
    load_fleet_event_log_plugin,
)

try:
    from flwr.ee import add_ee_args_superlink, get_ee_artifact_provider
except ImportError:

    # pylint: disable-next=unused-argument
    def add_ee_args_superlink(parser: argparse.ArgumentParser) -> None:
        """Add EE-specific arguments to the parser."""

    def get_ee_artifact_provider(config_path: str) -> ArtifactProvider:
        """Return the EE artifact provider."""
        raise NotImplementedError("No artifact provider is currently supported.")


class SuperLinkLifespan:
    """Own the shared SuperLink lifespan state and network services."""

    def __init__(
        self,
        config: SuperLinkLifespanConfig,
        state_factory: LinkStateFactory,
    ) -> None:
        self.config = config
        self.bckg_threads: list[threading.Thread] = []
        self.superexec_process: subprocess.Popen[bytes] | None = None
        self.state_factory = state_factory
        self._started = False

    def startup(self) -> None:
        """Start shared lifespan resources."""
        log(INFO, "SuperLinkLifespan: start")
        if self._started:
            return

        # Force initialization before starting network servers
        self.state_factory.state()

        self._start_superexec_if_needed()
        self._started = True

    def shutdown(self) -> None:
        """Stop services started by this lifespan."""
        log(INFO, "SuperLinkLifespan: stop")
        if not self._started and self.superexec_process is None:
            return

        if self.superexec_process is not None:
            self.superexec_process.terminate()
            try:
                self.superexec_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                log(WARN, "SuperExec subprocess did not terminate within 1 second.")

        self.superexec_process = None
        self._started = False

    def wait_until_background_thread_exits(self) -> None:
        """Block like the historical `flower-superlink` command.

        With no background threads, `all([])` is intentionally true, so this loop
        blocks until a signal handler exits the process.
        """
        while all(thread.is_alive() for thread in self.bckg_threads):
            sleep(0.1)

    def _start_superexec_if_needed(self) -> None:
        config = self.config
        if config.isolation != ISOLATION_MODE_SUBPROCESS:
            return

        runtime_host = f"[{config.host}]" if ":" in config.host else config.host
        runtime_address = resolve_bind_address(f"{runtime_host}:{config.port}")
        command = _get_superexec_command(
            runtime_address=runtime_address,
            runtime_certificates=config.certificates,
            runtime_root_certificates_path=config.ssl_ca_certfile,
            parent_pid=os.getpid(),
            runtime_dependency_install=config.runtime_dependency_install,
        )
        # pylint: disable-next=consider-using-with
        self.superexec_process = subprocess.Popen(command, **get_popen_detach_kwargs())


# pylint: disable=too-many-branches, too-many-locals, too-many-statements
def _parse_superlink_lifespan_config() -> SuperLinkLifespanConfig:
    """Parse SuperLink CLI args and return the startup configuration."""
    args = _parse_args_run_superlink().parse_args()

    if args.log_file:
        configure_superlink_log_file(
            filename=args.log_file,
            interval_hours=args.log_rotation_interval_hours,
            backup_count=args.log_rotation_backup_count,
        )

    explicit_args = {arg.split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")}

    # The old opt-in flag is accepted for compatibility, but no longer needed.
    if "--allow-runtime-dependency-installation" in explicit_args:
        log(
            WARN,
            "The `--allow-runtime-dependency-installation` argument is deprecated. "
            "Runtime dependency installation is enabled by default for SuperLink. "
            "Use `--disable-runtime-dependency-installation` to disable it.",
        )

    control_api_set = "--control-api-address" in explicit_args

    if control_api_set:
        log(
            WARN,
            "The `--control-api-address` argument is deprecated. The Control API "
            "now operates over HTTP. Use `--host` and `--port` instead.",
        )

    if "--fleet-api-type" in explicit_args:
        log(
            WARN,
            "The `--fleet-api-type` argument is deprecated. The Fleet API now "
            "operates over HTTP.",
        )
    if "--fleet-api-address" in explicit_args:
        log(
            WARN,
            "The `--fleet-api-address` argument is deprecated. The Fleet API "
            "now operates over HTTP. Use `--host` and `--port` instead.",
        )

    # Obtain certificates
    certificates = _obtain_superlink_certificates(args)

    # Load SuperExec auth secret
    superexec_auth_secret: bytes | None = None
    if args.superexec_auth_secret_file is not None:
        log(
            WARN,
            "EXPERIMENTAL: SuperExec authentication is experimental and "
            "may change in future releases.",
        )
    if args.isolation == ISOLATION_MODE_SUBPROCESS:
        if args.superexec_auth_secret_file is not None:
            log(
                WARN,
                "SuperExec auth secret is ignored in subprocess isolation mode.",
            )
    else:
        # Enable SuperExec auth in process mode when secret is provided
        if args.superexec_auth_secret_file is not None:
            try:
                superexec_auth_secret = load_superexec_auth_secret(
                    secret_file=args.superexec_auth_secret_file,
                )
            except ValueError as err:
                flwr_exit(
                    ExitCode.SUPERLINK_INVALID_ARGS,
                    f"Failed to load SuperExec authentication secret: {err}",
                )

    authn_plugin = load_control_authn_plugin()
    event_log_plugin = (
        load_control_event_log_plugin()
        if getattr(args, "enable_event_log", False)
        else None
    )
    fleet_event_log_plugin = (
        load_fleet_event_log_plugin()
        if getattr(args, "enable_event_log", False)
        else None
    )

    # Load artifact provider if the args.artifact_provider_config is provided
    artifact_provider = None
    if cfg_path := getattr(args, "artifact_provider_config", None):
        log(WARN, "The `--artifact-provider-config` flag is highly experimental.")
        artifact_provider = get_ee_artifact_provider(cfg_path)

    # Check for incompatible args with SuperNode authentication
    enable_supernode_auth: bool = args.enable_supernode_auth
    if enable_supernode_auth:
        if args.insecure:
            url_v = f"https://flower.ai/docs/framework/v{package_version}/en/"
            page = "how-to-authenticate-supernodes.html"
            flwr_exit(
                ExitCode.SUPERLINK_INVALID_ARGS,
                "The `--enable-supernode-auth` flag requires encrypted TLS "
                "communications. Please provide TLS certificates using the "
                "`--ssl-certfile`, `--ssl-keyfile` and `--ssl-ca-certfile` "
                "arguments to your SuperLink. Please refer to the Flower "
                f"documentation for more information: {url_v}{page}",
            )
        if args.simulation:
            log(
                WARN,
                "SuperNode authentication is not applicable with the simulation "
                "runtime as no SuperNodes can connect to this SuperLink. "
                "Proceeding...",
            )
    # If supernode authentication is disabled, warn users
    else:
        log(
            WARN,
            "SuperNode authentication is disabled. The SuperLink will accept "
            "connections from any SuperNode.",
        )

    if args.auth_list_public_keys:
        url_v = f"https://flower.ai/docs/framework/v{package_version}/en/"
        page = "how-to-authenticate-supernodes.html"
        flwr_exit(
            ExitCode.SUPERLINK_INVALID_ARGS,
            "The `--auth-list-public-keys` "
            "argument is no longer supported. To enable SuperNode authentication,  "
            "use the `--enable-supernode-auth` flag and use the Flower CLI to register "
            "SuperNodes by supplying their public keys. Please refer"
            f" to the Flower documentation for more information: {url_v}{page}",
        )

    return SuperLinkLifespanConfig(
        host=args.host,
        port=args.port,
        insecure=args.insecure,
        certificates=certificates,
        superexec_auth_secret=superexec_auth_secret,
        authn_plugin=authn_plugin,
        event_log_plugin=event_log_plugin,
        fleet_event_log_plugin=fleet_event_log_plugin,
        enable_event_log=getattr(args, "enable_event_log", False),
        artifact_provider=artifact_provider,
        enable_supernode_auth=enable_supernode_auth,
        fleet_api_type="http",
        simulation=args.simulation,
        ssl_ca_certfile=(
            str(Path(args.ssl_ca_certfile).expanduser())
            if certificates is not None
            else None
        ),
        ssl_certfile=(
            str(Path(args.ssl_certfile).expanduser())
            if certificates is not None
            else None
        ),
        ssl_keyfile=(
            str(Path(args.ssl_keyfile).expanduser())
            if certificates is not None
            else None
        ),
        database=args.database,
        isolation=args.isolation,
        runtime_dependency_install=args.runtime_dependency_install,
    )


def flower_superlink() -> None:
    """Run Flower SuperLink (Runtime API and Fleet API)."""
    warn_if_flwr_update_available(process_name="flower-superlink")

    config = _parse_superlink_lifespan_config()

    log(INFO, "Starting Flower SuperLink")

    event(EventType.RUN_SUPERLINK_ENTER)

    # Blocking: FastAPI serves the Runtime, Control, and Fleet HTTP APIs.
    _run_superlink_http_api(lifespan_config=config)


def _format_address(address: str) -> tuple[str, str, int]:
    parsed_address = parse_address(address)
    if not parsed_address:
        flwr_exit(
            ExitCode.COMMON_ADDRESS_INVALID,
            f"Address ({address}) cannot be parsed.",
        )
    host, port, is_v6 = parsed_address
    return (f"[{host}]:{port}" if is_v6 else f"{host}:{port}", host, port)


def _run_superlink_http_api(lifespan_config: SuperLinkLifespanConfig) -> None:
    """Run the FastAPI-owned SuperLink service."""
    from flwr.superlink.main import (  # pylint: disable=import-outside-toplevel
        create_app,
    )

    fastapi_app = create_app(lifespan_config, SuperLinkLifespan)

    log(
        INFO,
        "Starting the SuperLink Runtime HTTP API on %s:%s.",
        lifespan_config.host,
        lifespan_config.port,
    )
    log(
        INFO,
        "Starting the SuperLink Control HTTP API on %s:%s.",
        lifespan_config.host,
        lifespan_config.port,
    )
    log(
        INFO,
        "Starting the SuperLink Fleet HTTP API on %s:%s.",
        lifespan_config.host,
        lifespan_config.port,
    )

    uvicorn.run(
        app=fastapi_app,
        host=lifespan_config.host,
        port=lifespan_config.port,
        reload=False,
        access_log=True,
        log_config=get_uvicorn_log_config(console_handler.level),
        ssl_keyfile=lifespan_config.ssl_keyfile,
        ssl_certfile=lifespan_config.ssl_certfile,
        workers=1,
    )


def _obtain_superlink_certificates(
    args: argparse.Namespace,
) -> tuple[bytes, bytes, bytes] | None:
    """Return TLS certificate tuple used by all APIs (Fleet, Control, and Runtime)."""
    if args.insecure:
        log(
            WARN,
            "Option `--insecure` was set. Starting insecure HTTP server with "
            "unencrypted communication (TLS disabled). Proceed only if you understand "
            "the risks.",
        )
        return None
    return try_obtain_server_certificates(args)


def _get_superexec_command(
    runtime_address: str,
    runtime_certificates: tuple[bytes, bytes, bytes] | None,
    runtime_root_certificates_path: str | None,
    parent_pid: int,
    runtime_dependency_install: bool,
) -> list[str]:
    """Return the auto-launched SuperExec command for ServerApp subprocesses."""
    command = ["flower-superexec"]
    command += get_client_tls_args(
        insecure=runtime_certificates is None,
        root_certificates_path=runtime_root_certificates_path,
    )
    command += ["--runtime-api-address", runtime_address]
    command += ["--parent-pid", str(parent_pid)]
    if runtime_dependency_install:
        # SuperLink subprocess isolation owns this SuperExec, so install dependencies.
        command += ["--allow-runtime-dependency-installation"]
    return command


def _runtime_dependency_install_default() -> bool:
    """Return default runtime dependency installation setting."""
    return os.getenv(FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION) != "1"


def _parse_args_run_superlink() -> argparse.ArgumentParser:
    """Parse command line arguments for both Runtime API and Fleet API."""
    parser = argparse.ArgumentParser(
        description="Start a Flower SuperLink",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"Flower version: {package_version}",
    )

    _add_args_common(parser=parser)
    add_ee_args_superlink(parser=parser)
    _add_args_http_api(parser=parser)
    _add_args_runtime_api(parser=parser)
    _add_args_fleet_api(parser=parser)
    _add_args_control_api(parser=parser)
    add_args_removed_health_server(parser)

    return parser


def _add_args_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Run the server without HTTPS, regardless of whether certificate "
        "paths are provided. Data transmitted between the gRPC client and server "
        "is not encrypted. By default, the server runs with HTTPS enabled. "
        "Use this flag only if you understand the risks.",
    )
    parser.add_argument(
        "--ssl-certfile",
        help="Server TLS certificate file for Fleet, Control, and Runtime APIs "
        "(as a path str) to create secure connections.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ssl-keyfile",
        help="Server TLS private key file for Fleet, Control, and Runtime APIs "
        "(as a path str) to create secure connections.",
        type=str,
    )
    parser.add_argument(
        "--ssl-ca-certfile",
        help="Server TLS CA certificate file for Fleet, Control, and Runtime APIs "
        "(as a path str) to create secure connections.",
        type=str,
    )
    parser.add_argument(
        "--isolation",
        default=ISOLATION_MODE_SUBPROCESS,
        required=False,
        choices=[
            ISOLATION_MODE_SUBPROCESS,
            ISOLATION_MODE_PROCESS,
        ],
        help="Isolation mode when running a `ServerApp` (`subprocess` by default, "
        "possible values: `subprocess`, `process`). Use `subprocess` to configure "
        "SuperLink to run a `ServerApp` in a subprocess. Use `process` to indicate "
        "that a separate independent process gets created outside of SuperLink.",
    )
    parser.add_argument(
        "--database",
        help="A string representing the path to the database "
        "file that will be opened. If nothing is provided, "
        "Flower will just create a state in memory.",
        default=FLWR_IN_MEMORY_DB_NAME,
    )
    parser.add_argument(
        "--auth-list-public-keys",
        type=str,
        help="This argument is deprecated and will be removed in a future release.",
    )
    parser.add_argument(
        "--enable-supernode-auth",
        action="store_true",
        help="Enable supernode authentication.",
    )
    add_args_runtime_dependency_install(
        parser,
        default=_runtime_dependency_install_default(),
        include_disable_flag=True,
        allow_flag_help=(
            "Deprecated. Runtime dependency installation is enabled by "
            "default. Use `--disable-runtime-dependency-installation` to disable it."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to the SuperLink log file. If provided, logs are written to this "
        "file and rotated on a fixed schedule.",
    )
    parser.add_argument(
        "--log-rotation-interval-hours",
        type=_positive_int,
        default=24,
        help="Rotate SuperLink log files every N hours.",
    )
    parser.add_argument(
        "--log-rotation-backup-count",
        type=_positive_int,
        default=7,
        help="Maximum number of rotated SuperLink log files to keep.",
    )
    add_superexec_auth_secret_args(parser)


def _add_args_http_api(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--serverappio-api-address",
        type=_unsupported_runtime_api_address,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--host",
        default=UVICORN_DEFAULT_HOST,
        help=(
            "Host for the Runtime HTTP API. "
            f"By default, it is set to {UVICORN_DEFAULT_HOST}."
        ),
    )
    parser.add_argument(
        "--port",
        type=_port_int,
        default=SUPERLINK_UVICORN_DEFAULT_PORT,
        help=(
            "Port for the Runtime HTTP API. "
            f"By default, it is set to {SUPERLINK_UVICORN_DEFAULT_PORT}."
        ),
    )


def _add_args_runtime_api(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--appio-ssl-certfile",
        dest="runtime_ssl_certfile",
        help=argparse.SUPPRESS,
        type=_unsupported_appio_ssl_flag,
    )
    parser.add_argument(
        "--appio-ssl-keyfile",
        dest="runtime_ssl_keyfile",
        help=argparse.SUPPRESS,
        type=_unsupported_appio_ssl_flag,
    )
    parser.add_argument(
        "--appio-ssl-ca-certfile",
        dest="runtime_ssl_ca_certfile",
        help=argparse.SUPPRESS,
        type=_unsupported_appio_ssl_flag,
    )


def _unsupported_appio_ssl_flag(_value: str) -> str:
    """Reject a removed --appio-ssl-* flag."""
    raise argparse.ArgumentTypeError(
        "this flag no longer exists; Control API, Fleet API, and "
        "Runtime API use the same TLS certificates. Use `--ssl-certfile`, "
        "`--ssl-keyfile`, and `--ssl-ca-certfile` instead."
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _port_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 65535:
        raise argparse.ArgumentTypeError("value must be between 1 and 65535")
    return parsed


def _unsupported_runtime_api_address(_value: str) -> str:
    """Reject the removed combined Runtime API address option."""
    raise argparse.ArgumentTypeError(
        "this option is no longer supported; use `--host` and `--port` instead"
    )


def _add_args_fleet_api(parser: argparse.ArgumentParser) -> None:
    # Fleet API transport layer type
    parser.add_argument(
        "--fleet-api-type",
        default="grpc-rere",
        type=str,
        choices=[
            "grpc-rere",
            "grpc-adapter",
        ],
        help="Deprecated. The Fleet API now operates over HTTP.",
    )
    parser.add_argument(
        "--fleet-api-address",
        help="Deprecated. Use `--host` and `--port` for the Fleet HTTP API.",
    )


def _add_args_control_api(parser: argparse.ArgumentParser) -> None:
    """Add command line arguments for Control API."""
    parser.add_argument(
        "--control-api-address",
        help="Deprecated. The Control API now operates over HTTP. Use `--host` and "
        "`--port` instead.",
        default=CONTROL_API_DEFAULT_SERVER_ADDRESS,
    )
    parser.add_argument(  # To be removed in follow-up PRs
        "--simulation",
        action="store_true",
        default=False,
        help="Enable simulation runtime behavior.",
    )
