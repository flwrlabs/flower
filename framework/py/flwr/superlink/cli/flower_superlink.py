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
from collections.abc import Sequence
from logging import INFO, WARN
from pathlib import Path
from time import sleep
from typing import cast

import grpc
import uvicorn

from flwr.common.args import (
    add_args_runtime_dependency_install,
    try_obtain_server_certificates,
)
from flwr.common.constant import (
    CONTROL_API_DEFAULT_SERVER_ADDRESS,
    FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
    FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION,
    ISOLATION_MODE_PROCESS,
    ISOLATION_MODE_SUBPROCESS,
    TRANSPORT_TYPE_GRPC_ADAPTER,
    TRANSPORT_TYPE_GRPC_RERE,
    EventLogWriterType,
    ExecPluginType,
)
from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.proto.fleet_pb2_grpc import (  # pylint: disable=E0611
    add_FleetServicer_to_server,
)
from flwr.proto.grpcadapter_pb2_grpc import add_GrpcAdapterServicer_to_server
from flwr.server.fleet_event_log_interceptor import FleetEventLogInterceptor
from flwr.server.superlink.fleet.grpc_adapter.grpc_adapter_servicer import (
    GrpcAdapterServicer,
)
from flwr.server.superlink.fleet.grpc_rere.fleet_servicer import FleetServicer
from flwr.server.superlink.fleet.grpc_rere.node_auth_server_interceptor import (
    NodeAuthServerInterceptor,
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
    UVICORN_DEFAULT_HOST,
    UVICORN_DEFAULT_PORT,
)
from flwr.supercore.exit import ExitCode, flwr_exit
from flwr.supercore.grpc import GRPC_MAX_MESSAGE_LENGTH, generic_create_grpc_server
from flwr.supercore.grpc_health import add_args_health, run_health_server_grpc_no_tls
from flwr.supercore.http_logging import get_uvicorn_log_config
from flwr.supercore.interceptors import (
    RpcErrorTranslationServerInterceptor,
    create_fleet_runtime_version_server_interceptor,
)
from flwr.supercore.logger import configure_superlink_log_file, console_handler
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.telemetry import EventType, event
from flwr.supercore.tls import (
    get_client_tls_args,
    try_obtain_optional_runtime_server_certificates,
)
from flwr.supercore.update_check import warn_if_flwr_update_available
from flwr.supercore.utils import get_popen_detach_kwargs
from flwr.supercore.version import package_version
from flwr.superlink.artifact_provider import ArtifactProvider
from flwr.superlink.config_loader import (
    SuperLinkLifespanConfig,
    load_control_authn_plugin,
    load_control_event_log_plugin,
)
from flwr.superlink.servicer.control import run_control_api_grpc

try:
    from flwr.ee import (
        add_ee_args_superlink,
        get_ee_artifact_provider,
        get_fleet_event_log_writer_plugins,
    )
except ImportError:

    # pylint: disable-next=unused-argument
    def add_ee_args_superlink(parser: argparse.ArgumentParser) -> None:
        """Add EE-specific arguments to the parser."""

    def get_ee_artifact_provider(config_path: str) -> ArtifactProvider:
        """Return the EE artifact provider."""
        raise NotImplementedError("No artifact provider is currently supported.")

    def get_fleet_event_log_writer_plugins() -> dict[str, type[EventLogWriterPlugin]]:
        """Return all Fleet API event log writer plugins."""
        raise NotImplementedError(
            "No event log writer plugins are currently supported."
        )


class SuperLinkLifespan:  # pylint: disable=too-many-instance-attributes
    """Own the shared SuperLink lifespan state and legacy network servers.

    Long-term, the gRPC-specific parts of this class should shrink until it only
    initializes shared services used by FastAPI routers. During the migration,
    FastAPI lifespan can use this object to start the existing gRPC APIs as
    compatibility adapters.
    """

    def __init__(
        self,
        config: SuperLinkLifespanConfig,
        state_factory: LinkStateFactory,
    ) -> None:
        self.config = config
        self.grpc_servers: list[grpc.Server] = []
        self.bckg_threads: list[threading.Thread] = []
        self.superexec_process: subprocess.Popen[bytes] | None = None
        self.objectstore_factory = state_factory.objectstore_factory
        self.state_factory = state_factory
        self._started = False

    def startup(self) -> None:
        """Start shared lifespan and legacy SuperLink gRPC servers."""
        log(INFO, "SuperLinkLifespan: start")
        if self._started:
            return

        # Force initialization before starting network servers
        self.state_factory.state()

        self._start_control_api()
        self._start_fleet_api()
        self._start_superexec_if_needed()
        self._start_health_server_if_needed()
        self._started = True

    def shutdown(self) -> None:
        """Stop legacy gRPC servers started by this lifespan."""
        log(INFO, "SuperLinkLifespan: stop")
        if (
            not self._started
            and not self.grpc_servers
            and self.superexec_process is None
        ):
            return

        # Stop in reverse startup order so dependent services disappear before
        # their backing state is considered unavailable.
        for grpc_server in reversed(self.grpc_servers):
            grpc_server.stop(grace=1)

        if self.superexec_process is not None:
            self.superexec_process.terminate()
            try:
                self.superexec_process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                log(WARN, "SuperExec subprocess did not terminate within 1 second.")

        self.grpc_servers.clear()
        self.superexec_process = None
        self._started = False

    def wait_until_background_thread_exits(self) -> None:
        """Block like the historical `flower-superlink` command.

        With only gRPC servers, `self.bckg_threads` is empty and `all([])` is
        intentionally true, so this loop blocks until a signal handler exits the
        process. This preserves the current CLI behavior.
        """
        while all(thread.is_alive() for thread in self.bckg_threads):
            sleep(0.1)

    def _start_control_api(self) -> None:
        config = self.config
        control_server: grpc.Server = run_control_api_grpc(
            address=config.control_address,
            state_factory=self.state_factory,
            objectstore_factory=self.objectstore_factory,
            certificates=config.certificates,
            authn_plugin=config.authn_plugin,
            event_log_plugin=config.event_log_plugin,
            artifact_provider=config.artifact_provider,
            fleet_api_type=config.fleet_api_type,
        )
        self.grpc_servers.append(control_server)

    def _start_fleet_api(self) -> None:
        config = self.config
        if config.simulation:
            return

        fleet_api_address = config.fleet_api_address
        if not fleet_api_address:
            if config.fleet_api_type in [
                TRANSPORT_TYPE_GRPC_RERE,
                TRANSPORT_TYPE_GRPC_ADAPTER,
            ]:
                fleet_api_address = FLEET_API_GRPC_RERE_DEFAULT_ADDRESS

        fleet_address, _, _ = _format_address(cast(str, fleet_api_address))
        if config.fleet_api_type == TRANSPORT_TYPE_GRPC_RERE:
            self._start_legacy_fleet_grpc_rere(fleet_address)
        elif config.fleet_api_type == TRANSPORT_TYPE_GRPC_ADAPTER:
            self._start_legacy_fleet_grpc_adapter(fleet_address)
        else:
            raise ValueError(f"Unknown fleet_api_type: {config.fleet_api_type}")

    def _start_legacy_fleet_grpc_rere(self, fleet_address: str) -> None:
        """Start the current Fleet gRPC request-response API."""
        interceptors = [NodeAuthServerInterceptor(self.state_factory)]
        if self.config.enable_event_log:
            fleet_log_plugin = _try_obtain_fleet_event_log_writer_plugin()
            if fleet_log_plugin is not None:
                interceptors.append(FleetEventLogInterceptor(fleet_log_plugin))
                log(INFO, "Flower Fleet event logging enabled")

        fleet_server = _run_fleet_api_grpc_rere(
            address=fleet_address,
            state_factory=self.state_factory,
            objectstore_factory=self.objectstore_factory,
            enable_supernode_auth=self.config.enable_supernode_auth,
            certificates=self.config.certificates,
            interceptors=interceptors,
        )
        self.grpc_servers.append(fleet_server)

    def _start_legacy_fleet_grpc_adapter(self, fleet_address: str) -> None:
        """Start the current Fleet GrpcAdapter compatibility API."""
        fleet_server = _run_fleet_api_grpc_adapter(
            address=fleet_address,
            state_factory=self.state_factory,
            objectstore_factory=self.objectstore_factory,
            certificates=self.config.certificates,
        )
        self.grpc_servers.append(fleet_server)

    def _start_superexec_if_needed(self) -> None:
        config = self.config
        if config.isolation != ISOLATION_MODE_SUBPROCESS:
            return

        runtime_host = f"[{config.host}]" if ":" in config.host else config.host
        runtime_address = resolve_bind_address(f"{runtime_host}:{config.port}")
        command = _get_superexec_command(
            runtime_address=runtime_address,
            runtime_certificates=config.runtime_certificates,
            runtime_root_certificates_path=config.runtime_ssl_ca_certfile,
            parent_pid=os.getpid(),
            runtime_dependency_install=config.runtime_dependency_install,
        )
        # pylint: disable-next=consider-using-with
        self.superexec_process = subprocess.Popen(command, **get_popen_detach_kwargs())

    def _start_health_server_if_needed(self) -> None:
        if self.config.health_server_address is None:
            return

        health_server = run_health_server_grpc_no_tls(self.config.health_server_address)
        self.grpc_servers.append(health_server)


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

    # Detect if `--executor*` arguments were set
    if args.executor or args.executor_dir or args.executor_config:
        flwr_exit(
            ExitCode.SUPERLINK_INVALID_ARGS,
            "The arguments `--executor`, `--executor-dir`, and `--executor-config` are "
            "deprecated and will be removed in a future release. To run SuperLink with "
            "the simulation runtime, please use `--simulation`.",
        )

    # Detect if both Control API and Exec API addresses were set explicitly
    explicit_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            explicit_args.add(
                arg.split("=")[0]
            )  # handles both `--arg val` and `--arg=val`

    # The old opt-in flag is accepted for compatibility, but no longer needed.
    if "--allow-runtime-dependency-installation" in explicit_args:
        log(
            WARN,
            "The `--allow-runtime-dependency-installation` argument is deprecated. "
            "Runtime dependency installation is enabled by default for SuperLink. "
            "Use `--disable-runtime-dependency-installation` to disable it.",
        )

    control_api_set = "--control-api-address" in explicit_args
    exec_api_set = "--exec-api-address" in explicit_args

    if control_api_set and exec_api_set:
        flwr_exit(
            ExitCode.SUPERLINK_INVALID_ARGS,
            "Both `--control-api-address` and `--exec-api-address` are set. "
            "Please use only `--control-api-address` as `--exec-api-address` is "
            "deprecated.",
        )

    # Warn deprecated `--exec-api-address` argument
    if args.exec_api_address is not None:
        log(
            WARN,
            "The `--exec-api-address` argument is deprecated and will be removed in a "
            "future release. Use `--control-api-address` instead.",
        )
        args.control_api_address = args.exec_api_address

    # Parse IP addresses
    control_address, _, _ = _format_address(args.control_api_address)
    health_server_address = None
    if args.health_server_address is not None:
        health_server_address, _, _ = _format_address(args.health_server_address)

    # Obtain certificates
    certificates, runtime_certificates = _obtain_superlink_certificates(args)

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
        if args.fleet_api_type != TRANSPORT_TYPE_GRPC_RERE:
            flwr_exit(
                ExitCode.SUPERLINK_INVALID_ARGS,
                "The `--enable-supernode-auth` flag is only supported "
                "with the gRPC-rere Fleet API transport. Please set "
                f"`--fleet-api-type` to `{TRANSPORT_TYPE_GRPC_RERE}`.",
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

    fleet_api_address = args.fleet_api_address
    if not args.simulation and not fleet_api_address:
        if args.fleet_api_type in [
            TRANSPORT_TYPE_GRPC_RERE,
            TRANSPORT_TYPE_GRPC_ADAPTER,
        ]:
            fleet_api_address = FLEET_API_GRPC_RERE_DEFAULT_ADDRESS

    return SuperLinkLifespanConfig(
        control_address=control_address,
        health_server_address=health_server_address,
        host=args.host,
        port=args.port,
        insecure=args.insecure,
        certificates=certificates,
        runtime_certificates=runtime_certificates,
        superexec_auth_secret=superexec_auth_secret,
        authn_plugin=authn_plugin,
        event_log_plugin=event_log_plugin,
        enable_event_log=getattr(args, "enable_event_log", False),
        artifact_provider=artifact_provider,
        enable_supernode_auth=enable_supernode_auth,
        fleet_api_type=args.fleet_api_type,
        fleet_api_address=fleet_api_address,
        simulation=args.simulation,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        database=args.database,
        isolation=args.isolation,
        runtime_ssl_ca_certfile=args.runtime_ssl_ca_certfile,
        runtime_ssl_certfile=(
            str(Path(args.runtime_ssl_certfile).expanduser())
            if runtime_certificates is not None
            else None
        ),
        runtime_ssl_keyfile=(
            str(Path(args.runtime_ssl_keyfile).expanduser())
            if runtime_certificates is not None
            else None
        ),
        runtime_dependency_install=args.runtime_dependency_install,
    )


def flower_superlink() -> None:
    """Run Flower SuperLink (Runtime API and Fleet API)."""
    warn_if_flwr_update_available(process_name="flower-superlink")

    config = _parse_superlink_lifespan_config()

    log(INFO, "Starting Flower SuperLink")

    event(EventType.RUN_SUPERLINK_ENTER)

    # Blocking: FastAPI serves Runtime HTTP while its lifespan owns Control and
    # Fleet gRPC servers.
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

    # Uvicorn workers must stay at 1 while the lifespan starts gRPC servers. With
    # multiple workers, every worker process would try to bind the same Control,
    # Fleet and Runtime API ports.
    uvicorn.run(
        app=fastapi_app,
        host=lifespan_config.host,
        port=lifespan_config.port,
        reload=False,
        access_log=True,
        log_config=get_uvicorn_log_config(console_handler.level),
        ssl_keyfile=lifespan_config.runtime_ssl_keyfile,
        ssl_certfile=lifespan_config.runtime_ssl_certfile,
        workers=1,
    )


def _obtain_superlink_certificates(
    args: argparse.Namespace,
) -> tuple[tuple[bytes, bytes, bytes] | None, tuple[bytes, bytes, bytes] | None]:
    """Return Fleet/Control and Runtime API certificate tuples."""
    if args.insecure:
        log(
            WARN,
            "Option `--insecure` was set. Starting insecure HTTP server with "
            "unencrypted communication (TLS disabled). Proceed only if you understand "
            "the risks.",
        )
        return None, None
    certificates = try_obtain_server_certificates(args)
    runtime_certificates = try_obtain_optional_runtime_server_certificates(args)
    return certificates, runtime_certificates


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
    command += ["--appio-api-address", runtime_address]
    command += ["--plugin-type", ExecPluginType.SERVER_APP]
    command += ["--parent-pid", str(parent_pid)]
    if runtime_dependency_install:
        # SuperLink subprocess isolation owns this SuperExec, so install dependencies.
        command += ["--allow-runtime-dependency-installation"]
    return command


def _runtime_dependency_install_default() -> bool:
    """Return default runtime dependency installation setting."""
    return os.getenv(FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION) != "1"


def _try_obtain_fleet_event_log_writer_plugin() -> EventLogWriterPlugin | None:
    """Return an instance of the Fleet Servicer event log writer plugin."""
    try:
        all_plugins: dict[str, type[EventLogWriterPlugin]] = (
            get_fleet_event_log_writer_plugins()
        )
        plugin_class = all_plugins[EventLogWriterType.STDOUT]
        return plugin_class()
    except KeyError:
        sys.exit("No Fleet API event log writer plugin is provided.")
    except NotImplementedError:
        sys.exit("No Fleet API event log writer plugins are currently supported.")


def _run_fleet_api_grpc_rere(  # pylint: disable=R0913, R0917
    address: str,
    state_factory: LinkStateFactory,
    objectstore_factory: ObjectStoreFactory,
    enable_supernode_auth: bool,
    certificates: tuple[bytes, bytes, bytes] | None,
    interceptors: Sequence[grpc.ServerInterceptor] | None = None,
) -> grpc.Server:
    """Run Fleet API (gRPC, request-response)."""
    interceptors = [
        RpcErrorTranslationServerInterceptor(),
        *list(interceptors or []),
    ]
    interceptors.append(create_fleet_runtime_version_server_interceptor())

    # Create Fleet API gRPC server
    fleet_servicer = FleetServicer(
        state_factory=state_factory,
        objectstore_factory=objectstore_factory,
        enable_supernode_auth=enable_supernode_auth,
    )
    fleet_add_servicer_to_server_fn = add_FleetServicer_to_server
    fleet_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(fleet_servicer, fleet_add_servicer_to_server_fn),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
        interceptors=interceptors,
    )

    log(
        INFO,
        "Flower Deployment Runtime: Starting Fleet API (gRPC-rere) on %s",
        fleet_grpc_server.bound_address,
    )
    fleet_grpc_server.start()

    return fleet_grpc_server


# pylint: disable=R0913, R0917
def _run_fleet_api_grpc_adapter(
    address: str,
    state_factory: LinkStateFactory,
    objectstore_factory: ObjectStoreFactory,
    certificates: tuple[bytes, bytes, bytes] | None,
) -> grpc.Server:
    """Run Fleet API (GrpcAdapter)."""
    # Create Fleet API gRPC server
    fleet_servicer = GrpcAdapterServicer(
        state_factory=state_factory,
        objectstore_factory=objectstore_factory,
        enable_supernode_auth=False,
    )
    fleet_add_servicer_to_server_fn = add_GrpcAdapterServicer_to_server
    fleet_grpc_server = generic_create_grpc_server(
        servicer_and_add_fn=(fleet_servicer, fleet_add_servicer_to_server_fn),
        server_address=address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        certificates=certificates,
    )

    log(
        INFO,
        "Flower Deployment Runtime: Starting Fleet API (GrpcAdapter) on %s",
        fleet_grpc_server.bound_address,
    )
    fleet_grpc_server.start()

    return fleet_grpc_server


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
    add_args_health(parser=parser)

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
        help="Server TLS certificate file for Fleet API and Control API "
        "(as a path str) to create a secure connection.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ssl-keyfile",
        help="Server TLS private key file for Fleet API and Control API "
        "(as a path str) to create a secure connection.",
        type=str,
    )
    parser.add_argument(
        "--ssl-ca-certfile",
        help="Server TLS CA certificate file for Fleet API and Control API "
        "(as a path str) to create a secure connection.",
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
        default=UVICORN_DEFAULT_PORT,
        help=(
            "Port for the Runtime HTTP API. "
            f"By default, it is set to {UVICORN_DEFAULT_PORT}."
        ),
    )


def _add_args_runtime_api(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--appio-ssl-certfile",
        dest="runtime_ssl_certfile",
        help="Runtime API server TLS certificate file (as a path str) "
        "to create a secure connection. The certificate must include SANs for "
        "the Runtime API address used by SuperExec.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--appio-ssl-keyfile",
        dest="runtime_ssl_keyfile",
        help="Runtime API server TLS private key file (as a path str) "
        "to create a secure connection.",
        type=str,
    )
    parser.add_argument(
        "--appio-ssl-ca-certfile",
        dest="runtime_ssl_ca_certfile",
        help="Path to the PEM-encoded CA certificate file used by SuperExec to verify "
        "the Runtime API server certificate. This is not a client certificate "
        "for mTLS.",
        type=str,
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


def _add_args_fleet_api(parser: argparse.ArgumentParser) -> None:
    # Fleet API transport layer type
    parser.add_argument(
        "--fleet-api-type",
        default=TRANSPORT_TYPE_GRPC_RERE,
        type=str,
        choices=[
            TRANSPORT_TYPE_GRPC_RERE,
            TRANSPORT_TYPE_GRPC_ADAPTER,
        ],
        help="Start a Fleet API server.",
    )
    parser.add_argument(
        "--fleet-api-address",
        help="Fleet API server address (IPv4, IPv6, or a domain name).",
    )


def _add_args_control_api(parser: argparse.ArgumentParser) -> None:
    """Add command line arguments for Control API."""
    parser.add_argument(
        "--control-api-address",
        help="Control API server address (IPv4, IPv6, or a domain name) "
        f"By default, it is set to {CONTROL_API_DEFAULT_SERVER_ADDRESS}.",
        default=CONTROL_API_DEFAULT_SERVER_ADDRESS,
    )
    parser.add_argument(
        "--exec-api-address",
        help="This argument is deprecated and will be removed in a future release. "
        "Use `--control-api-address` instead.",
        default=None,
    )
    parser.add_argument(
        "--executor",
        help="This argument is deprecated and will be removed in a future release.",
        default=None,
    )
    parser.add_argument(
        "--executor-dir",
        help="This argument is deprecated and will be removed in a future release.",
        default=None,
    )
    parser.add_argument(
        "--executor-config",
        help="This argument is deprecated and will be removed in a future release.",
        default=None,
    )
    parser.add_argument(  # To be removed in follow-up PRs
        "--simulation",
        action="store_true",
        default=False,
        help="Enable simulation runtime behavior.",
    )
