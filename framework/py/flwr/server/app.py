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
"""Flower server app."""


import argparse
import importlib.util
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from logging import INFO, WARN
from pathlib import Path
from time import sleep
from typing import TypeVar, cast

import grpc
import yaml

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, EventType, event
from flwr.common.args import (
    add_args_runtime_dependency_install,
    try_obtain_server_certificates,
)
from flwr.common.constant import (
    AUTHN_TYPE_YAML_KEY,
    AUTHZ_TYPE_YAML_KEY,
    CONTROL_API_DEFAULT_SERVER_ADDRESS,
    FLEET_API_GRPC_RERE_DEFAULT_ADDRESS,
    FLEET_API_REST_DEFAULT_ADDRESS,
    ISOLATION_MODE_PROCESS,
    ISOLATION_MODE_SUBPROCESS,
    SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
    TRANSPORT_TYPE_GRPC_ADAPTER,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_REST,
    AuthnType,
    AuthzType,
    EventLogWriterType,
    ExecPluginType,
)
from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.common.exit import ExitCode, flwr_exit, register_signal_handlers
from flwr.common.grpc import generic_create_grpc_server
from flwr.common.logger import configure_superlink_log_file, log
from flwr.proto.fleet_pb2_grpc import (  # pylint: disable=E0611
    add_FleetServicer_to_server,
)
from flwr.proto.grpcadapter_pb2_grpc import add_GrpcAdapterServicer_to_server
from flwr.server.fleet_event_log_interceptor import FleetEventLogInterceptor
from flwr.supercore.address import parse_address, resolve_bind_address
from flwr.supercore.auth import (
    add_superexec_auth_secret_args,
    load_superexec_auth_secret,
)
from flwr.supercore.constant import FLWR_IN_MEMORY_DB_NAME
from flwr.supercore.grpc_health import add_args_health, run_health_server_grpc_no_tls
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.update_check import warn_if_flwr_update_available
from flwr.supercore.version import package_version
from flwr.superlink.artifact_provider import ArtifactProvider
from flwr.superlink.auth_plugin import (
    ControlAuthnPlugin,
    ControlAuthzPlugin,
    NoOpControlAuthnPlugin,
    NoOpControlAuthzPlugin,
)
from flwr.superlink.federation import FederationManager, NoOpFederationManager
from flwr.superlink.servicer.control import run_control_api_grpc

from .superlink.fleet.grpc_adapter.grpc_adapter_servicer import GrpcAdapterServicer
from .superlink.fleet.grpc_rere.fleet_servicer import FleetServicer
from .superlink.fleet.grpc_rere.node_auth_server_interceptor import (
    NodeAuthServerInterceptor,
)
from .superlink.linkstate import LinkStateFactory
from .superlink.serverappio.serverappio_grpc import run_serverappio_api_grpc

P = TypeVar("P", ControlAuthnPlugin, ControlAuthzPlugin)


try:
    from flwr.ee import (
        add_ee_args_superlink,
        get_control_authn_ee_plugins,
        get_control_authz_ee_plugins,
        get_control_event_log_writer_plugins,
        get_ee_artifact_provider,
        get_ee_federation_manager,
        get_fleet_event_log_writer_plugins,
    )
except ImportError:

    # pylint: disable-next=unused-argument
    def add_ee_args_superlink(parser: argparse.ArgumentParser) -> None:
        """Add EE-specific arguments to the parser."""

    def get_control_event_log_writer_plugins() -> dict[str, type[EventLogWriterPlugin]]:
        """Return all Control API event log writer plugins."""
        raise NotImplementedError(
            "No event log writer plugins are currently supported."
        )

    def get_ee_artifact_provider(config_path: str) -> ArtifactProvider:
        """Return the EE artifact provider."""
        raise NotImplementedError("No artifact provider is currently supported.")

    def get_fleet_event_log_writer_plugins() -> dict[str, type[EventLogWriterPlugin]]:
        """Return all Fleet API event log writer plugins."""
        raise NotImplementedError(
            "No event log writer plugins are currently supported."
        )

    def get_control_authn_ee_plugins() -> dict[str, type[ControlAuthnPlugin]]:
        """Return all Control API authentication plugins for EE."""
        return {}

    def get_control_authz_ee_plugins() -> dict[str, type[ControlAuthzPlugin]]:
        """Return all Control API authorization plugins for EE."""
        return {}

    def get_ee_federation_manager() -> FederationManager:
        """Return the EE FederationManager."""
        raise NotImplementedError("No federation manager is currently supported.")


def get_control_authn_plugins() -> dict[str, type[ControlAuthnPlugin]]:
    """Return all Control API authentication plugins."""
    ee_dict: dict[str, type[ControlAuthnPlugin]] = get_control_authn_ee_plugins()
    return ee_dict | {AuthnType.NOOP: NoOpControlAuthnPlugin}


def get_control_authz_plugins() -> dict[str, type[ControlAuthzPlugin]]:
    """Return all Control API authorization plugins."""
    ee_dict: dict[str, type[ControlAuthzPlugin]] = get_control_authz_ee_plugins()
    return ee_dict | {AuthzType.NOOP: NoOpControlAuthzPlugin}


def get_federation_manager(is_simulation: bool = False) -> FederationManager:
    """Return the FederationManager."""
    try:
        federation_manager: FederationManager = get_ee_federation_manager()
        return federation_manager
    except NotImplementedError:
        return NoOpFederationManager(simulation=is_simulation)


# pylint: disable=too-many-branches, too-many-locals, too-many-statements
def run_superlink() -> None:
    """Run Flower SuperLink (ServerAppIo API and Fleet API)."""
    warn_if_flwr_update_available(process_name="flower-superlink")

    args = _parse_args_run_superlink().parse_args()

    if args.log_file:
        configure_superlink_log_file(
            filename=args.log_file,
            interval_hours=args.log_rotation_interval_hours,
            backup_count=args.log_rotation_backup_count,
        )

    log(INFO, "Starting Flower SuperLink")

    event(EventType.RUN_SUPERLINK_ENTER)

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
    serverappio_address, _, _ = _format_address(args.serverappio_api_address)
    control_address, _, _ = _format_address(args.control_api_address)
    health_server_address = None
    if args.health_server_address is not None:
        health_server_address, _, _ = _format_address(args.health_server_address)

    # Obtain certificates
    certificates = try_obtain_server_certificates(args)

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

    # Disable the account auth TLS check if args.disable_oidc_tls_cert_verification is
    # provided
    verify_tls_cert = not getattr(args, "disable_oidc_tls_cert_verification", None)

    authn_plugin: ControlAuthnPlugin | None = None
    authz_plugin: ControlAuthzPlugin | None = None
    event_log_plugin: EventLogWriterPlugin | None = None
    # Load the auth plugin if the args.account_auth_config is provided
    if cfg_path := getattr(args, "user_auth_config", None):
        log(
            WARN,
            "The `--user-auth-config` flag is deprecated and will be removed in a "
            "future release. Please use `--account-auth-config` instead.",
        )
        args.account_auth_config = cfg_path
    cfg_path = getattr(args, "account_auth_config", None)
    authn_plugin, authz_plugin = _load_control_auth_plugins(cfg_path, verify_tls_cert)
    if cfg_path is not None:
        # Enable event logging if the args.enable_event_log is True
        if args.enable_event_log:
            event_log_plugin = _try_obtain_control_event_log_writer_plugin()

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

    # Load Federation Manager
    federation_manager = get_federation_manager(is_simulation=args.simulation)

    # Initialize ObjectStoreFactory
    objectstore_factory = ObjectStoreFactory(args.database)

    # Initialize StateFactory
    state_factory = LinkStateFactory(
        args.database, federation_manager, objectstore_factory
    )
    state_factory.state()  # Force initialization before starting servers

    # Start Control API
    is_simulation = args.simulation
    control_server: grpc.Server = run_control_api_grpc(
        address=control_address,
        state_factory=state_factory,
        objectstore_factory=objectstore_factory,
        certificates=certificates,
        authn_plugin=authn_plugin,
        authz_plugin=authz_plugin,
        event_log_plugin=event_log_plugin,
        artifact_provider=artifact_provider,
        fleet_api_type=args.fleet_api_type,
    )
    grpc_servers = [control_server]
    bckg_threads: list[threading.Thread] = []

    # Start ServerAppIo API for both deployment and simulation runtimes.
    serverappio_server: grpc.Server = run_serverappio_api_grpc(
        address=serverappio_address,
        state_factory=state_factory,
        objectstore_factory=objectstore_factory,
        certificates=None,  # ServerAppIo API doesn't support SSL yet
        superexec_auth_secret=superexec_auth_secret,
    )
    grpc_servers.append(serverappio_server)

    # Start Fleet API
    if not is_simulation:
        if not args.fleet_api_address:
            if args.fleet_api_type in [
                TRANSPORT_TYPE_GRPC_RERE,
                TRANSPORT_TYPE_GRPC_ADAPTER,
            ]:
                args.fleet_api_address = FLEET_API_GRPC_RERE_DEFAULT_ADDRESS
            elif args.fleet_api_type == TRANSPORT_TYPE_REST:
                args.fleet_api_address = FLEET_API_REST_DEFAULT_ADDRESS

        fleet_address, host, port = _format_address(args.fleet_api_address)

        num_workers = args.fleet_api_num_workers
        if num_workers != 1:
            log(
                WARN,
                "The Fleet API currently supports only 1 worker. "
                "You have specified %d workers. "
                "Support for multiple workers will be added in future releases. "
                "Proceeding with a single worker.",
                args.fleet_api_num_workers,
            )
            num_workers = 1

        if args.fleet_api_type == TRANSPORT_TYPE_REST:
            if (
                importlib.util.find_spec("requests")
                and importlib.util.find_spec("starlette")
                and importlib.util.find_spec("uvicorn")
            ) is None:
                flwr_exit(ExitCode.COMMON_MISSING_EXTRA_REST)

            fleet_thread = threading.Thread(
                target=_run_fleet_api_rest,
                args=(
                    host,
                    port,
                    args.ssl_keyfile,
                    args.ssl_certfile,
                    state_factory,
                    objectstore_factory,
                    num_workers,
                ),
                daemon=True,
            )
            fleet_thread.start()
            bckg_threads.append(fleet_thread)
        elif args.fleet_api_type == TRANSPORT_TYPE_GRPC_RERE:

            interceptors = [NodeAuthServerInterceptor(state_factory)]
            if getattr(args, "enable_event_log", None):
                fleet_log_plugin = _try_obtain_fleet_event_log_writer_plugin()
                if fleet_log_plugin is not None:
                    interceptors.append(FleetEventLogInterceptor(fleet_log_plugin))
                    log(INFO, "Flower Fleet event logging enabled")

            fleet_server = _run_fleet_api_grpc_rere(
                address=fleet_address,
                state_factory=state_factory,
                objectstore_factory=objectstore_factory,
                enable_supernode_auth=enable_supernode_auth,
                certificates=certificates,
                interceptors=interceptors,
            )
            grpc_servers.append(fleet_server)
        elif args.fleet_api_type == TRANSPORT_TYPE_GRPC_ADAPTER:
            fleet_server = _run_fleet_api_grpc_adapter(
                address=fleet_address,
                state_factory=state_factory,
                objectstore_factory=objectstore_factory,
                certificates=certificates,
            )
            grpc_servers.append(fleet_server)
        else:
            raise ValueError(f"Unknown fleet_api_type: {args.fleet_api_type}")

    # Launch SuperExec if isolation mode is subprocess
    if args.isolation == ISOLATION_MODE_SUBPROCESS:
        # bound_address contains the actual address when the port is set to :0
        # which means let the OS choose a free port.
        appio_address = resolve_bind_address(serverappio_server.bound_address)
        command = ["flower-superexec", "--insecure"]
        command += ["--appio-api-address", appio_address]
        command += ["--plugin-type", ExecPluginType.SERVER_APP]
        command += ["--parent-pid", str(os.getpid())]
        if args.runtime_dependency_install:
            command += ["--allow-runtime-dependency-installation"]
        # pylint: disable-next=consider-using-with
        subprocess.Popen(command)

    # Launch gRPC health server
    if health_server_address is not None:
        health_server = run_health_server_grpc_no_tls(health_server_address)
        grpc_servers.append(health_server)

    # Graceful shutdown
    register_signal_handlers(
        event_type=EventType.RUN_SUPERLINK_LEAVE,
        exit_message="SuperLink terminated gracefully.",
        grpc_servers=grpc_servers,
    )

    # Block until a thread exits prematurely
    while all(thread.is_alive() for thread in bckg_threads):
        sleep(0.1)

    # Exit if any thread has exited prematurely
    # This code will not be reached if the SuperLink stops gracefully
    flwr_exit(ExitCode.SUPERLINK_THREAD_CRASH)


def _format_address(address: str) -> tuple[str, str, int]:
    parsed_address = parse_address(address)
    if not parsed_address:
        flwr_exit(
            ExitCode.COMMON_ADDRESS_INVALID,
            f"Address ({address}) cannot be parsed.",
        )
    host, port, is_v6 = parsed_address
    return (f"[{host}]:{port}" if is_v6 else f"{host}:{port}", host, port)


def _load_control_auth_plugins(
    config_path: str | None, verify_tls_cert: bool
) -> tuple[ControlAuthnPlugin, ControlAuthzPlugin]:
    """Obtain Control API authentication and authorization plugins."""
    # Load NoOp plugins if no config path is provided
    if config_path is None:
        config_path = ""
        config = {
            "authentication": {AUTHN_TYPE_YAML_KEY: AuthnType.NOOP},
            "authorization": {AUTHZ_TYPE_YAML_KEY: AuthzType.NOOP},
        }
    # Load YAML file
    else:
        with Path(config_path).expanduser().open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

    def _load_plugin(
        section: str, yaml_key: str, loader: Callable[[], dict[str, type[P]]]
    ) -> P:
        section_cfg = config.get(section, {})
        auth_plugin_name = section_cfg.get(yaml_key, "")
        try:
            plugins: dict[str, type[P]] = loader()
            plugin_cls: type[P] = plugins[auth_plugin_name]
            return plugin_cls(Path(cast(str, config_path)), verify_tls_cert)
        except KeyError:
            if auth_plugin_name:
                sys.exit(
                    f"{yaml_key}: {auth_plugin_name} is not supported. "
                    f"Please provide a valid {section} type in the configuration."
                )
            sys.exit(f"No {section} type is provided in the configuration.")

    # Warn deprecated auth_type key
    if authn_type := config["authentication"].pop("auth_type", None):
        log(
            WARN,
            "The `auth_type` key in the authentication configuration is deprecated. "
            "Use `%s` instead.",
            AUTHN_TYPE_YAML_KEY,
        )
        config["authentication"][AUTHN_TYPE_YAML_KEY] = authn_type

    # Load authentication plugin
    authn_plugin = _load_plugin(
        section="authentication",
        yaml_key=AUTHN_TYPE_YAML_KEY,
        loader=get_control_authn_plugins,
    )

    # Load authorization plugin
    authz_plugin = _load_plugin(
        section="authorization",
        yaml_key=AUTHZ_TYPE_YAML_KEY,
        loader=get_control_authz_plugins,
    )

    return authn_plugin, authz_plugin


def _try_obtain_control_event_log_writer_plugin() -> EventLogWriterPlugin | None:
    """Return an instance of the event log writer plugin."""
    try:
        all_plugins: dict[str, type[EventLogWriterPlugin]] = (
            get_control_event_log_writer_plugins()
        )
        plugin_class = all_plugins[EventLogWriterType.STDOUT]
        return plugin_class()
    except KeyError:
        sys.exit("No event log writer plugin is provided.")
    except NotImplementedError:
        sys.exit("No event log writer plugins are currently supported.")


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


# pylint: disable=import-outside-toplevel,too-many-arguments
# pylint: disable=too-many-positional-arguments
def _run_fleet_api_rest(
    host: str,
    port: int,
    ssl_keyfile: str | None,
    ssl_certfile: str | None,
    state_factory: LinkStateFactory,
    objectstore_factory: ObjectStoreFactory,
    num_workers: int,
) -> None:
    """Run ServerAppIo API (REST-based)."""
    try:
        import uvicorn

        from flwr.server.superlink.fleet.rest_rere.rest_api import app as fast_api_app
    except ModuleNotFoundError:
        flwr_exit(ExitCode.COMMON_MISSING_EXTRA_REST)

    log(INFO, "Starting Flower REST server")

    # See: https://www.starlette.io/applications/#accessing-the-app-instance
    fast_api_app.state.STATE_FACTORY = state_factory
    fast_api_app.state.OBJECTSTORE_FACTORY = objectstore_factory

    uvicorn.run(
        app="flwr.server.superlink.fleet.rest_rere.rest_api:app",
        port=port,
        host=host,
        reload=False,
        access_log=True,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        workers=num_workers,
    )


def _parse_args_run_superlink() -> argparse.ArgumentParser:
    """Parse command line arguments for both ServerAppIo API and Fleet API."""
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
    _add_args_serverappio_api(parser=parser)
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
        help="Fleet API server SSL certificate file (as a path str) "
        "to create a secure connection.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--ssl-keyfile",
        help="Fleet API server SSL private key file (as a path str) "
        "to create a secure connection.",
        type=str,
    )
    parser.add_argument(
        "--ssl-ca-certfile",
        help="Fleet API server SSL CA certificate file (as a path str) "
        "to create a secure connection.",
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
    add_args_runtime_dependency_install(parser)
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


def _add_args_serverappio_api(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--serverappio-api-address",
        "--simulationio-api-address",
        dest="serverappio_api_address",
        default=SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS,
        help="ServerAppIo API (gRPC) server address (IPv4, IPv6, or a domain name). "
        "`--simulationio-api-address` is accepted as a deprecated alias. "
        f"By default, it is set to {SERVERAPPIO_API_DEFAULT_SERVER_ADDRESS}.",
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
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
            TRANSPORT_TYPE_REST,
        ],
        help="Start a gRPC-rere or REST (experimental) Fleet API server.",
    )
    parser.add_argument(
        "--fleet-api-address",
        help="Fleet API server address (IPv4, IPv6, or a domain name).",
    )
    parser.add_argument(
        "--fleet-api-num-workers",
        default=1,
        type=int,
        help="Set the number of concurrent workers for the Fleet API server.",
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
