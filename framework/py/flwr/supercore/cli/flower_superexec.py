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
"""`flower-superexec` command."""


import argparse
from logging import INFO, WARN
from pathlib import Path
from typing import Any

import yaml

from flwr.common import EventType, event
from flwr.common.args import add_args_runtime_dependency_install
from flwr.common.constant import ExecPluginType
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.logger import log
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.auth import (
    add_superexec_auth_secret_args,
    load_superexec_auth_secret,
)
from flwr.supercore.constant import EXEC_PLUGIN_SECTION
from flwr.supercore.grpc_health import add_args_health
from flwr.supercore.superexec.plugin import (
    ClientAppExecPlugin,
    ExecPlugin,
    ServerAppEphemeralExecPlugin,
    ServerAppExecPlugin,
)
from flwr.supercore.superexec.run_superexec import run_superexec
from flwr.supercore.update_check import warn_if_flwr_update_available
from flwr.supercore.utils import disable_process_dumping
from flwr.supercore.version import package_version


def flower_superexec() -> None:
    """Run `flower-superexec` command."""
    disable_process_dumping(strict=False)
    warn_if_flwr_update_available(process_name="flower-superexec")
    args = _parse_args().parse_args()

    # Log the first message after parsing arguments in case of `--help`
    log(INFO, "Starting Flower SuperExec")

    event(EventType.RUN_SUPEREXEC_ENTER, {"plugin_type": args.plugin_type})

    # Load plugin config from YAML file if provided
    plugin_config = None
    if plugin_config_path := getattr(args, "plugin_config", None):
        try:
            with open(plugin_config_path, encoding="utf-8") as file:
                yaml_config: dict[str, Any] | None = yaml.safe_load(file)
                if yaml_config is None or EXEC_PLUGIN_SECTION not in yaml_config:
                    raise ValueError(f"Missing '{EXEC_PLUGIN_SECTION}' section.")
                plugin_config = yaml_config[EXEC_PLUGIN_SECTION]
        except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
            flwr_exit(
                ExitCode.SUPEREXEC_INVALID_PLUGIN_CONFIG,
                f"Failed to load plugin config from '{plugin_config_path}': {e!r}",
            )

    # Get the plugin class and stub class based on the plugin type
    if args.plugin_type == ExecPluginType.SIMULATION:
        log(
            WARN,
            "The '%s' plugin type is deprecated and will be removed in a future "
            "release. Please use '%s' instead, which supports both simulation "
            "and deployment.",
            ExecPluginType.SIMULATION,
            ExecPluginType.SERVER_APP,
        )
        args.plugin_type = ExecPluginType.SERVER_APP

    if args.plugin_type == ExecPluginType.SERVER_APP_EPHEMERAL:
        log(
            WARN,
            "The '%s' plugin type is experimental and may be removed in a future "
            "release. Please use '%s' for production deployments.",
            ExecPluginType.SERVER_APP_EPHEMERAL,
            ExecPluginType.SERVER_APP,
        )

    plugin_class, stub_class = _get_plugin_and_stub_class(args.plugin_type)
    superexec_auth_secret = None
    if args.superexec_auth_secret_file is not None:
        try:
            superexec_auth_secret = load_superexec_auth_secret(
                secret_file=args.superexec_auth_secret_file,
            )
        except ValueError as err:
            flwr_exit(
                ExitCode.SUPEREXEC_AUTH_SECRET_LOAD_FAILED,
                f"Failed to load SuperExec authentication secret: {err}",
            )

        # Destroy the auth secret file immediately after loading
        if args.plugin_type == ExecPluginType.SERVER_APP_EPHEMERAL:
            try:
                secret_path = Path(args.superexec_auth_secret_file).expanduser()
                secret_path.write_bytes(b"\x00" * secret_path.stat().st_size)
                secret_path.unlink()
            except OSError as e:
                log(WARN, "Failed to destroy authentication secret file: %s", e)

    run_superexec(
        plugin_class=plugin_class,
        stub_class=stub_class,  # type: ignore
        appio_api_address=args.appio_api_address,
        insecure=args.insecure,
        root_certificates_path=args.root_certificates,
        superexec_auth_secret=superexec_auth_secret,
        plugin_config=plugin_config,
        parent_pid=args.parent_pid,
        health_server_address=args.health_server_address,
        runtime_dependency_install=args.runtime_dependency_install,
    )


def _parse_args() -> argparse.ArgumentParser:
    """Parse `flower-superexec` command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Flower SuperExec.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"Flower version: {package_version}",
    )
    parser.add_argument(
        "--appio-api-address", type=str, required=True, help="Address of the AppIO API"
    )
    parser.add_argument(
        "--plugin-type",
        type=str,
        choices=ExecPluginType.all(),
        required=True,
        help="The type of plugin to use.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Connect to the AppIO API without TLS. "
        "Data transmitted between the client and server is not encrypted. "
        "Use this flag only if you understand the risks.",
    )
    parser.add_argument(
        "--root-certificates",
        metavar="ROOT_CERT",
        type=str,
        help="Path to a PEM-encoded root CA certificate (or CA bundle) used to verify "
        "the server's TLS certificate. This is not a client certificate for mTLS.",
    )
    parser.add_argument(
        "--parent-pid",
        type=int,
        default=None,
        help="The PID of the parent process. When set, the process will terminate "
        "when the parent process exits.",
    )
    add_superexec_auth_secret_args(parser)
    add_args_health(parser)
    add_args_runtime_dependency_install(parser)
    return parser


def _get_plugin_and_stub_class(
    plugin_type: str,
) -> tuple[type[ExecPlugin], type[object]]:
    """Get the plugin class and stub class based on the plugin type."""
    mapping: dict[str, tuple[type[ExecPlugin], type[object]]] = {
        ExecPluginType.CLIENT_APP: (ClientAppExecPlugin, ClientAppIoStub),
        ExecPluginType.SERVER_APP: (ServerAppExecPlugin, ServerAppIoStub),
        ExecPluginType.SERVER_APP_EPHEMERAL: (
            ServerAppEphemeralExecPlugin,
            ServerAppIoStub,
        ),
    }
    if plugin_type in mapping:
        return mapping[plugin_type]
    raise ValueError(f"Unknown plugin type: {plugin_type}")
