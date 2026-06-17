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
import sys
from collections.abc import Sequence
from logging import INFO, WARN
from pathlib import Path
from typing import Any

import yaml

from flwr.common import EventType, event
from flwr.common.args import (
    RuntimeDependencyInstallHelp,
    add_args_runtime_dependency_install,
)
from flwr.common.constant import ExecPluginType
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.logger import log
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.auth import (
    add_superexec_auth_secret_args,
    load_superexec_auth_secret,
)
from flwr.supercore.constant import EXEC_PLUGIN_SECTION, ExecutorType
from flwr.supercore.grpc_health import add_args_health
from flwr.supercore.superexec.executor.config import (
    ExecutorConfig,
    ExecutorConfigError,
    load_executor_config,
)
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

# Plugin types that install dependencies by default
_SERVERAPP_PLUGIN_TYPES = {
    ExecPluginType.SERVER_APP,
    ExecPluginType.SIMULATION,
    ExecPluginType.SERVER_APP_EPHEMERAL,
}


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

    executor_config = _load_executor_config(
        getattr(args, "executor_config", None), args.executor
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
        executor_type=args.executor,
        executor_config=executor_config,
    )


class _SuperExecArgumentParser:
    """Plugin-aware argument parser for `flower-superexec`."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        """Parse arguments after selecting plugin-specific runtime flags."""
        args_to_parse = list(args) if args is not None else sys.argv[1:]
        # Runtime dependency flags differ by plugin, so inspect plugin type first.
        parser = _build_parser(_parse_plugin_type(args_to_parse))
        parsed = parser.parse_args(args_to_parse, namespace)
        _warn_deprecated_serverapp_runtime_dependency_install(parsed, args_to_parse)
        return parsed


def _parse_args() -> _SuperExecArgumentParser:
    """Return a plugin-aware `flower-superexec` argument parser."""
    return _SuperExecArgumentParser()


def _warn_deprecated_serverapp_runtime_dependency_install(
    parsed: argparse.Namespace, args_to_parse: Sequence[str]
) -> None:
    """Warn if the deprecated ServerApp dependency installation flag is passed."""
    if (
        parsed.plugin_type in _SERVERAPP_PLUGIN_TYPES
        and "--allow-runtime-dependency-installation"
        in {arg.split("=")[0] for arg in args_to_parse if arg.startswith("--")}
    ):
        log(
            WARN,
            "The `--allow-runtime-dependency-installation` argument is "
            "deprecated for ServerApp plugins. Runtime dependency installation "
            "is now enabled by default. Use "
            "`--disable-runtime-dependency-installation` to disable it.",
        )


def _parse_plugin_type(args: Sequence[str]) -> str | None:
    """Parse the plugin type without validating the full command line."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--plugin-type", type=str, choices=ExecPluginType.all())
    parsed, _ = parser.parse_known_args(args)
    plugin_type = parsed.plugin_type
    return plugin_type if isinstance(plugin_type, str) else None


def _build_parser(plugin_type: str | None) -> argparse.ArgumentParser:
    """Build the `flower-superexec` parser for the selected plugin type."""
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
    parser.add_argument(
        "--executor",
        type=ExecutorType,
        choices=tuple(ExecutorType),
        default=ExecutorType.SUBPROCESS,
        help="The executor used to run task processes, for example as local "
        "subprocesses.",
    )
    parser.add_argument(
        "--executor-config",
        metavar="PATH",
        type=str,
        help="Path to a YAML config file for the selected executor.",
    )
    add_superexec_auth_secret_args(parser)
    add_args_health(parser)
    if plugin_type is None:
        # Generic help should explain the plugin-dependent defaults.
        add_args_runtime_dependency_install(
            parser,
            default=True,
            include_disable_flag=True,
            help_texts=RuntimeDependencyInstallHelp(
                allow_flag=(
                    "Allow runtime installation of app dependencies via `uv sync`. "
                    "This enables installation for `clientapp`. For `serverapp`, "
                    "`simulation`, and `serverapp-ephemeral`, installation is already "
                    "enabled by default, so this flag is deprecated."
                ),
                disable_flag=(
                    "Disable runtime installation of app dependencies via `uv sync`. "
                    "Only valid for `serverapp`, `simulation`, and "
                    "`serverapp-ephemeral`, where installation is enabled by default."
                ),
                default="",
            ),
        )
    elif plugin_type in _SERVERAPP_PLUGIN_TYPES:
        # ServerApp plugins install dependencies by default and expose opt-out.
        add_args_runtime_dependency_install(
            parser,
            default=True,
            include_disable_flag=True,
            help_texts=RuntimeDependencyInstallHelp(
                allow_flag=(
                    "Deprecated for ServerApp plugins. Use "
                    "`--disable-runtime-dependency-installation` to disable runtime "
                    "dependency installation."
                ),
            ),
        )
    else:
        add_args_runtime_dependency_install(parser)
    return parser


def _load_executor_config(
    executor_config_path: str | None, executor_type: ExecutorType
) -> ExecutorConfig | None:
    """Load executor config from a YAML file if needed."""
    if executor_config_path is None:
        return None

    try:
        return load_executor_config(executor_config_path, executor_type)
    except ExecutorConfigError as err:
        flwr_exit(ExitCode.SUPEREXEC_INVALID_EXECUTOR_CONFIG, str(err))


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
