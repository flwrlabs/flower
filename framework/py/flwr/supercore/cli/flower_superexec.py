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
from logging import INFO, WARN
from typing import Any

import yaml

from flwr.common.args import add_args_runtime_dependency_install
from flwr.common.constant import ExecPluginType
from flwr.supercore import log
from flwr.supercore.auth import (
    add_superexec_auth_secret_args,
    load_superexec_auth_secret,
)
from flwr.supercore.constant import EXEC_PLUGIN_SECTION, ExecutorType
from flwr.supercore.exit import ExitCode, flwr_exit
from flwr.supercore.grpc_health import add_args_health
from flwr.supercore.runtime import RuntimeHttpClient
from flwr.supercore.superexec.executor.config import (
    ExecutorConfig,
    ExecutorConfigError,
    load_executor_config,
)
from flwr.supercore.superexec.plugin import AutoExecPlugin
from flwr.supercore.superexec.run_superexec import run_superexec
from flwr.supercore.telemetry import EventType, event
from flwr.supercore.update_check import warn_if_flwr_update_available
from flwr.supercore.utils import disable_process_dumping
from flwr.supercore.version import package_version


def flower_superexec() -> None:
    """Run `flower-superexec` command."""
    disable_process_dumping(strict=False)
    warn_if_flwr_update_available(process_name="flower-superexec")
    args = _parse_args().parse_args()

    if any(
        arg == "--appio-api-address" or arg.startswith("--appio-api-address=")
        for arg in sys.argv[1:]
    ):
        log(
            WARN,
            "The `--appio-api-address` argument has been renamed to "
            "`--runtime-api-address`. Please update your command; the old name will "
            "be removed in a future release.",
        )

    # Log the first message after parsing arguments in case of `--help`
    log(INFO, "Starting Flower SuperExec")
    if args.plugin_type is not None:
        log(
            WARN,
            "The `--plugin-type` argument is deprecated and ignored; SuperExec "
            "selects execution from the task type.",
        )

    event(EventType.RUN_SUPEREXEC_ENTER, {"plugin_type": "auto"})

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

    run_superexec(
        plugin_class=AutoExecPlugin,
        client_class=RuntimeHttpClient,
        runtime_api_address=args.runtime_api_address,
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
    runtime_api_address_group = parser.add_mutually_exclusive_group(required=True)
    runtime_api_address_group.add_argument(
        "--runtime-api-address",
        dest="runtime_api_address",
        type=str,
        help="Address of the Runtime API",
    )
    runtime_api_address_group.add_argument(
        "--appio-api-address",
        dest="runtime_api_address",
        type=str,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--plugin-type",
        type=str,
        choices=ExecPluginType.all(),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Connect to the Runtime API without TLS. "
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
