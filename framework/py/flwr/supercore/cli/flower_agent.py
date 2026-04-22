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
"""`flower-agent` command."""


import argparse
from logging import INFO

from flwr.common import EventType, event
from flwr.common.args import add_args_runtime_dependency_install
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.logger import log
from flwr.supercore.auth import (
    add_superexec_auth_secret_args,
    load_superexec_auth_secret,
)
from flwr.supercore.grpc_health import add_args_health
from flwr.supercore.agent.run_flower_agent import run_flower_agent
from flwr.supercore.update_check import warn_if_flwr_update_available
from flwr.supercore.utils import disable_process_dumping
from flwr.supercore.version import package_version


def flower_agent() -> None:
    """Run `flower-agent` command."""
    disable_process_dumping(strict=False)
    warn_if_flwr_update_available(process_name="flower-agent")
    args = _parse_args().parse_args()

    if not args.insecure:
        flwr_exit(
            ExitCode.COMMON_TLS_NOT_SUPPORTED,
            "`flower-agent` does not support TLS yet.",
        )

    log(INFO, "Starting Flower Agent")

    event(EventType.RUN_AGENT_ENTER)

    superexec_auth_secret = None
    if args.superexec_auth_secret_file is not None:
        try:
            superexec_auth_secret = load_superexec_auth_secret(
                secret_file=args.superexec_auth_secret_file,
            )
        except ValueError as err:
            flwr_exit(
                ExitCode.SUPEREXEC_AUTH_SECRET_LOAD_FAILED,
                f"Failed to load Flower Agent authentication secret: {err}",
            )

    run_flower_agent(
        appio_api_address=args.appio_api_address,
        parent_pid=args.parent_pid,
        health_server_address=args.health_server_address,
        superexec_auth_secret=superexec_auth_secret,
        runtime_dependency_install=args.runtime_dependency_install,
    )


def _parse_args() -> argparse.ArgumentParser:
    """Parse `flower-agent` command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Flower Agent.",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"Flower version: {package_version}",
    )
    parser.add_argument(
        "--appio-api-address",
        type=str,
        required=True,
        help="Address of the AppIO API",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Connect to the AppIO API without TLS. "
        "Data transmitted between the client and server is not encrypted. "
        "Use this flag only if you understand the risks.",
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
