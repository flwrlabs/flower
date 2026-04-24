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
"""`flwr-model` command."""


import argparse
from logging import DEBUG, INFO
from queue import Queue

from flwr.common.args import add_args_flwr_app_common
from flwr.common.constant import SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.logger import log, mirror_output_to_queue, restore_output
from flwr.supercore.executors.run_model import run_model


def flwr_model() -> None:
    """Run process-isolated Flower ModelApp."""
    args = _parse_args_run_flwr_model().parse_args()

    if not args.insecure:
        flwr_exit(
            ExitCode.COMMON_TLS_NOT_SUPPORTED,
            "`flwr-model` does not support TLS yet.",
        )

    # Capture stdout/stderr
    log_queue: Queue[str | None] = Queue()
    mirror_output_to_queue(log_queue)

    log(INFO, "Start `flwr-model` process")
    log(
        DEBUG,
        "`flwr-model` will attempt to connect to SuperLink's ServerAppIo API at %s",
        args.serverappio_api_address,
    )
    run_model(
        serverappio_api_address=args.serverappio_api_address,
        log_queue=log_queue,
        token=args.token,
        certificates=None,
        parent_pid=args.parent_pid,
        runtime_dependency_install=args.runtime_dependency_install,
    )

    # Restore stdout/stderr
    restore_output()


def _parse_args_run_flwr_model() -> argparse.ArgumentParser:
    """Parse `flwr-model` command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a Flower ModelApp",
    )
    parser.add_argument(
        "--serverappio-api-address",
        default=SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS,
        type=str,
        help="Address of SuperLink's ServerAppIo API (IPv4, IPv6, or a domain name)."
        f"By default, it is set to {SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS}.",
    )
    add_args_flwr_app_common(parser=parser)
    return parser
