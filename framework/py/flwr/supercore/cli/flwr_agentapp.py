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
"""`flwr-agentapp` command."""


import argparse
from logging import DEBUG, INFO
from queue import Queue

from flwr.common.args import add_args_flwr_app_common, try_obtain_flwr_app_token
from flwr.supercore import log
from flwr.supercore.constant import SUPERLINK_DEFAULT_CLIENT_ADDRESS
from flwr.supercore.logger import mirror_output_to_queue, restore_output
from flwr.supercore.task_process import run_agentapp
from flwr.supercore.tls import validate_and_resolve_root_certificates


def flwr_agentapp() -> None:
    """Run process-isolated Flower AgentApp."""
    args = _parse_args_run_flwr_agentapp().parse_args()
    token = try_obtain_flwr_app_token(args)

    # Capture stdout/stderr
    log_queue: Queue[str | None] = Queue()
    mirror_output_to_queue(log_queue)

    log(INFO, "Start `flwr-agentapp` process")
    log(
        DEBUG,
        "`flwr-agentapp` will attempt to connect to SuperLink's Runtime API at %s",
        args.runtime_api_address,
    )
    run_agentapp(
        runtime_api_address=args.runtime_api_address,
        log_queue=log_queue,
        token=token,
        insecure=args.insecure,
        certificates=validate_and_resolve_root_certificates(
            args.root_certificates, args.insecure
        ),
        parent_pid=args.parent_pid,
        runtime_dependency_install=args.runtime_dependency_install,
    )

    # Restore stdout/stderr
    restore_output()


def _parse_args_run_flwr_agentapp() -> argparse.ArgumentParser:
    """Parse `flwr-agentapp` command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a Flower AgentApp",
    )
    parser.add_argument(
        "--serverappio-api-address",
        dest="runtime_api_address",
        default=SUPERLINK_DEFAULT_CLIENT_ADDRESS,
        type=str,
        help="Address of SuperLink's Runtime API (IPv4, IPv6, or a domain name)."
        f"By default, it is set to {SUPERLINK_DEFAULT_CLIENT_ADDRESS}.",
    )
    add_args_flwr_app_common(parser=parser)
    return parser
