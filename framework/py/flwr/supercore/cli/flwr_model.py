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

from flwr.common.args import add_args_flwr_app_common, try_obtain_flwr_app_token
from flwr.common.constant import SERVERAPPIO_API_DEFAULT_CLIENT_ADDRESS
from flwr.common.logger import log, restore_output
from flwr.supercore.task_process import run_model
from flwr.supercore.tls import validate_and_resolve_root_certificates


def flwr_model() -> None:
    """Run process-isolated Flower model task."""
    args = _parse_args_run_flwr_model().parse_args()
    token = try_obtain_flwr_app_token(args)

    log(INFO, "Start `flwr-model` process")
    log(
        DEBUG,
        "`flwr-model` will attempt to connect to SuperLink's ServerAppIo API at %s",
        args.serverappio_api_address,
    )
    run_model(
        serverappio_api_address=args.serverappio_api_address,
        token=token,
        insecure=args.insecure,
        certificates=validate_and_resolve_root_certificates(
            args.root_certificates, args.insecure
        ),
        parent_pid=args.parent_pid,
    )

    # Restore stdout/stderr
    restore_output()


def _parse_args_run_flwr_model() -> argparse.ArgumentParser:
    """Parse `flwr-model` command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a Flower model task",
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
