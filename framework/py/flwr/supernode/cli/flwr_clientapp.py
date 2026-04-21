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
"""`flwr-clientapp` command."""


import argparse
from logging import DEBUG, INFO

from flwr.common.args import add_args_flwr_app_common
from flwr.common.constant import CLIENTAPPIO_API_DEFAULT_CLIENT_ADDRESS
from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.logger import log
from flwr.supercore.utils import mask_string
from flwr.supernode.runtime.run_clientapp import run_clientapp


def flwr_clientapp() -> None:
    """Run process-isolated Flower ClientApp."""
    args = _parse_args_run_flwr_clientapp().parse_args()
    if not args.insecure:
        flwr_exit(
            ExitCode.COMMON_TLS_NOT_SUPPORTED,
            "flwr-clientapp does not support TLS yet.",
        )

    log(INFO, "Start `flwr-clientapp` process")
    log(
        DEBUG,
        "`flwr-clientapp` will attempt to connect to SuperNode's "
        "ClientAppIo API at %s with token %s",
        args.clientappio_api_address,
        mask_string(args.token) if args.token else "None",
    )
    run_clientapp(
        clientappio_api_address=args.clientappio_api_address,
        token=args.token,
        certificates=None,
        parent_pid=args.parent_pid,
        runtime_dependency_install=args.runtime_dependency_install,
    )


def _parse_args_run_flwr_clientapp() -> argparse.ArgumentParser:
    """Parse flwr-clientapp command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a Flower ClientApp",
    )
    parser.add_argument(
        "--clientappio-api-address",
        default=CLIENTAPPIO_API_DEFAULT_CLIENT_ADDRESS,
        type=str,
        help="Address of SuperNode's ClientAppIo API (IPv4, IPv6, or a domain name)."
        f"By default, it is set to {CLIENTAPPIO_API_DEFAULT_CLIENT_ADDRESS}.",
    )
    add_args_flwr_app_common(parser=parser)
    return parser
