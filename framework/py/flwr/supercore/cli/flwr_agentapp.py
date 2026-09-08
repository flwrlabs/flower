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
import re
import sys
from io import BufferedReader
from logging import DEBUG, INFO
from pathlib import Path
from queue import Queue
from typing import cast

from flwr.common.args import add_args_flwr_app_common, try_obtain_flwr_app_token
from flwr.common.constant import FLWR_TASK_TOKEN_LENGTH
from flwr.supercore import log
from flwr.supercore.constant import SUPERLINK_DEFAULT_CLIENT_ADDRESS
from flwr.supercore.logger import mirror_output_to_queue, restore_output
from flwr.supercore.task_process import run_agentapp

_TOKEN_STDIN_ACKNOWLEDGEMENT = "FLWR_AGENTAPP_TOKEN_ACCEPTED"
_TASK_TOKEN_PATTERN = re.compile(rf"[0-9a-f]{{{FLWR_TASK_TOKEN_LENGTH * 2}}}")
_TOKEN_STDIN_READ_LIMIT = FLWR_TASK_TOKEN_LENGTH * 2 + 2


def flwr_agentapp() -> None:
    """Run process-isolated Flower AgentApp."""
    args = _parse_args_run_flwr_agentapp().parse_args()
    token = _try_obtain_agentapp_token(args)

    if cast(bool, getattr(args, "token_stdin", False)):
        print(_TOKEN_STDIN_ACKNOWLEDGEMENT, flush=True)

    # Capture stdout/stderr
    log_queue: Queue[str | None] = Queue()
    mirror_output_to_queue(log_queue)

    log(INFO, "Start `flwr-agentapp` process")
    log(
        DEBUG,
        "`flwr-agentapp` will attempt to connect to SuperLink's Runtime API at %s",
        args.runtime_api_address,
    )

    # Resolve root certificates path if provided
    root_certificates_path = None
    if args.root_certificates is not None:
        root_certificates_path = str(
            Path(args.root_certificates).expanduser().resolve()
        )

    run_agentapp(
        runtime_api_address=args.runtime_api_address,
        log_queue=log_queue,
        token=token,
        insecure=args.insecure,
        certificates_path=root_certificates_path,
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
        "--runtime-api-address",
        dest="runtime_api_address",
        default=SUPERLINK_DEFAULT_CLIENT_ADDRESS,
        type=str,
        help="Address of SuperLink's Runtime API (IPv4, IPv6, or a domain name)."
        f"By default, it is set to {SUPERLINK_DEFAULT_CLIENT_ADDRESS}.",
    )
    add_args_flwr_app_common(parser=parser, include_token_stdin=True)
    return parser


def _try_obtain_agentapp_token(args: argparse.Namespace) -> str:
    """Return the AgentApp token from an existing source or private stdin mode."""
    if not cast(bool, getattr(args, "token_stdin", False)):
        return try_obtain_flwr_app_token(args)

    try:
        token_bytes = bytearray()
        token_stdin = cast(BufferedReader, sys.stdin.buffer)
        while b"\n" not in token_bytes and len(token_bytes) < _TOKEN_STDIN_READ_LIMIT:
            chunk = token_stdin.read1(_TOKEN_STDIN_READ_LIMIT - len(token_bytes))
            if not chunk:
                break
            token_bytes.extend(chunk)
        token_input = token_bytes.decode("ascii")
    except (AttributeError, OSError, UnicodeError):
        sys.exit("Standard input does not contain exactly one valid task token.")
    finally:
        try:
            sys.stdin.close()
        except OSError:
            pass

    token = token_input.removesuffix("\n")
    if _TASK_TOKEN_PATTERN.fullmatch(token) is None:
        sys.exit("Standard input does not contain exactly one valid task token.")
    return token
