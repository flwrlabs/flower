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
"""Single-use prestarted worker for selected warm AgentApp Pods."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from flwr.common.args import add_args_flwr_app_common, try_obtain_flwr_app_token
from flwr.supercore import task_worker
from flwr.supercore.typing import JSONObject
from flwr.supercore.warm_executor_constants import (
    WARM_AGENTAPP_EXECUTOR_SOCKET,
    WARM_EXECUTOR_BUSY_FILE,
    WARM_EXECUTOR_READY_FILE,
)


@dataclass(frozen=True)
class AgentAppInvocation:
    """Describe one invocation for a FAB-specific AgentApp worker."""

    token: str
    runtime_api_address: str
    insecure: bool
    root_certificates_path: str | None
    fab_hash: str

    @classmethod
    def from_payload(cls, payload: object) -> AgentAppInvocation:
        """Validate and construct one invocation request."""
        if not isinstance(payload, dict) or set(payload) != {
            "token",
            "runtime_api_address",
            "insecure",
            "root_certificates_path",
            "fab_hash",
        }:
            raise ValueError("AgentApp invocation contains unexpected fields.")
        common_payload: JSONObject = {
            key: value for key, value in payload.items() if key != "fab_hash"
        }
        common = task_worker.parse_invocation_payload(common_payload, "AgentApp")
        fab_hash = payload["fab_hash"]
        if (
            not isinstance(fab_hash, str)
            or len(fab_hash) != 64
            or any(char not in "0123456789abcdef" for char in fab_hash)
        ):
            raise ValueError(
                "AgentApp invocation field 'fab_hash' must be a full SHA-256 hash."
            )
        return cls(*common, fab_hash)


def serve_prestarted_agentapp_worker(
    fab_hash: str,
    fab_path: Path,
    socket_path: Path = Path(WARM_AGENTAPP_EXECUTOR_SOCKET),
    ready_file: Path = Path(WARM_EXECUTOR_READY_FILE),
    busy_file: Path = Path(WARM_EXECUTOR_BUSY_FILE),
) -> int:
    """Preload one trusted AgentApp, then serve exactly one invocation."""
    from .task_process.agent.run_agentapp import (  # pylint: disable=import-outside-toplevel
        preload_agentapp,
        run_agentapp_once,
    )

    preloaded = preload_agentapp(fab_path, fab_hash)
    certificates_path: str | None = None

    def parse_invocation(payload: object) -> AgentAppInvocation:
        nonlocal certificates_path
        invocation = AgentAppInvocation.from_payload(payload)
        if invocation.fab_hash != preloaded.fab_hash:
            raise ValueError("AgentApp invocation does not match the preloaded FAB.")
        certificates_path = invocation.root_certificates_path
        return invocation

    def run_once(
        runtime_api_address: str,
        token: str,
        insecure: bool,
        certificates: bytes | None,
        on_started: Callable[[], None],
    ) -> int:
        return run_agentapp_once(
            runtime_api_address,
            token,
            insecure,
            certificates,
            on_started,
            preloaded=preloaded,
            certificates_path=certificates_path,
        )

    return task_worker.serve_prestarted_worker(
        socket_path,
        ready_file,
        busy_file,
        run_once,
        parse_invocation,
        "AgentApp",
    )


def dispatch_prestarted_agentapp(
    invocation: AgentAppInvocation,
    socket_path: Path = Path(WARM_AGENTAPP_EXECUTOR_SOCKET),
) -> int:
    """Relay one invocation to the matching prestarted AgentApp worker."""
    return task_worker.dispatch_prestarted_task(
        asdict(invocation), socket_path, "AgentApp"
    )


def _parse_args() -> argparse.ArgumentParser:
    """Build the internal prestarted AgentApp worker argument parser."""
    parser = argparse.ArgumentParser(description="Run a prestarted AgentApp worker")
    modes = parser.add_subparsers(dest="mode", required=True)
    serve = modes.add_parser("serve")
    serve.add_argument("--fab-hash", required=True)
    serve.add_argument("--fab-path", required=True, type=Path)
    dispatch = modes.add_parser("dispatch")
    dispatch.add_argument("--runtime-api-address", required=True)
    dispatch.add_argument("--fab-hash", required=True)
    add_args_flwr_app_common(
        dispatch,
        include_token_stdin=True,
        include_runtime_dependency_install=False,
    )
    return parser


def main() -> None:
    """Run the resident worker or its exec-side dispatcher."""
    args = _parse_args().parse_args()
    if args.mode == "serve":
        raise SystemExit(serve_prestarted_agentapp_worker(args.fab_hash, args.fab_path))
    invocation = AgentAppInvocation(
        token=try_obtain_flwr_app_token(args),
        runtime_api_address=args.runtime_api_address,
        insecure=args.insecure,
        root_certificates_path=args.root_certificates,
        fab_hash=args.fab_hash,
    )
    raise SystemExit(dispatch_prestarted_agentapp(invocation))


if __name__ == "__main__":
    main()
