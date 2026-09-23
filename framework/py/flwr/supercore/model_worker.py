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
"""Single-use prestarted worker for warm Model TaskExecutor Pods."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

from flwr.common.args import add_args_flwr_app_common, try_obtain_flwr_app_token
from flwr.supercore import task_worker
from flwr.supercore.warm_executor_constants import (
    WARM_EXECUTOR_BUSY_FILE,
    WARM_EXECUTOR_READY_FILE,
    WARM_MODEL_EXECUTOR_SOCKET,
)


@dataclass(frozen=True)
class ModelInvocation:
    """Describe one token-scoped invocation for the prestarted Model worker."""

    token: str
    runtime_api_address: str
    insecure: bool
    root_certificates_path: str | None

    @classmethod
    def from_payload(cls, payload: object) -> ModelInvocation:
        """Validate and construct one invocation request."""
        token, runtime_api_address, insecure, root_certificates_path = (
            task_worker.parse_invocation_payload(payload, "Model")
        )
        return cls(
            token=token,
            runtime_api_address=runtime_api_address,
            insecure=insecure,
            root_certificates_path=root_certificates_path,
        )


def serve_prestarted_model_worker(
    socket_path: Path = Path(WARM_MODEL_EXECUTOR_SOCKET),
    ready_file: Path = Path(WARM_EXECUTOR_READY_FILE),
    busy_file: Path = Path(WARM_EXECUTOR_BUSY_FILE),
) -> int:
    """Preload the Model task path, then serve exactly one invocation."""
    # Keep the exec-side dispatcher lightweight. Only the resident process pays
    # the Model/provider import cost, and it does so before reporting readiness.
    from .task_process.model.run_model import (  # pylint: disable=import-outside-toplevel
        run_model_once,
    )

    return task_worker.serve_prestarted_worker(
        socket_path,
        ready_file,
        busy_file,
        run_model_once,
        ModelInvocation.from_payload,
        "Model",
    )


def dispatch_prestarted_model(
    invocation: ModelInvocation,
    socket_path: Path = Path(WARM_MODEL_EXECUTOR_SOCKET),
) -> int:
    """Relay one exec-delivered invocation to the prestarted Model worker."""
    return task_worker.dispatch_prestarted_task(
        asdict(invocation), socket_path, "Model"
    )


def _parse_args() -> argparse.ArgumentParser:
    """Build the internal prestarted Model worker argument parser."""
    parser = argparse.ArgumentParser(description="Run a prestarted Model worker")
    modes = parser.add_subparsers(dest="mode", required=True)
    modes.add_parser("serve")
    dispatch = modes.add_parser("dispatch")
    dispatch.add_argument("--runtime-api-address", required=True)
    add_args_flwr_app_common(dispatch, include_token_stdin=True)
    return parser


def main() -> None:
    """Run the resident worker or its exec-side dispatcher."""
    args = _parse_args().parse_args()
    if args.mode == "serve":
        raise SystemExit(serve_prestarted_model_worker())
    invocation = ModelInvocation(
        token=try_obtain_flwr_app_token(args),
        runtime_api_address=args.runtime_api_address,
        insecure=args.insecure,
        root_certificates_path=args.root_certificates,
    )
    raise SystemExit(dispatch_prestarted_model(invocation))


if __name__ == "__main__":
    main()
