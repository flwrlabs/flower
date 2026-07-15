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
"""Tests for the AgentApp task-process lifecycle."""

from pathlib import Path
from queue import Queue
from unittest.mock import Mock, patch

from flwr.agentapp import AgentApp
from flwr.app import Context, RecordDict
from flwr.common.constant import SubStatus
from flwr.proto.appio_pb2 import PushTaskOutputRequest  # pylint: disable=E0611

from .run_agentapp import run_agentapp


def _run_with_title(
    *, app_error: Exception | None, title_error: Exception | None
) -> tuple[PushTaskOutputRequest, list[str]]:
    order: list[str] = []
    app = AgentApp()

    @app.main()
    def main(_agent: object, _context: Context) -> None:
        order.append("app")
        if app_error:
            raise app_error

    context = Context(1, 0, {}, RecordDict(), {}, series_id=11)
    run = Mock(
        run_id=1,
        override_config={},
        fab_id="app",
        fab_version="1.0.0",
    )
    fab = Mock(content=b"fab", hash_str="fab-hash")
    grid = Mock()
    grid._stub.PullTaskInput.return_value = Mock(
        context=Mock(),
        run=Mock(),
        fab=Mock(),
        task_id=7,
        should_generate_series_description=True,
    )
    heartbeat = Mock(is_running=False)
    exit_handlers: list[object] = []

    def register_handlers(**kwargs: object) -> None:
        exit_handlers.extend(kwargs["exit_handlers"])  # type: ignore[arg-type]

    def exit_process(**_kwargs: object) -> None:
        for handler in reversed(exit_handlers):
            handler()  # type: ignore[operator]

    def generate_title(_responses: object, _seed: str) -> str:
        order.append("title")
        if title_error:
            raise title_error
        return "Generated title"

    module = "flwr.supercore.task_process.agent.run_agentapp"
    with (
        patch(f"{module}.GrpcGrid", return_value=grid),
        patch(f"{module}.HeartbeatSender", return_value=heartbeat),
        patch(f"{module}.make_task_heartbeat_fn_grpc"),
        patch(f"{module}.register_signal_handlers", side_effect=register_handlers),
        patch(f"{module}.flwr_exit", side_effect=exit_process),
        patch(f"{module}.context_from_proto", return_value=context),
        patch(f"{module}.run_from_proto", return_value=run),
        patch(f"{module}.fab_from_proto", return_value=fab),
        patch(f"{module}.get_sha256_hash", return_value="run-hash"),
        patch(f"{module}.start_log_uploader", return_value=None),
        patch(f"{module}.install_from_fab"),
        patch(f"{module}.get_fab_metadata", return_value=("app", "1.0.0")),
        patch(f"{module}.get_project_dir", return_value=Path("/tmp/app")),
        patch(
            f"{module}.get_project_config",
            return_value={
                "tool": {"flwr": {"app": {"components": {"agentapp": "app"}}}}
            },
        ),
        patch(
            f"{module}.get_fused_config_from_dir",
            return_value={"agent.input": "one two three four five"},
        ),
        patch(f"{module}.load_app", return_value=app),
        patch(f"{module}.generate_series_description", side_effect=generate_title),
        patch(f"{module}.cleanup_app_runtime_environment"),
        patch(f"{module}.event"),
    ):
        run_agentapp(
            "localhost:9091",
            Queue(),
            "token",
            insecure=True,
            runtime_dependency_install=False,
        )

    request = grid._stub.PushTaskOutput.call_args.args[0]
    return request, order


def test_title_failure_does_not_change_success_status() -> None:
    """A title exception preserves success and the deterministic fallback."""
    request, order = _run_with_title(
        app_error=None, title_error=RuntimeError("title failed")
    )

    assert order == ["app", "title"]
    assert request.sub_status == SubStatus.COMPLETED
    assert request.details == ""
    assert request.series_description == "one two three four"


def test_title_generation_does_not_mask_agentapp_failure() -> None:
    """The original AgentApp failure is preserved after title generation."""
    request, order = _run_with_title(
        app_error=RuntimeError("app failed"), title_error=None
    )

    assert order == ["app", "title"]
    assert request.sub_status == SubStatus.FAILED
    assert "app failed" in request.details
    assert request.series_description == "Generated title"
