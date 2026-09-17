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
"""Tests for one-task Model execution inside a prestarted worker."""

import importlib
from unittest.mock import Mock

import pytest

from flwr.common.constant import SubStatus
from flwr.proto.run_pb2 import Run as ProtoRun  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import PullTaskInputResponse  # pylint: disable=E0611
from flwr.supercore.telemetry import EventType

run_model_module = importlib.import_module(
    "flwr.supercore.task_process.model.run_model"
)


@pytest.mark.parametrize(
    ("failure", "expected_returncode", "expected_status", "expected_details"),
    [
        (None, 0, SubStatus.COMPLETED, ""),
        (
            RuntimeError("provider failed"),
            1,
            SubStatus.FAILED,
            "Model task failed with exception: provider failed",
        ),
    ],
)
def test_run_model_once_uses_fresh_task_scoped_state_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception | None,
    expected_returncode: int,
    expected_status: SubStatus,
    expected_details: str,
) -> None:
    """One invocation should own and close its Runtime and heartbeat state."""
    client = Mock()
    client.PullTaskInput.return_value = PullTaskInputResponse(
        task_id=17, run=ProtoRun(run_id=42)
    )
    retry_invoker = Mock(max_tries=10)
    create_client = Mock(return_value=(client, retry_invoker))
    heartbeat_sender = Mock(is_running=True)
    heartbeat_cls = Mock(return_value=heartbeat_sender)
    handle_task = Mock(side_effect=failure)
    telemetry_event = Mock()
    monkeypatch.setattr(run_model_module, "_create_runtime_client", create_client)
    monkeypatch.setattr(run_model_module, "HeartbeatSender", heartbeat_cls)
    monkeypatch.setattr(run_model_module, "handle_task", handle_task)
    monkeypatch.setattr(run_model_module, "event", telemetry_event)

    returncode = run_model_module.run_model_once(
        "runtime.example:9092", "task-token", True
    )

    assert returncode == expected_returncode
    create_client.assert_called_once_with(
        runtime_api_address="runtime.example:9092",
        token="task-token",
        insecure=True,
        certificates=None,
    )
    heartbeat_sender.start.assert_called_once_with()
    handle_task.assert_called_once_with(client=client, task_id=17, run_id=42)
    output = client.PushTaskOutput.call_args.args[0]
    assert output.sub_status == expected_status
    assert output.details == expected_details
    assert retry_invoker.max_tries == 1
    heartbeat_sender.stop.assert_called_once_with()
    client.close.assert_called_once_with()
    assert telemetry_event.call_args_list[0].args == (EventType.FLWR_MODEL_RUN_ENTER,)
    assert telemetry_event.call_args_list[-1].args == (
        EventType.FLWR_MODEL_RUN_LEAVE,
        {"exit_code": 0 if failure is None else 800},
    )
