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
"""Tests for RunSeries title generation."""

from unittest.mock import Mock, patch

from flwr.common.constant import Status
from flwr.proto.task_pb2 import Task, TaskStatus  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.constant import TaskType
from flwr.supercore.json_message.model_message import ModelResponse

from .conversation_title import start_title_generation


def test_start_title_generation() -> None:
    """Create a model task and persist its response in a daemon thread."""
    state = Mock(spec=LinkState)
    state.create_task.return_value = 22
    state.store_task_message.return_value = True
    state.get_tasks.return_value = [Task(status=TaskStatus(status=Status.FINISHED))]
    state.get_task_message.return_value = [
        ModelResponse(
            dst_task_id=22,
            response={
                "object": "response",
                "output": [
                    {"content": [{"type": "output_text", "text": " Model title "}]}
                ],
            },
            reply_to_message_id="request-id",
        )
    ]

    with patch("flwr.superlink.servicer.control.conversation_title.Thread") as thread:
        start_title_generation(state, 1, 33, "Prompt")
        thread.call_args.kwargs["target"](*thread.call_args.kwargs["args"])

    state.create_task.assert_called_once_with(
        TaskType.MODEL,
        1,
        model_ref="openai/gpt-5-nano",
    )
    request = state.store_task_message.call_args.args[0]
    assert request.metadata.src_task_id == 22
    assert request.metadata.dst_task_id == 22
    state.set_run_series_description.assert_called_once_with(33, "Model title")
    assert thread.call_args.kwargs["daemon"] is True
