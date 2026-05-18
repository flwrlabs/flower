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
"""SuperLink task command resolution for SuperExec plugins."""


from collections.abc import Sequence

from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.constant import TaskType

_SUPERLINK_TASK_TYPE_TO_COMMAND: dict[str, str] = {
    str(TaskType.SERVER_APP): "flwr-serverapp",
    str(TaskType.SIMULATION): "flwr-simulation",
    str(TaskType.AGENT_APP): "flwr-agentapp",
    str(TaskType.MODEL): "flwr-model",
}


def resolve_superlink_task_command(task_type: str) -> str | None:
    """Resolve a SuperLink task type to its fixed executor command."""
    return _SUPERLINK_TASK_TYPE_TO_COMMAND.get(task_type)


def select_superlink_task(candidate_tasks: Sequence[Task]) -> Task | None:
    """Select the first candidate task supported by the SuperLink plugin."""
    for task in candidate_tasks:
        if resolve_superlink_task_command(task.type) is not None:
            return task
    return None
