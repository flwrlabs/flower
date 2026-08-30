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
"""Tests for the SuperNode Runtime task dependency."""

from unittest.mock import Mock, patch

from fastapi import FastAPI, Request

from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.supercore.constant import TASK_TOKEN_HEADER
from flwr.supernode.nodestate import NodeState

from .task import TaskDependency, get_task


def test_get_task_delegates_to_shared_authentication() -> None:
    """Authenticate a task using the supplied SuperNode state."""
    request = Mock(spec=Request)
    state = Mock(spec=NodeState)
    expected_task = Mock(spec=Task)

    with patch(
        "flwr.supernode.dependencies.task.authenticate_task",
        return_value=expected_task,
    ) as authenticate:
        task = get_task(request, "token", state)

    assert task is expected_task
    authenticate.assert_called_once_with(request, "token", state)


def test_task_dependency_declares_task_token_security_scheme() -> None:
    """Describe task-token authentication in the generated OpenAPI schema."""
    app = FastAPI()

    @app.get("/runtime/test")
    def runtime_endpoint(task: TaskDependency) -> None:
        _ = task

    schema = app.openapi()

    assert schema["components"]["securitySchemes"]["RuntimeTaskToken"] == {
        "type": "apiKey",
        "description": "Task token issued by the Runtime API.",
        "in": "header",
        "name": TASK_TOKEN_HEADER,
    }
    assert schema["paths"]["/runtime/test"]["get"]["security"] == [
        {"RuntimeTaskToken": []}
    ]
