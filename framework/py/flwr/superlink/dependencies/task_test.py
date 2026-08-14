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
"""Tests for the SuperLink Runtime task dependency."""

from fastapi import FastAPI

from flwr.supercore.constant import TASK_TOKEN_HEADER

from .task import TaskDependency


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
