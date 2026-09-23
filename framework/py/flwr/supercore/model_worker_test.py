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
"""Tests for the single-use prestarted Model worker adapter."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from . import model_worker, task_worker
from .model_worker import ModelInvocation


def test_model_invocation_from_payload() -> None:
    """The Model adapter should preserve the shared payload field mapping."""
    assert ModelInvocation.from_payload(
        {
            "token": "task-token",
            "runtime_api_address": "runtime.example:9092",
            "insecure": False,
            "root_certificates_path": "/runtime/ca.pem",
        }
    ) == ModelInvocation(
        token="task-token",
        runtime_api_address="runtime.example:9092",
        insecure=False,
        root_certificates_path="/runtime/ca.pem",
    )


def test_serve_prestarted_model_worker_delegates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Model adapter should configure the shared worker."""
    serve = Mock(return_value=0)
    monkeypatch.setattr(task_worker, "serve_prestarted_worker", serve)
    socket_path = tmp_path / "model.sock"
    ready_file = tmp_path / "ready"
    busy_file = tmp_path / "busy"

    assert (
        model_worker.serve_prestarted_model_worker(socket_path, ready_file, busy_file)
        == 0
    )

    args = serve.call_args.args
    assert args[:3] == (socket_path, ready_file, busy_file)
    assert args[4](
        {
            "token": "task-token",
            "runtime_api_address": "runtime.example:9092",
            "insecure": True,
            "root_certificates_path": None,
        }
    ) == ModelInvocation("task-token", "runtime.example:9092", True, None)
    assert args[5] == "Model"


def test_dispatch_prestarted_model_delegates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The Model adapter should serialize invocations for the shared dispatcher."""
    dispatch = Mock(return_value=3)
    monkeypatch.setattr(task_worker, "dispatch_prestarted_task", dispatch)
    socket_path = tmp_path / "model.sock"
    invocation = ModelInvocation("task-token", "runtime.example:9092", True, None)

    assert model_worker.dispatch_prestarted_model(invocation, socket_path) == 3
    dispatch.assert_called_once_with(
        {
            "token": "task-token",
            "runtime_api_address": "runtime.example:9092",
            "insecure": True,
            "root_certificates_path": None,
        },
        socket_path,
        "Model",
    )
