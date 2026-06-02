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
"""Tests for ModelApp process exit handling."""


from __future__ import annotations

import importlib
from queue import Queue
from typing import Any
from unittest.mock import Mock

import pytest

from flwr.common.exit import ExitCode

model_module: Any = importlib.import_module(
    "flwr.supercore.task_process.model.run_model"
)


def test_run_model_uses_task_process_exception_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model task process should not use ServerApp-specific exception codes."""
    flwr_exit = Mock()

    monkeypatch.setattr(model_module, "register_signal_handlers", Mock())
    monkeypatch.setattr(model_module, "flwr_exit", flwr_exit)

    model_module.run_model(
        serverappio_api_address="127.0.0.1:9091",
        log_queue=Queue(),
        token="test-token",
        runtime_dependency_install=False,
    )

    assert flwr_exit.call_args.args[0] == ExitCode.TASK_PROC_EXCEPTION
