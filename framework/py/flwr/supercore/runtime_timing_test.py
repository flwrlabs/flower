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
"""Tests for opt-in runtime timing markers."""

import json
from logging import INFO
from unittest.mock import patch

import pytest

from . import runtime_timing


def test_emit_runtime_timing_is_opt_in_and_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit only enabled markers and never propagate a logging failure."""
    monkeypatch.delenv(runtime_timing.RUNTIME_TIMING_LOGGING_ENV, raising=False)

    with patch("flwr.supercore.runtime_timing.FLOWER_LOGGER.log") as log:
        runtime_timing.emit_runtime_timing(
            "runtime.example", run_id=7, task_id=11, root_task_id=11
        )

        log.assert_not_called()

        monkeypatch.setenv(runtime_timing.RUNTIME_TIMING_LOGGING_ENV, "1")
        with (
            patch("flwr.supercore.runtime_timing.time.time_ns", return_value=100),
            patch("flwr.supercore.runtime_timing.time.monotonic_ns", return_value=200),
        ):
            runtime_timing.emit_runtime_timing(
                "runtime.example", run_id=7, task_id=11, root_task_id=11
            )

        assert log.call_args.args[:3] == (INFO, "%s %s", "runtime.timing")
        assert json.loads(log.call_args.args[3]) == {
            "emitted_at_unix_ns": 100,
            "event": "runtime.example",
            "monotonic_ns": 200,
            "root_task_id": 11,
            "run_id": 7,
            "task_id": 11,
        }
        assert log.call_args.kwargs == {"extra": {"runtime_timing": True}}

        log.side_effect = RuntimeError("logging failed")
        runtime_timing.emit_runtime_timing(
            "runtime.example", run_id=7, task_id=11, root_task_id=11
        )
