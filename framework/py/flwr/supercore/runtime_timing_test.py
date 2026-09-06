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
"""Tests for runtime timing markers."""

import json
from logging import INFO, LogRecord
from unittest.mock import Mock

import pytest

from . import runtime_timing


def test_emit_runtime_timing_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled timing logging should not create a LogRecord."""
    log = Mock()
    monkeypatch.delenv(runtime_timing.RUNTIME_TIMING_LOGGING_ENV, raising=False)
    monkeypatch.setattr("flwr.supercore.runtime_timing.FLOWER_LOGGER.log", log)

    runtime_timing.emit_runtime_timing(
        "runtime.task.claimed",
        component="superlink",
        run_id=1,
        task_id=2,
        task_type="flwr-agentapp",
    )

    log.assert_not_called()


def test_emit_runtime_timing_uses_only_structured_safe_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabled markers should have join fields but no payload-bearing fields."""
    log = Mock()
    monkeypatch.setenv(runtime_timing.RUNTIME_TIMING_LOGGING_ENV, "1")
    monkeypatch.setattr("flwr.supercore.runtime_timing.FLOWER_LOGGER.log", log)
    monkeypatch.setattr(
        "flwr.supercore.runtime_timing.time.time_ns", Mock(return_value=123)
    )
    monkeypatch.setattr(
        "flwr.supercore.runtime_timing.time.monotonic_ns", Mock(return_value=456)
    )

    runtime_timing.emit_runtime_timing(
        "runtime.model.provider.stream.finished",
        component="model_task",
        run_id=10,
        task_id=20,
        parent_task_id=11,
        root_task_id=11,
        task_type="flwr-model",
        outcome="error",
        error_kind="provider",
        executor_mode="fresh",
        process_mode="new",
    )

    expected_attributes = {
        "event": "runtime.model.provider.stream.finished",
        "emitted_at_unix_ns": 123,
        "monotonic_ns": 456,
        "run_id": 10,
        "task_id": 20,
        "parent_task_id": 11,
        "root_task_id": 11,
        "task_type": "flwr-model",
        "component": "model_task",
        "outcome": "error",
        "error_kind": "provider",
        "executor_mode": "fresh",
        "process_mode": "new",
    }
    assert log.call_args.args[:3] == (
        INFO,
        "%s %s",
        runtime_timing.RUNTIME_TIMING_MESSAGE,
    )
    record = LogRecord(
        "flwr",
        INFO,
        "",
        0,
        log.call_args.args[1],
        log.call_args.args[2:],
        None,
    )
    message = record.getMessage()
    assert message.startswith(f"{runtime_timing.RUNTIME_TIMING_MESSAGE} ")
    assert json.loads(message.split(" ", maxsplit=1)[1]) == expected_attributes
    assert log.call_args.kwargs["extra"] == expected_attributes


def test_runtime_task_lineage_is_write_once() -> None:
    """Server-owned lineage must not be overwritten by a later caller."""
    runtime_timing.register_runtime_task_lineage(
        run_id=10,
        task_id=20,
        parent_task_id=11,
        root_task_id=11,
    )
    runtime_timing.register_runtime_task_lineage(
        run_id=10,
        task_id=20,
        parent_task_id=12,
        root_task_id=12,
    )

    assert runtime_timing.get_runtime_task_lineage(run_id=10, task_id=20) == (11, 11)

    runtime_timing.discard_runtime_task_lineage(run_id=10, task_id=20)


def test_first_persisted_event_marker_is_write_once() -> None:
    """Only the first successfully persisted event should be marked."""
    assert runtime_timing.mark_runtime_task_first_event_persisted(run_id=10, task_id=20)
    assert not runtime_timing.mark_runtime_task_first_event_persisted(
        run_id=10, task_id=20
    )

    runtime_timing.discard_runtime_task_lineage(run_id=10, task_id=20)
