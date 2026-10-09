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
"""Privacy, filtering and forwarding contracts for runtime timing probes."""

import json

# pylint: disable=protected-access
import sys
from io import StringIO
from logging import DEBUG, INFO
from queue import Queue
from typing import Any
from unittest.mock import Mock

import pytest

from flwr.supercore import timing_probe
from flwr.supercore.logger import console_handler, mirror_output_to_queue
from flwr.supercore.superexec.executor.warm_executor_dispatch import (
    KubernetesWarmExecutorDispatch,
)
from flwr.supercore.task_worker_protocol import relay_task_output
from flwr.supercore.timing_probe import TimingProbe


@pytest.fixture(name="records")
def captured_records(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture only timing JSON while enabling the profiling flag and DEBUG."""
    captured: list[dict[str, Any]] = []
    monkeypatch.setenv("FLWR_TIMING_LOGGING", "1")
    monkeypatch.setattr(console_handler, "level", DEBUG)
    monkeypatch.setattr(
        timing_probe,
        "log_timing_probe",
        lambda value: captured.append(json.loads(value.removeprefix("timing_probe "))),
    )
    return captured


@pytest.mark.parametrize(("flag", "level"), [(None, DEBUG), ("0", DEBUG), ("1", INFO)])
def test_disabled_has_no_output_or_clocks(
    monkeypatch: pytest.MonkeyPatch, flag: str | None, level: int
) -> None:
    """Disabled probes do not read clocks or serialize metadata."""
    if flag is None:
        monkeypatch.delenv("FLWR_TIMING_LOGGING", raising=False)
    else:
        monkeypatch.setenv("FLWR_TIMING_LOGGING", flag)
    monkeypatch.setattr(console_handler, "level", level)
    clock = Mock(side_effect=AssertionError("disabled probe read a clock"))
    output = Mock()
    monkeypatch.setattr("flwr.supercore.timing_probe.time.monotonic_ns", clock)
    monkeypatch.setattr(timing_probe, "log_timing_probe", output)
    timing = TimingProbe(run_id=7, task_id=11)
    with timing.span("agent.user_code"):
        timing.first_event("agent.events", "response.output_text.delta")
    output.assert_not_called()
    clock.assert_not_called()
    assert not timing._seen


def test_span_order_correlation_and_failure_redaction(
    records: list[dict[str, Any]],
) -> None:
    """Failed intervals retain correlation without exception text or authority."""
    timing = TimingProbe(run_id=7, task_id=22, parent_task_id=11)
    with pytest.raises(ValueError):
        with timing.span("model.provider_post"):
            raise ValueError("secret prompt token response-body")
    assert [record["marker"] for record in records] == [
        "model.provider_post.started",
        "model.provider_post.failed",
    ]
    start, end = records
    assert start["span_id"] == end["span_id"]
    assert start["clock_domain"] == end["clock_domain"]
    assert start["scope_id"] == end["scope_id"]
    assert int(end["duration_ns"]) >= 0
    assert int(end["monotonic_ns"]) >= int(start["monotonic_ns"])
    assert int(end["unix_time_ns"]) > 0
    assert end["success"] is False
    assert (end["run_id"], end["task_id"], end["parent_task_id"]) == ("7", "22", "11")
    assert "secret" not in json.dumps(records)


def test_first_events_are_bounded_and_separate_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    records: list[dict[str, Any]],
) -> None:
    """Report distinct first events, then skip enablement checks for repeated events."""
    timing = TimingProbe(run_id=7, task_id=11)
    timing.first_event("client.received", "response.reasoning_summary_text.delta", 1)
    timing.first_event("client.received", "response.output_text.delta", 2)
    enabled = Mock(side_effect=AssertionError("already-seen marker checked enablement"))
    monkeypatch.setattr(timing_probe, "timing_enabled", enabled)
    for _ in range(100):
        timing.first_event(
            "client.received", "response.reasoning_summary_text.delta", 1
        )
        timing.first_event("client.received", "response.output_text.delta", 2)
        timing.mark_once("client.received.first_text")
    enabled.assert_not_called()
    assert [record["marker"] for record in records] == [
        "client.received.first_event",
        "client.received.first_reasoning",
        "client.received.first_text",
    ]
    assert [record["event_id"] for record in records] == ["1", "1", "2"]


def test_probe_output_failure_does_not_fail_task(
    monkeypatch: pytest.MonkeyPatch, records: list[dict[str, Any]]
) -> None:
    """Closed stdout or a failing handler must not change task execution."""
    del records
    monkeypatch.setattr(
        timing_probe, "log_timing_probe", Mock(side_effect=OSError("closed"))
    )
    with TimingProbe(run_id=7).span("agent.user_code"):
        pass


def test_markers_bypass_task_upload_and_survive_worker_relay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use standard DEBUG logging with the current resident worker output path."""
    monkeypatch.setenv("FLWR_TIMING_LOGGING", "1")
    monkeypatch.setattr(console_handler, "level", DEBUG)
    stream = StringIO()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", StringIO())
    monkeypatch.setattr(console_handler, "stream", stream)
    queue: Queue[str | None] = Queue()
    mirror_output_to_queue(queue)
    TimingProbe(run_id=7, task_id=11).mark("agent.input_ready")
    assert "timing_probe {" in stream.getvalue()
    assert queue.empty()
    print("normal task output")
    assert queue.get() == "normal task output"
    assert queue.get() == "\n"
    sender = Mock()
    with relay_task_output(sender, "secret-token"):
        TimingProbe(run_id=7, task_id=11).mark("agent.preloaded")
    relayed = "".join(call.args[1] for call in sender.write.call_args_list)
    assert "timing_probe {" in relayed
    assert "secret-token" not in relayed
    assert queue.empty()


def test_forwarded_markers_stay_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    """Warm output forwarding must not promote DEBUG timing markers to INFO."""
    dispatch = KubernetesWarmExecutorDispatch(
        Mock(), pod_name="pod-1", task_id=11, log_output=True
    )
    with caplog.at_level(DEBUG, logger="flwr"):
        dispatch._log_output_line(
            "stdout", 'DEBUG: timing_probe {"marker":"agent.preloaded"}'
        )  # pylint: disable=protected-access
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == DEBUG


def test_invalid_fab_metadata_is_not_serialized(records: list[dict[str, Any]]) -> None:
    """Preserve uint64 IDs without exporting arbitrary FAB metadata."""
    TimingProbe(
        run_id=2**64 - 1, fab_hash="private credential disguised as hash"
    ).mark("runtime.child_created")
    assert records[0]["run_id"] == "18446744073709551615"
    assert records[0]["fab_hash"] is None
    assert "private" not in json.dumps(records)
