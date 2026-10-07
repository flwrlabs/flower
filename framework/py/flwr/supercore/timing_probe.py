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
"""Opt-in, metadata-only DEBUG timings for Agent runs.

Monotonic timestamps belong to one clock domain. Wall timestamps are correlation
hints only; they must not be used for cross-process latency subtraction.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import DEBUG
from uuid import uuid4

from flwr.supercore.logger import console_handler, log_timing_probe
from flwr.supercore.task_identity import TaskIdentity
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_dumps

TIMING_ENV = "FLWR_TIMING_LOGGING"
_CLOCK_DOMAIN = uuid4().hex


def timing_enabled() -> bool:
    """Require explicit opt-in and DEBUG console output."""
    return os.getenv(TIMING_ENV) == "1" and console_handler.level <= DEBUG


@dataclass
class TimingProbe:  # pylint: disable=too-many-instance-attributes
    """Record bounded stage boundaries with existing run and task identities."""

    run_id: int | None = None
    task_id: int | None = None
    parent_task_id: int | None = None
    task_type: str | None = None
    fab_hash: str | None = None
    pod_name: str | None = None
    route: str | None = None
    scope_id: str | None = field(default=None, init=False)
    _seen: set[str] = field(default_factory=set, init=False, repr=False)

    @classmethod
    def for_task(cls) -> TimingProbe:
        """Capture identity after PullTaskInput, including in publisher threads."""
        # Unset identity is valid in tests and before the first task input.
        return cls(
            run_id=TaskIdentity._run_id,  # pylint: disable=protected-access
            task_id=TaskIdentity._task_id,  # pylint: disable=protected-access
        )

    # pylint: disable-next=too-many-arguments
    def mark(
        self,
        marker: str,
        *,
        span_id: str | None = None,
        duration_ns: int | None = None,
        event_id: int | None = None,
        success: bool | None = None,
    ) -> None:
        """Emit only fixed metadata fields; never serialize request/event bodies."""
        if not timing_enabled():
            return
        if self.scope_id is None:
            self.scope_id = uuid4().hex
        fab_hash = self.fab_hash
        if fab_hash is not None and (
            len(fab_hash) != 64
            or any(char not in "0123456789abcdef" for char in fab_hash)
        ):
            fab_hash = None
        record: JSONObject = {
            "scope_id": self.scope_id,
            "schema": 1,
            "marker": marker,
            "clock_domain": f"{_CLOCK_DOMAIN}:{os.getpid()}",
            "monotonic_ns": time.monotonic_ns(),
            "unix_time_ns": time.time_ns(),
            "run_id": self.run_id,
            "task_id": self.task_id,
            "parent_task_id": self.parent_task_id,
            "task_type": self.task_type,
            "fab_hash": fab_hash,
            "pod_name": self.pod_name,
            "route": self.route,
            "span_id": span_id,
            "duration_ns": duration_ns,
            "event_id": event_id,
            "success": success,
        }
        try:
            log_timing_probe("timing_probe " + strict_json_dumps(record, compact=True))
        except Exception:  # pylint: disable=broad-exception-caught
            # Observational output must not affect task or stream execution.
            pass

    @contextmanager
    def span(self, stage: str) -> Iterator[None]:
        """Measure a same-process interval without logging exception details."""
        if not timing_enabled():
            yield
            return
        span_id = uuid4().hex
        self.mark(f"{stage}.started", span_id=span_id)
        started = time.monotonic_ns()
        success = False
        try:
            yield
            success = True
        finally:
            self.mark(
                f"{stage}.finished" if success else f"{stage}.failed",
                span_id=span_id,
                duration_ns=time.monotonic_ns() - started,
                success=success,
            )

    def first_event(
        self, stage: str, event_type: str, event_id: int | None = None
    ) -> None:
        """Separate first event, output text and reasoning without per-token logs."""
        markers = [f"{stage}.first_event"]
        if event_type == "response.output_text.delta":
            markers.append(f"{stage}.first_text")
        elif event_type == "response.reasoning_summary_text.delta":
            markers.append(f"{stage}.first_reasoning")
        unseen_markers = [marker for marker in markers if marker not in self._seen]
        if not unseen_markers or not timing_enabled():
            return
        for marker in unseen_markers:
            self.mark_once(marker, event_id=event_id)

    def mark_once(self, marker: str, *, event_id: int | None = None) -> None:
        """Emit one boundary per scope while profiling is enabled."""
        if marker not in self._seen and timing_enabled():
            self._seen.add(marker)
            self.mark(marker, event_id=event_id)
