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
"""Optional, metadata-only tracing with no core SDK dependency.

Set ``FLWR_TRACING_ENABLED=1`` in services and workers to use the installed
``flwr.ee.supercore.tracing`` backend. Backend failures do not affect execution.
Application exceptions retain their type, without exporting messages or bodies.

Validated version-00 HTTP carriers preserve direct request context. Independent
task traces correlate through run/task metadata, not persisted parent context.
The first output-text delta records ``provider.first_text``, excluding reasoning
and non-streamed responses. Span end and ERROR status record completion/failure.
Export is best effort with bounded flushing and metadata-only attributes.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module
from time import monotonic_ns, time_ns
from types import ModuleType
from typing import Any

TraceScalar = str | bool | int | float
_TRACEPARENT = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})\Z")


def validate_traceparent(value: object) -> str:
    """Return a valid version-00 carrier or an empty string."""
    if not isinstance(value, str):
        return ""
    match = _TRACEPARENT.fullmatch(value)
    if match is None or int(match[1], 16) == 0 or int(match[2], 16) == 0:
        return ""
    return value


@lru_cache(maxsize=1)
def _load_backend() -> ModuleType | None:
    try:
        return import_module("flwr.ee.supercore.tracing")
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _backend() -> ModuleType | None:
    return _load_backend() if os.getenv("FLWR_TRACING_ENABLED") == "1" else None


def initialize_tracing(component: str) -> None:
    """Prepare optional tracing before worker readiness, without creating a span."""
    backend = _backend()
    if backend is not None:
        try:
            backend.initialize_tracing(component)
        except Exception:  # pylint: disable=broad-exception-caught
            pass


class TraceSpan:
    """Isolate optional tracing failures from application work."""

    def __init__(self, span: Any = None) -> None:
        self._span = span

    def set_attribute(self, name: str, value: TraceScalar) -> None:
        """Attach one caller-selected metadata scalar."""
        if self._span is not None:
            try:
                self._span.set_attribute(name, value)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def add_event(self, name: str) -> None:
        """Record a fixed milestone without an event payload."""
        if self._span is not None:
            try:
                self._span.add_event(name)
            except Exception:  # pylint: disable=broad-exception-caught
                pass


class PhaseTimings:
    """Retain measured phases until their run/task identity is available."""

    def __init__(self) -> None:
        self._enabled = os.getenv("FLWR_TRACING_ENABLED") == "1"
        self._wall = time_ns() if self._enabled else 0
        self._clock = monotonic_ns() if self._enabled else 0
        self._phases: list[tuple[str, int, int]] = []

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Measure elapsed time without importing or invoking a tracing SDK."""
        if not self._enabled:
            yield
            return
        start = self._wall + monotonic_ns() - self._clock
        try:
            yield
        finally:
            end = self._wall + monotonic_ns() - self._clock
            self._phases.append((name, start, end))

    def publish(self, name: str, attributes: Mapping[str, TraceScalar]) -> None:
        """Export the completed group once, with original measured timestamps."""
        phases, self._phases = self._phases, []
        if not phases:
            return
        backend = _backend()
        if backend is not None:
            try:
                backend.record_phase_spans(name, phases, dict(attributes))
            except Exception:  # pylint: disable=broad-exception-caught
                pass


@contextmanager
def trace_span(
    name: str,
    *,
    traceparent: str = "",
    attributes: Mapping[str, TraceScalar] | None = None,
) -> Iterator[TraceSpan]:
    """Scope metadata-only tracing without affecting application exceptions."""
    manager: Any = None
    span = TraceSpan()
    backend = _backend()
    if backend is not None:
        try:
            manager = backend.trace_span(
                name,
                traceparent=validate_traceparent(traceparent),
                attributes=dict(attributes or {}),
            )
            # Enter separately so backend failures do not catch application work.
            span = TraceSpan(
                manager.__enter__()  # pylint: disable=unnecessary-dunder-call
            )
        except Exception:  # pylint: disable=broad-exception-caught
            manager = None
    try:
        yield span
    except BaseException as error:
        span.set_attribute("error.type", type(error).__name__)
        raise
    finally:
        if manager is not None:
            try:
                # Do not hand application exceptions to an SDK that might capture
                # their messages, tracebacks or provider response bodies.
                manager.__exit__(None, None, None)
            except Exception:  # pylint: disable=broad-exception-caught
                pass


def current_traceparent() -> str:
    """Return the current validated carrier when tracing is enabled."""
    backend = _backend()
    if backend is not None:
        try:
            return validate_traceparent(backend.current_traceparent())
        except Exception:  # pylint: disable=broad-exception-caught
            pass
    return ""


def flush_traces() -> None:
    """Ask the private backend to flush with its bounded timeout, best effort."""
    backend = _backend()
    if backend is not None:
        try:
            backend.flush_traces()
        except Exception:  # pylint: disable=broad-exception-caught
            pass
