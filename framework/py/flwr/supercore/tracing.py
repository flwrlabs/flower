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
"""Optional, metadata-only tracing through a private process-local backend.

Set ``FLWR_TRACING_ENABLED=1`` in SuperLink, SuperExec and task workers to use
``flwr.ee.supercore.tracing`` when installed. Core has no tracing SDK or exporter
dependency. The backend implements ``trace_span``, ``current_traceparent`` and
``flush_traces`` with the signatures below; flushing must use a bounded timeout.

``run.create`` covers persistent run creation, not the full lifetime of a run.
Its validated version-00 carrier is stored on the primary task and returned in
task acquisition and input. Acquisition, dispatch and worker spans can be
siblings under that creation context. Child model creation restores the
authenticated task context, so ordinary Responses clients need no extra headers.
This first slice establishes a shared trace ID rather than exact stack nesting.

Provider milestones record the first output-text delta, terminal completion or
error once. Reasoning deltas do not count as first text; non-streamed requests
have no first-text milestone. Caller-selected attributes contain only metadata,
never request/response bodies, credentials, URLs or exception messages. Backend
errors are ignored and application exceptions are passed back unchanged.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import Any

TraceScalar = str | bool | int | float
_TRACEPARENT = re.compile(r"00-([0-9a-f]{32})-([0-9a-f]{16})-(00|01)\Z")


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
