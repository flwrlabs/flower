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
"""Optional per-invocation SuperExec status observer."""

from __future__ import annotations

from importlib import import_module
from types import TracebackType
from typing import Protocol


class _StatusObserver(Protocol):
    """Interface implemented by optional status observers."""

    def set_phase(self, phase: str) -> None:
        """Observe the next worker phase."""

    def runtime_available(self, available: bool) -> None:
        """Observe the outcome of a Runtime API connection attempt."""

    def close(self) -> None:
        """Release resources owned by this observer."""


class SuperExecStatus:
    """Isolate optional observer notifications from SuperExec execution."""

    def __init__(self, observer: _StatusObserver | None) -> None:
        self._observer = observer

    def set_phase(self, phase: str) -> None:
        """Notify the observer before entering a SuperExec phase."""
        if self._observer is not None:
            try:
                self._observer.set_phase(phase)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def runtime_available(self, available: bool) -> None:
        """Notify the observer of a Runtime connection attempt's outcome."""
        if self._observer is not None:
            try:
                self._observer.runtime_available(available)
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def close(self) -> None:
        """Close the observer at most once, preserving application errors."""
        observer, self._observer = self._observer, None
        if observer is not None:
            try:
                observer.close()
            except Exception:  # pylint: disable=broad-exception-caught
                pass

    def __enter__(self) -> SuperExecStatus:
        """Enter the per-invocation observer scope."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the observer when execution leaves its scope."""
        self.close()


def start_superexec_status() -> SuperExecStatus:
    """Start an optional status observer for one SuperExec invocation.

    Missing extensions are permitted. Errors importing or starting an installed
    extension propagate to the caller.
    """
    module_name = "flwr.ee.supercore.superexec_status"
    try:
        backend = import_module(module_name)
    except ModuleNotFoundError as err:
        if err.name is None or not (
            err.name == module_name or module_name.startswith(f"{err.name}.")
        ):
            raise
        return SuperExecStatus(None)
    return SuperExecStatus(backend.start_superexec_status())
