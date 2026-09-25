# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Common function to register exit handlers."""


import signal
import threading
from collections.abc import Callable
from dataclasses import dataclass

from .exit_code import ExitCode

SIGNAL_TO_EXIT_CODE: dict[int, int] = {
    signal.SIGINT: ExitCode.GRACEFUL_EXIT_SIGINT,
    signal.SIGTERM: ExitCode.GRACEFUL_EXIT_SIGTERM,
}


@dataclass(frozen=True)
class _RegisteredExitHandler:
    """An exit handler and the phase in which it should run."""

    exit_handler: Callable[[], None]
    run_before_force_exit: bool


registered_exit_handlers: list[_RegisteredExitHandler] = []
# Python signal handlers run synchronously on the main thread and can interrupt this
# critical section, so nested exit handling must be able to reacquire the lock.
_lock_handlers = threading.RLock()

# SIGQUIT is not available on Windows
if hasattr(signal, "SIGQUIT"):
    SIGNAL_TO_EXIT_CODE[signal.SIGQUIT] = ExitCode.GRACEFUL_EXIT_SIGQUIT


def add_exit_handler(
    exit_handler: Callable[[], None],
    *,
    run_before_force_exit: bool = False,
) -> None:
    """Add an exit handler to be called on graceful exit.

    This function allows you to register additional exit handlers
    that will be executed when `flwr_exit` is called.

    Parameters
    ----------
    exit_handler : Callable[[], None]
        A callable that takes no arguments and performs cleanup or
        other actions before the application exits.
    run_before_force_exit : bool (default: False)
        Whether to execute the handler before starting the force-exit timer.
        Handlers in this phase must be short and non-blocking because the
        watchdog has not started yet.

    Notes
    -----
    Handlers are called in LIFO order within each phase. Pre-force handlers
    run before the watchdog starts; regular handlers run after it starts.
    """
    with _lock_handlers:
        registered_exit_handlers.append(
            _RegisteredExitHandler(exit_handler, run_before_force_exit)
        )


def trigger_exit_handlers(*, run_before_force_exit: bool) -> None:
    """Trigger one phase of registered exit handlers in LIFO order.

    The pre-force phase runs before the force-exit watchdog starts and must only
    contain short, non-blocking operations. The regular phase runs after the
    watchdog starts and is bounded by that watchdog.

    Handlers selected for this phase are removed before execution, so each is
    invoked at most once. Handlers registered while callbacks run remain for a
    subsequent trigger.
    """
    with _lock_handlers:
        handlers = []
        remaining_handlers = []
        for registered_handler in registered_exit_handlers:
            if registered_handler.run_before_force_exit == run_before_force_exit:
                handlers.append(registered_handler.exit_handler)
            else:
                remaining_handlers.append(registered_handler)
        registered_exit_handlers[:] = remaining_handlers

    # Run callbacks without holding the registry lock. Callbacks may block or
    # register additional handlers.
    for exit_handler in reversed(handlers):
        try:
            exit_handler()
        except Exception:  # pylint: disable=broad-exception-caught
            # Ignore exceptions in exit handlers
            pass
