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

from .exit_code import ExitCode

SIGNAL_TO_EXIT_CODE: dict[int, int] = {
    signal.SIGINT: ExitCode.GRACEFUL_EXIT_SIGINT,
    signal.SIGTERM: ExitCode.GRACEFUL_EXIT_SIGTERM,
}
registered_exit_handlers: list[Callable[[], None]] = []
# Python signal handlers run synchronously on the main thread and can interrupt this
# critical section, so nested exit handling must be able to reacquire the lock.
_lock_handlers = threading.RLock()

# SIGQUIT is not available on Windows
if hasattr(signal, "SIGQUIT"):
    SIGNAL_TO_EXIT_CODE[signal.SIGQUIT] = ExitCode.GRACEFUL_EXIT_SIGQUIT


def add_exit_handler(
    exit_handler: Callable[[], None], *, run_after_existing: bool = False
) -> None:
    """Add an exit handler to be called on graceful exit.

    This function allows you to register additional exit handlers
    that will be executed when `flwr_exit` is called.

    Parameters
    ----------
    exit_handler : Callable[[], None]
        A callable that takes no arguments and performs cleanup or
        other actions before the application exits.
    run_after_existing : bool (default: False)
        If True, run this handler after all handlers currently registered.

    Notes
    -----
    By default, registered exit handlers are called in LIFO order. A handler
    added with ``run_after_existing=True`` is placed before existing handlers in
    the registry so it runs after them when the registry is reversed.
    """
    with _lock_handlers:
        if run_after_existing:
            registered_exit_handlers.insert(0, exit_handler)
        else:
            registered_exit_handlers.append(exit_handler)


def trigger_exit_handlers() -> None:
    """Trigger registered exit handlers in LIFO order.

    Handlers registered before this call are removed from the registry before
    execution. Each handler is invoked at most once, and handlers registered
    while callbacks are running remain registered for a subsequent call.
    """
    with _lock_handlers:
        handlers = list(reversed(registered_exit_handlers))
        registered_exit_handlers.clear()

    # Run handlers without holding the registry lock. This keeps registration and
    # nested exit handling independent of long-running callbacks.
    for handler in handlers:
        try:
            handler()
        except Exception:  # pylint: disable=broad-exception-caught
            # Ignore exceptions in exit handlers
            pass
