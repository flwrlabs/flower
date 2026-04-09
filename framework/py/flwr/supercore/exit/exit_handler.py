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
_lock_handlers = threading.Lock()

# SIGQUIT is not available on Windows
if hasattr(signal, "SIGQUIT"):
    SIGNAL_TO_EXIT_CODE[signal.SIGQUIT] = ExitCode.GRACEFUL_EXIT_SIGQUIT


def add_exit_handler(exit_handler: Callable[[], None]) -> None:
    """Add an exit handler to be called on graceful exit.

    This function allows you to register additional exit handlers
    that will be executed when `flwr_exit` is called.

    Parameters
    ----------
    exit_handler : Callable[[], None]
        A callable that takes no arguments and performs cleanup or
        other actions before the application exits.

    Notes
    -----
    The registered exit handlers will be called in LIFO order, i.e.,
    the last registered handler will be the first to be called.
    """
    with _lock_handlers:
        registered_exit_handlers.append(exit_handler)


def trigger_exit_handlers() -> None:
    """Trigger all registered exit handlers in LIFO order."""
    with _lock_handlers:
        for handler in reversed(registered_exit_handlers):
            try:
                handler()
            except Exception:  # pylint: disable=broad-exception-caught
                # Ignore exceptions in exit handlers
                pass
        registered_exit_handlers.clear()
