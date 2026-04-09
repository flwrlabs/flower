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
"""Common function to register signal handlers."""


import signal
from collections.abc import Callable
from threading import Thread
from types import FrameType

from grpc import Server

from flwr.common.telemetry import EventType

from .exit import flwr_exit
from .exit_code import ExitCode
from .exit_handler import add_exit_handler

SIGNAL_TO_EXIT_CODE: dict[int, int] = {
    signal.SIGINT: ExitCode.GRACEFUL_EXIT_SIGINT,
    signal.SIGTERM: ExitCode.GRACEFUL_EXIT_SIGTERM,
}

# SIGQUIT is not available on Windows
if hasattr(signal, "SIGQUIT"):
    SIGNAL_TO_EXIT_CODE[signal.SIGQUIT] = ExitCode.GRACEFUL_EXIT_SIGQUIT


def register_signal_handlers(
    event_type: EventType,
    exit_message: str | None = None,
    grpc_servers: list[Server] | None = None,
    bckg_threads: list[Thread] | None = None,
    exit_handlers: list[Callable[[], None]] | None = None,
) -> None:
    """Register exit handlers for `SIGINT`, `SIGTERM` and `SIGQUIT` signals.

    Parameters
    ----------
    event_type : EventType
        The telemetry event that should be logged before exit.
    exit_message : Optional[str] (default: None)
        The message to be logged before exiting.
    grpc_servers: Optional[List[Server]] (default: None)
        An otpional list of gRPC servers that need to be gracefully
        terminated before exiting.
    bckg_threads: Optional[List[Thread]] (default: None)
        An optional list of threads that need to be gracefully
        terminated before exiting.
    exit_handlers: Optional[List[Callable[[], None]]] (default: None)
        An optional list of exit handlers to be called before exiting.
        Additional exit handlers can be added using `add_exit_handler`.
    """
    default_handlers: dict[int, Callable[[int, FrameType], None]] = {}

    def _wait_to_stop() -> None:
        if grpc_servers is not None:
            for grpc_server in grpc_servers:
                grpc_server.stop(grace=1)

        if bckg_threads is not None:
            for bckg_thread in bckg_threads:
                bckg_thread.join()

    # Ensure that `_wait_to_stop` is the last handler called on exit
    add_exit_handler(_wait_to_stop)

    for handler in exit_handlers or []:
        add_exit_handler(handler)

    def graceful_exit_handler(signalnum: int, _frame: FrameType) -> None:
        """Exit handler to be registered with `signal.signal`.

        When called will reset signal handler to original signal handler from
        default_handlers.
        """
        # Reset to default handler
        signal.signal(signalnum, default_handlers[signalnum])  # type: ignore

        # Setup things for graceful exit
        flwr_exit(
            code=SIGNAL_TO_EXIT_CODE[signalnum],
            message=exit_message,
            event_type=event_type,
        )

    # Register signal handlers
    for sig in SIGNAL_TO_EXIT_CODE:
        default_handler = signal.signal(sig, graceful_exit_handler)  # type: ignore
        default_handlers[sig] = default_handler  # type: ignore
