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
"""Heartbeat sender."""


import random
import signal
import threading
from collections.abc import Callable

import grpc

from flwr.common.constant import (
    HEARTBEAT_BASE_MULTIPLIER,
    HEARTBEAT_CALL_TIMEOUT,
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_RANDOM_RANGE,
)
from flwr.common.retry_invoker import RetryInvoker, exponential
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub

# pylint: disable=E0611
from flwr.proto.heartbeat_pb2 import SendAppHeartbeatRequest
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub

# pylint: enable=E0611


class HeartbeatFailure(Exception):
    """Exception raised when a heartbeat fails."""


class HeartbeatSender:
    """Periodically send heartbeat signals to a server in a background thread.

    This class uses the provided `heartbeat_fn` to send heartbeats. If a heartbeat
    attempt fails, it will be retried using an exponential backoff strategy.

    Parameters
    ----------
    heartbeat_fn : Callable[[], bool]
        Function used to send a heartbeat signal. It should return True if the heartbeat
        succeeds, or False if it fails. Any internal exceptions (e.g., gRPC errors)
        should be handled within this function to ensure boolean return values.
    """

    def __init__(
        self,
        heartbeat_fn: Callable[[], bool],
    ) -> None:
        self.heartbeat_fn = heartbeat_fn
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._retry_invoker = RetryInvoker(
            lambda: exponential(max_delay=20),
            HeartbeatFailure,  # The only exception we want to retry on
            max_tries=None,
            max_time=None,
            # Allow the stop event to interrupt the wait
            wait_function=self._stop_event.wait,  # type: ignore
        )

    def start(self) -> None:
        """Start the heartbeat sender."""
        if self._thread.is_alive():
            raise RuntimeError("Heartbeat sender is already running.")
        if self._stop_event.is_set():
            raise RuntimeError("Cannot start a stopped heartbeat sender.")
        self._thread.start()

    def stop(self) -> None:
        """Stop the heartbeat sender."""
        if not self._thread.is_alive():
            raise RuntimeError("Heartbeat sender is not running.")
        self._stop_event.set()
        self._thread.join()

    @property
    def is_running(self) -> bool:
        """Return True if the heartbeat sender is running, False otherwise."""
        return self._thread.is_alive() and not self._stop_event.is_set()

    def _run(self) -> None:
        """Periodically send heartbeats until stopped."""
        while not self._stop_event.is_set():
            # Attempt to send a heartbeat with retry on failure
            self._retry_invoker.invoke(self._heartbeat)

            # Calculate the interval for the next heartbeat
            # Formula: next_interval = (interval - timeout) * random.uniform(0.7, 0.9)
            rd = random.uniform(*HEARTBEAT_RANDOM_RANGE)
            next_interval: float = HEARTBEAT_DEFAULT_INTERVAL - HEARTBEAT_CALL_TIMEOUT
            next_interval *= HEARTBEAT_BASE_MULTIPLIER + rd

            # Wait for the calculated interval or exit early if stopped
            self._stop_event.wait(next_interval)

    def _heartbeat(self) -> None:
        """Send a single heartbeat and raise an exception if it fails.

        Call the provided `heartbeat_fn`. If the function returns False,
        a `HeartbeatFailure` exception is raised to trigger the retry mechanism.
        """
        if not self._stop_event.is_set():
            if not self.heartbeat_fn():
                raise HeartbeatFailure


def make_app_heartbeat_fn_grpc(
    stub: ServerAppIoStub | ClientAppIoStub,
    token: str,
) -> Callable[[], bool]:
    """Get the function to send a heartbeat to gRPC endpoint from an app process.

    Parameters
    ----------
    stub : ServerAppIoStub | ClientAppIoStub
        gRPC stub to send the heartbeat.
    token : str
        The token to use in the heartbeat request.

    Returns
    -------
    Callable[[], bool]
        Function that sends a heartbeat to the gRPC endpoint.
    """
    # Construct the heartbeat request
    req = SendAppHeartbeatRequest(token=token)

    def fn() -> bool:
        # Call ServerAppIo API
        try:
            res = stub.SendAppHeartbeat(req)
        except grpc.RpcError as e:
            status_code = e.code()
            if status_code == grpc.StatusCode.UNAVAILABLE:
                return False
            if status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                return False
            raise

        # Raise SIGINT to trigger graceful shutdown if heartbeat failed
        if not res.success:
            # Never reach here due to token authentication unless race conditions occur
            signal.raise_signal(signal.SIGINT)
        return True

    return fn
