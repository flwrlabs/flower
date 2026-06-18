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
from dataclasses import dataclass

import grpc

from flwr.common.constant import (
    HEARTBEAT_BASE_MULTIPLIER,
    HEARTBEAT_CALL_TIMEOUT,
    HEARTBEAT_DEFAULT_INTERVAL,
    HEARTBEAT_RANDOM_RANGE,
)
from flwr.common.grpc import create_channel, on_channel_state_change
from flwr.common.retry_invoker import RetryInvoker, exponential

# pylint: disable=E0611
from flwr.proto.appio_pb2 import SendTaskHeartbeatRequest
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.interceptors import (
    AppIoTokenClientInterceptor,
    RuntimeVersionClientInterceptor,
)

# pylint: enable=E0611


class HeartbeatFailure(Exception):
    """Exception raised when a heartbeat fails."""


@dataclass(frozen=True)
class TaskHeartbeatConfig:
    """Configuration for task heartbeat gRPC clients."""

    appio_service_address: str
    insecure: bool
    root_certificates: bytes | None
    token: str
    component_name: str


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


def make_task_heartbeat_fn_grpc(
    stub: ServerAppIoStub | ClientAppIoStub,
) -> Callable[[], bool]:
    """Get the function to send a heartbeat to gRPC endpoint from a task executor.

    Parameters
    ----------
    stub : ServerAppIoStub | ClientAppIoStub
        gRPC stub to send the heartbeat.

    Returns
    -------
    Callable[[], bool]
        Function that sends a heartbeat to the gRPC endpoint.
    """
    # Construct the heartbeat request
    req = SendTaskHeartbeatRequest()

    def fn() -> bool:
        # Call ServerAppIo API
        try:
            res = stub.SendTaskHeartbeat(req, timeout=HEARTBEAT_CALL_TIMEOUT)
        except grpc.RpcError as e:
            status_code = e.code()  # pylint: disable=no-member
            if status_code == grpc.StatusCode.UNAVAILABLE:
                return False
            if status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                return False
            if status_code in (
                grpc.StatusCode.PERMISSION_DENIED,
                grpc.StatusCode.UNAUTHENTICATED,
            ):
                signal.raise_signal(signal.SIGINT)
                return False
            raise

        # Raise SIGINT to trigger graceful shutdown if heartbeat failed
        if not res.success:
            # Never reach here due to token authentication unless race conditions occur
            signal.raise_signal(signal.SIGINT)
        return True

    return fn


class TaskHeartbeat:
    """Task heartbeat connection and sender."""

    def __init__(self, channel: grpc.Channel, sender: HeartbeatSender) -> None:
        self._channel = channel
        self._sender = sender

    def start(self) -> None:
        """Start sending task heartbeats."""
        self._sender.start()

    def close(self) -> None:
        """Stop sending task heartbeats and close the gRPC channel."""
        if self._sender.is_running:
            self._sender.stop()
        self._channel.close()


def create_task_heartbeat_grpc(
    config: TaskHeartbeatConfig,
    stub_class: type[ServerAppIoStub] | type[ClientAppIoStub],
) -> TaskHeartbeat:
    """Create a task heartbeat sender over an unwrapped AppIO stub."""
    channel, stub = _create_task_heartbeat_stub_grpc(config, stub_class)
    return TaskHeartbeat(channel, HeartbeatSender(make_task_heartbeat_fn_grpc(stub)))


def _create_task_heartbeat_stub_grpc(
    config: TaskHeartbeatConfig,
    stub_class: type[ServerAppIoStub] | type[ClientAppIoStub],
) -> tuple[grpc.Channel, ServerAppIoStub | ClientAppIoStub]:
    """Create an unwrapped AppIO stub for task heartbeats."""
    channel = create_channel(
        server_address=config.appio_service_address,
        insecure=config.insecure,
        root_certificates=config.root_certificates,
        interceptors=[
            RuntimeVersionClientInterceptor(component_name=config.component_name),
            AppIoTokenClientInterceptor(config.token),
        ],
    )
    channel.subscribe(on_channel_state_change)
    return channel, stub_class(channel)
