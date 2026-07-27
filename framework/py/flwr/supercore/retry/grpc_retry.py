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
"""gRPC retry utilities."""


import os
import signal
import threading
import time
from logging import DEBUG, ERROR, INFO, WARN
from typing import Any

import grpc

from flwr.common.constant import MAX_RETRY_DELAY
from flwr.common.logger import log
from flwr.supercore.constant import FORCE_EXIT_TIMEOUT_SECONDS
from flwr.supercore.run import RunNotRunningException

from .retry_invoker import RetryInvoker, RetryState, exponential


def make_simple_grpc_retry_invoker() -> RetryInvoker:
    """Create a simple gRPC retry invoker."""
    lock = threading.Lock()
    shutdown_lock = threading.Lock()
    shutdown_requested = False
    system_healthy = threading.Event()
    system_healthy.set()  # Initially, the connection is healthy

    def _on_success(retry_state: RetryState) -> None:
        system_healthy.set()
        if retry_state.tries > 1:
            log(
                INFO,
                "Connection successful after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )

    def _on_backoff(retry_state: RetryState) -> None:
        system_healthy.clear()
        log(
            DEBUG, "Connection attempt failed with exception: %s", retry_state.exception
        )

    def _on_giveup(retry_state: RetryState) -> None:
        system_healthy.clear()
        if retry_state.tries > 1:
            log(
                WARN,
                "Giving up reconnection after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )

    def _should_giveup_fn(e: Exception) -> bool:
        nonlocal shutdown_requested
        if e.code() == grpc.StatusCode.PERMISSION_DENIED:  # type: ignore
            raise RunNotRunningException
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:  # type: ignore
            # Authentication failures should trigger shutdown rather than retrying
            # This can occur, for example, when the user runs `flwr stop`
            # Note: On Windows, `os.kill` terminates the process abruptly, not ideal
            # Note: `signal.raise_signal` is not effective in `flwr-simulation`
            with shutdown_lock:
                if not shutdown_requested:
                    os.kill(os.getpid(), signal.SIGINT)
                    shutdown_requested = True
            time.sleep(FORCE_EXIT_TIMEOUT_SECONDS + 1)
            return False
        if e.code() == grpc.StatusCode.UNAVAILABLE:  # type: ignore
            # Check if this is an SSL handshake failure - these should fail fast
            details = str(e.details() if hasattr(e, "details") else "").lower()
            if "handshake failed" in details:
                log(ERROR, "SSL/TLS handshake error detected.")
                return True  # Give up on SSL/TLS handshake errors
            return False  # Retry on other UNAVAILABLE errors (network issues)
        return True

    def _wait(wait_time: float) -> None:
        # Use a lock to prevent multiple gRPC calls from retrying concurrently,
        # which is unnecessary since they are all likely to fail.
        with lock:
            # Log the wait time
            log(
                WARN,
                "Connection attempt failed, retrying in %.2f seconds",
                wait_time,
            )

            start = time.monotonic()
            # Avoid sequential waits if the system is healthy
            system_healthy.wait(wait_time)

        remaining_time = wait_time - (time.monotonic() - start)
        if remaining_time > 0:
            time.sleep(remaining_time)

    return RetryInvoker(
        wait_gen_factory=lambda: exponential(max_delay=MAX_RETRY_DELAY),
        recoverable_exceptions=grpc.RpcError,
        max_tries=None,
        max_time=None,
        on_success=_on_success,
        on_backoff=_on_backoff,
        on_giveup=_on_giveup,
        should_giveup=_should_giveup_fn,
        wait_function=_wait,
    )


def wrap_stub(
    stub: object,
    retry_invoker: RetryInvoker,
) -> None:
    """Wrap a gRPC stub with a retry invoker."""

    def make_lambda(original_method: Any) -> Any:
        return lambda *args, **kwargs: retry_invoker.invoke(
            original_method, *args, **kwargs
        )

    for method_name in vars(stub):
        method = getattr(stub, method_name)
        if callable(method):
            setattr(stub, method_name, make_lambda(method))
