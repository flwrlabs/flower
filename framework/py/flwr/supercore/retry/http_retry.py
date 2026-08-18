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
"""HTTP retry utilities."""

import os
import signal
import threading
import time
from logging import DEBUG, ERROR, INFO, WARNING

import httpx

from flwr.common.constant import MAX_RETRY_DELAY
from flwr.common.logger import log
from flwr.supercore.constant import FORCE_EXIT_TIMEOUT_SECONDS
from flwr.supercore.run import RunNotRunningException

from .retry_invoker import RetryInvoker, RetryState, exponential

_RETRYABLE_STATUS_CODES = frozenset(
    {
        httpx.codes.SERVICE_UNAVAILABLE,
        httpx.codes.GATEWAY_TIMEOUT,
    }
)
_RETRYABLE_TRANSPORT_ERRORS: tuple[type[httpx.TransportError], ...] = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)


def make_simple_http_retry_invoker() -> RetryInvoker:
    """Create a simple HTTP retry invoker."""
    lock = threading.Lock()
    shutdown_lock = threading.Lock()
    shutdown_requested = False
    system_healthy = threading.Event()
    system_healthy.set()

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
                WARNING,
                "Giving up reconnection after %.2f seconds and %s tries.",
                retry_state.elapsed_time,
                retry_state.tries,
            )

    def _should_giveup(exc: Exception) -> bool:
        nonlocal shutdown_requested
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            if status_code == httpx.codes.FORBIDDEN:
                raise RunNotRunningException
            if status_code == httpx.codes.UNAUTHORIZED:
                with shutdown_lock:
                    if shutdown_requested:
                        return True
                    shutdown_requested = True
                os.kill(os.getpid(), signal.SIGINT)
                time.sleep(FORCE_EXIT_TIMEOUT_SECONDS + 1)
                return False
            return status_code not in _RETRYABLE_STATUS_CODES

        error_message = str(exc).lower()
        if any(term in error_message for term in ("certificate", "ssl", "tls")):
            log(ERROR, "SSL/TLS handshake error detected.")
            return True
        return False

    def _wait(wait_time: float) -> None:
        with lock:
            log(
                WARNING,
                "Connection attempt failed, retrying in %.2f seconds",
                wait_time,
            )
            start = time.monotonic()
            system_healthy.wait(wait_time)

        remaining_time = wait_time - (time.monotonic() - start)
        if remaining_time > 0:
            time.sleep(remaining_time)

    return RetryInvoker(
        wait_gen_factory=lambda: exponential(max_delay=MAX_RETRY_DELAY),
        recoverable_exceptions=(
            *_RETRYABLE_TRANSPORT_ERRORS,
            httpx.HTTPStatusError,
        ),
        max_tries=None,
        max_time=None,
        on_success=_on_success,
        on_backoff=_on_backoff,
        on_giveup=_on_giveup,
        should_giveup=_should_giveup,
        wait_function=_wait,
    )
