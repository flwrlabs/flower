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
"""Tests for gRPC retry utilities."""


import inspect
import threading
from unittest.mock import Mock, patch

import grpc
import pytest

from .grpc_retry import make_simple_grpc_retry_invoker
from .retry_invoker import RetryState


class _UnauthenticatedError(grpc.RpcError):  # type: ignore[misc]
    """gRPC error reporting an authentication failure."""

    def code(self) -> grpc.StatusCode:
        """Return the gRPC status code."""
        return grpc.StatusCode.UNAUTHENTICATED


class _UnavailableError(grpc.RpcError):  # type: ignore[misc]
    """gRPC error reporting an unavailable endpoint."""

    def code(self) -> grpc.StatusCode:
        """Return the gRPC status code."""
        return grpc.StatusCode.UNAVAILABLE


@patch("flwr.supercore.retry.grpc_retry.os.kill")
def test_unauthenticated_does_not_signal_when_retries_disabled(
    mock_kill: Mock,
) -> None:
    """Late background RPC failures must not interrupt graceful shutdown."""
    retry_invoker = make_simple_grpc_retry_invoker()
    retry_invoker.disable_retries()

    with pytest.raises(_UnauthenticatedError):
        retry_invoker.invoke(Mock(side_effect=_UnauthenticatedError()))

    mock_kill.assert_not_called()


@patch("flwr.supercore.retry.grpc_retry.os.kill")
def test_unauthenticated_signals_when_max_tries_is_one(
    mock_kill: Mock,
) -> None:
    """A configured one-attempt limit must not look like executor shutdown."""
    retry_invoker = make_simple_grpc_retry_invoker()
    retry_invoker.max_tries = 1
    mock_kill.side_effect = lambda *_args: retry_invoker.disable_retries()

    with pytest.raises(_UnauthenticatedError):
        retry_invoker.invoke(Mock(side_effect=_UnauthenticatedError()))

    mock_kill.assert_called_once()


def test_disable_retries_interrupts_grpc_backoff() -> None:
    """Executor shutdown must not wait for an active gRPC retry backoff."""
    retry_invoker = make_simple_grpc_retry_invoker()
    target_called = threading.Event()
    errors: list[grpc.RpcError] = []

    def target() -> None:
        target_called.set()
        raise _UnavailableError

    def invoke() -> None:
        try:
            retry_invoker.invoke(target)
        except grpc.RpcError as err:
            errors.append(err)

    thread = threading.Thread(target=invoke)
    thread.start()
    assert target_called.wait(timeout=1.0)

    retry_invoker.disable_retries()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(errors) == 1


def test_disable_retries_interrupts_all_grpc_backoffs() -> None:
    """Shutdown must wake every invocation sharing the retry coordinator."""
    retry_invoker = make_simple_grpc_retry_invoker()
    retry_invoker.jitter = lambda _wait_time: 60.0
    backoff_barrier = threading.Barrier(3)
    errors: list[grpc.RpcError] = []
    original_on_backoff = retry_invoker.on_backoff
    assert original_on_backoff is not None

    def on_backoff(retry_state: RetryState) -> None:
        original_on_backoff(retry_state)
        backoff_barrier.wait(timeout=1.0)

    retry_invoker.on_backoff = on_backoff

    def target() -> None:
        raise _UnavailableError

    def invoke() -> None:
        try:
            retry_invoker.invoke(target)
        except grpc.RpcError as err:
            errors.append(err)

    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    backoff_barrier.wait(timeout=1.0)

    retry_invoker.disable_retries()
    for thread in threads:
        thread.join(timeout=1.0)

    assert all(not thread.is_alive() for thread in threads)
    assert len(errors) == 2


def test_disable_retries_is_reentrant_from_retry_callback() -> None:
    """A signal during a retry callback must not deadlock shutdown."""
    retry_invoker = make_simple_grpc_retry_invoker()
    on_backoff = retry_invoker.on_backoff
    assert on_backoff is not None
    retry_wait_state = inspect.getclosurevars(on_backoff).nonlocals["retry_wait_state"]
    thread_finished = threading.Event()

    def disable_while_locked() -> None:
        # Simulate SIGINT while a retry callback owns the state condition.
        with retry_wait_state._condition:  # pylint: disable=protected-access
            retry_invoker.disable_retries()
        thread_finished.set()

    thread = threading.Thread(target=disable_while_locked, daemon=True)
    thread.start()
    thread.join(timeout=1.0)

    assert thread_finished.is_set()
    assert not thread.is_alive()
