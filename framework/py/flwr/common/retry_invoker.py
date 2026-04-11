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
"""`RetryInvoker` to augment other callables with error handling and retries."""


import itertools
import os
import random
import signal
import threading
import time
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from logging import DEBUG, ERROR, INFO, WARN
from typing import Any, cast

import grpc

from flwr.client.grpc_rere_client.grpc_adapter import GrpcAdapter
from flwr.common.constant import MAX_RETRY_DELAY
from flwr.common.logger import log
from flwr.common.typing import RunNotRunningException
from flwr.proto.clientappio_pb2_grpc import ClientAppIoStub
from flwr.proto.fleet_pb2_grpc import FleetStub
from flwr.proto.serverappio_pb2_grpc import ServerAppIoStub
from flwr.supercore.constant import FORCE_EXIT_TIMEOUT_SECONDS


def exponential(
    base_delay: float = 1,
    multiplier: float = 2,
    max_delay: int | None = None,
) -> Generator[float, None, None]:
    """Wait time generator for exponential backoff strategy.

    Parameters
    ----------
    base_delay: float (default: 1)
        Initial delay duration before the first retry.
    multiplier: float (default: 2)
        Factor by which the delay is multiplied after each retry.
    max_delay: Optional[float] (default: None)
        The maximum delay duration between two consecutive retries.

    Returns
    -------
    Generator[float, None, None]
        A generator for the delay between 2 retries.
    """
    delay = base_delay if max_delay is None else min(base_delay, max_delay)
    while True:
        yield delay
        delay *= multiplier
        if max_delay is not None:
            delay = min(delay, max_delay)


def constant(
    interval: float | Iterable[float] = 1,
) -> Generator[float, None, None]:
    """Wait time generator for specified intervals.

    Parameters
    ----------
    interval: Union[float, Iterable[float]] (default: 1)
        A constant value to yield or an iterable of such values.

    Returns
    -------
    Generator[float, None, None]
        A generator for the delay between 2 retries.
    """
    if not isinstance(interval, Iterable):
        interval = itertools.repeat(interval)
    yield from interval


def full_jitter(max_value: float) -> float:
    """Randomize a float between 0 and the given maximum value.

    This function implements the "Full Jitter" algorithm as described in the
    AWS article discussing the efficacy of different jitter algorithms.
    Reference: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/

    Parameters
    ----------
    max_value : float
        The upper limit for the randomized value.

    Returns
    -------
    float
        A random float that is less than max_value.
    """
    return random.uniform(0, max_value)


@dataclass
class RetryState:
    """State for callbacks in RetryInvoker."""

    target: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    tries: int
    elapsed_time: float
    exception: Exception | None = None
    actual_wait: float | None = None


# pylint: disable-next=too-many-instance-attributes
class RetryInvoker:
    """Wrapper class for retry (with backoff) triggered by exceptions.

    Parameters
    ----------
    wait_gen_factory: Callable[[], Generator[float, None, None]]
        A generator yielding successive wait times in seconds. If the generator
        is finite, the giveup event will be triggered when the generator raises
        `StopIteration`.
    recoverable_exceptions: Union[Type[Exception], Tuple[Type[Exception]]]
        An exception type (or tuple of types) that triggers backoff.
    max_tries: Optional[int]
        The maximum number of attempts to make before giving up. Once exhausted,
        the exception will be allowed to escape. If set to None, there is no limit
        to the number of tries.
    max_time: Optional[float]
        The maximum total amount of time to try before giving up. Once this time
        has expired, this method won't be interrupted immediately, but the exception
        will be allowed to escape. If set to None, there is no limit to the total time.
    on_success: Optional[Callable[[RetryState], None]] (default: None)
        A callable to be executed in the event of success. The parameter is a
        data class object detailing the invocation.
    on_backoff: Optional[Callable[[RetryState], None]] (default: None)
        A callable to be executed in the event of a backoff. The parameter is a
        data class object detailing the invocation.
    on_giveup: Optional[Callable[[RetryState], None]] (default: None)
        A callable to be executed in the event that `max_tries` or `max_time` is
        exceeded, `should_giveup` returns True, or `wait_gen_factory()` generator raises
        `StopInteration`. The parameter is a data class object detailing the
        invocation.
    jitter: Optional[Callable[[float], float]] (default: full_jitter)
        A function of the value yielded by `wait_gen_factory()` returning the actual
        time to wait. This function helps distribute wait times stochastically to avoid
        timing collisions across concurrent clients. Wait times are jittered by
        default using the `full_jitter` function. To disable jittering, pass
        `jitter=None`.
    should_giveup: Optional[Callable[[Exception], bool]] (default: None)
        A function accepting an exception instance, returning whether or not
        to give up prematurely before other give-up conditions are evaluated.
        If set to None, the strategy is to never give up prematurely.
    wait_function: Optional[Callable[[float], None]] (default: None)
        A function that defines how to wait between retry attempts. It accepts
        one argument, the wait time in seconds, allowing the use of various waiting
        mechanisms (e.g., asynchronous waits or event-based synchronization) suitable
        for different execution environments. If set to `None`, the `wait_function`
        defaults to `time.sleep`, which is ideal for synchronous operations. Custom
        functions should manage execution flow to prevent blocking or interference.

    Examples
    --------
    Initialize a `RetryInvoker` with exponential backoff and invoke a function::

        invoker = RetryInvoker(
            exponential,  # Or use `lambda: exponential(3, 2)` to pass arguments
            grpc.RpcError,
            max_tries=3,
            max_time=None,
        )
        invoker.invoke(my_func, arg1, arg2, kw1=kwarg1)
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        wait_gen_factory: Callable[[], Generator[float, None, None]],
        recoverable_exceptions: type[Exception] | tuple[type[Exception], ...],
        max_tries: int | None,
        max_time: float | None,
        *,
        on_success: Callable[[RetryState], None] | None = None,
        on_backoff: Callable[[RetryState], None] | None = None,
        on_giveup: Callable[[RetryState], None] | None = None,
        jitter: Callable[[float], float] | None = full_jitter,
        should_giveup: Callable[[Exception], bool] | None = None,
        wait_function: Callable[[float], None] | None = None,
    ) -> None:
        self.wait_gen_factory = wait_gen_factory
        self.recoverable_exceptions = recoverable_exceptions
        self.max_tries = max_tries
        self.max_time = max_time
        self.on_success = on_success
        self.on_backoff = on_backoff
        self.on_giveup = on_giveup
        self.jitter = jitter
        self.should_giveup = should_giveup
        if wait_function is None:
            wait_function = time.sleep
        self.wait_function = wait_function

    # pylint: disable-next=too-many-locals
    def invoke(
        self,
        target: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Safely invoke the provided callable with retry mechanisms.

        This method attempts to invoke the given callable, and in the event of
        a recoverable exception, employs a retry mechanism that considers
        wait times, jitter, maximum attempts, and maximum time. During the
        retry process, various callbacks (`on_backoff`, `on_success`, and
        `on_giveup`) can be triggered based on the outcome.

        Parameters
        ----------
        target: Callable[..., Any]
            The callable to be invoked.
        *args: Tuple[Any, ...]
            Positional arguments to pass to `target`.
        **kwargs: Dict[str, Any]
            Keyword arguments to pass to `target`.

        Returns
        -------
        Any
            The result of the given callable invocation.

        Raises
        ------
        Exception
            If the number of tries exceeds `max_tries`, if the total time exceeds
            `max_time`, if `wait_gen_factory()` generator raises `StopInteration`,
            or if the `should_giveup` returns True for a raised exception.

        Notes
        -----
        The time between retries is determined by the provided `wait_gen_factory()`
        generator and can optionally be jittered using the `jitter` function.
        The recoverable exceptions that trigger a retry, as well as conditions to
        stop retries, are also determined by the class's initialization parameters.
        """

        def try_call_event_handler(
            handler: Callable[[RetryState], None] | None,
        ) -> None:
            if handler is not None:
                handler(cast(RetryState, ref_state[0]))

        try_cnt = 0
        wait_generator = self.wait_gen_factory()
        start = time.monotonic()
        ref_state: list[RetryState | None] = [None]

        while True:
            try_cnt += 1
            elapsed_time = time.monotonic() - start
            state = RetryState(
                target=target,
                args=args,
                kwargs=kwargs,
                tries=try_cnt,
                elapsed_time=elapsed_time,
            )
            ref_state[0] = state

            try:
                ret = target(*args, **kwargs)
            except self.recoverable_exceptions as err:
                state.exception = err
                # Check if giveup event should be triggered
                max_tries_exceeded = try_cnt == self.max_tries
                max_time_exceeded = (
                    self.max_time is not None and elapsed_time >= self.max_time
                )

                def giveup_check(_exception: Exception) -> bool:
                    if self.should_giveup is None:
                        return False
                    return self.should_giveup(_exception)

                if giveup_check(err) or max_tries_exceeded or max_time_exceeded:
                    # Trigger giveup event
                    try_call_event_handler(self.on_giveup)
                    raise

                try:
                    wait_time = next(wait_generator)
                    if self.jitter is not None:
                        wait_time = self.jitter(wait_time)
                    if self.max_time is not None:
                        wait_time = min(wait_time, self.max_time - elapsed_time)
                    state.actual_wait = wait_time
                except StopIteration:
                    # Trigger giveup event
                    try_call_event_handler(self.on_giveup)
                    raise err from None

                # Trigger backoff event
                try_call_event_handler(self.on_backoff)

                # Sleep
                self.wait_function(state.actual_wait)
            else:
                # Trigger success event
                try_call_event_handler(self.on_success)
                return ret


def make_simple_grpc_retry_invoker() -> RetryInvoker:
    """Create a simple gRPC retry invoker."""
    lock = threading.Lock()
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
        if e.code() == grpc.StatusCode.PERMISSION_DENIED:  # type: ignore
            raise RunNotRunningException
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:  # type: ignore
            # Authentication failures should trigger shutdown rather than retrying
            # This can occur, for example, when the user runs `flwr stop`
            # Note: On Windows, `os.kill` terminates the process abruptly, not ideal
            # Note: `signal.raise_signal` is not effective in `flwr-simulation`
            os.kill(os.getpid(), signal.SIGINT)
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
    stub: ServerAppIoStub | ClientAppIoStub | FleetStub | GrpcAdapter,
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
