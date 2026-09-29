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
"""Wake local Runtime requests after new task work becomes available."""

from asyncio import AbstractEventLoop, Event, get_running_loop
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

TASK_AVAILABLE_SESSION_KEY = "flwr_task_available"

_waiters: set[tuple[AbstractEventLoop, Event]] = set()
_waiters_lock = Lock()


@contextmanager
def subscribe_to_task_notifications() -> Iterator[Event]:
    """Register an asyncio waiter before checking task availability."""
    event = Event()
    waiter = (get_running_loop(), event)
    with _waiters_lock:
        _waiters.add(waiter)
    try:
        yield event
    finally:
        with _waiters_lock:
            _waiters.discard(waiter)


def notify_task_available() -> None:
    """Wake requests in this process; other processes use a bounded recheck."""
    with _waiters_lock:
        for loop, event in _waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:
                # A request can close its event loop as a task commits.
                continue
