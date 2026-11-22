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
"""Tests for exit handler functions."""


from collections.abc import Iterator

import pytest

from .exit_handler import (
    _lock_handlers,
    add_exit_handler,
    registered_exit_handlers,
    trigger_exit_handlers,
)


@pytest.fixture(autouse=True)
def clear_exit_handlers() -> Iterator[None]:
    """Clear exit handlers before and after each test."""
    registered_exit_handlers.clear()
    yield
    registered_exit_handlers.clear()


def test_trigger_exit_handlers_in_separate_phases() -> None:
    """Test phase selection and LIFO ordering within each phase."""
    # Prepare
    execution_order = []

    def handler1() -> None:
        execution_order.append(1)

    def handler2() -> None:
        execution_order.append(2)

    def handler3() -> None:
        execution_order.append(3)

    def handler4() -> None:
        execution_order.append(4)

    add_exit_handler(handler1, run_before_force_exit=True)
    add_exit_handler(handler2)
    add_exit_handler(handler3, run_before_force_exit=True)
    add_exit_handler(handler4)

    # Execute
    trigger_exit_handlers(run_before_force_exit=True)

    # Assert: Only pre-force handlers should run, in LIFO order
    assert execution_order == [3, 1]

    # Execute the regular phase
    trigger_exit_handlers(run_before_force_exit=False)

    # Assert: The regular handler was retained for its phase
    assert execution_order == [3, 1, 4, 2]


def test_trigger_exit_handlers_clears_list() -> None:
    """Test that trigger_exit_handlers clears the registered handlers."""
    # Prepare
    execution_count = []

    def handler() -> None:
        execution_count.append(1)

    add_exit_handler(handler)

    # Execute & assert
    trigger_exit_handlers(run_before_force_exit=False)
    assert len(execution_count) == 1

    # Trigger again. The handler should not be called again
    trigger_exit_handlers(run_before_force_exit=False)
    assert len(execution_count) == 1


def test_trigger_exit_handlers_ignores_exceptions() -> None:
    """Test that exceptions in handlers are ignored and other handlers run."""
    # Prepare
    execution_order = []

    def handler1() -> None:
        execution_order.append(1)

    def handler2_raises() -> None:
        execution_order.append(2)
        raise ValueError("Test exception")

    def handler3() -> None:
        execution_order.append(3)

    add_exit_handler(handler1)
    add_exit_handler(handler2_raises)
    add_exit_handler(handler3)

    # Execute - should not raise despite handler2 raising
    trigger_exit_handlers(run_before_force_exit=False)

    # Assert - all handlers should have been called in LIFO order
    assert execution_order == [3, 2, 1]


def test_trigger_exit_handlers_does_not_run_handler_twice() -> None:
    """A nested trigger does not invoke an already-snapshotted handler again."""
    execution_order = []

    def handler() -> None:
        execution_order.append(1)
        trigger_exit_handlers(run_before_force_exit=False)
        execution_order.append(2)

    add_exit_handler(handler)

    trigger_exit_handlers(run_before_force_exit=False)

    assert execution_order == [1, 2]


def test_exit_handler_registry_lock_is_reentrant() -> None:
    """The registry lock supports same-thread re-entry."""
    with _lock_handlers:
        # pylint: disable-next=consider-using-with
        reacquired = _lock_handlers.acquire(blocking=False)
        if reacquired:
            _lock_handlers.release()

    assert reacquired
