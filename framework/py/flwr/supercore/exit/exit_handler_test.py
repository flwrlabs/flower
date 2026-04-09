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


from .exit_handler import add_exit_handler, trigger_exit_handlers


def test_trigger_exit_handlers() -> None:
    """Test that exit handlers are triggered in LIFO order."""
    # Prepare
    execution_order = []

    def handler1() -> None:
        execution_order.append(1)

    def handler2() -> None:
        execution_order.append(2)

    def handler3() -> None:
        execution_order.append(3)

    add_exit_handler(handler1)
    add_exit_handler(handler2)
    add_exit_handler(handler3)

    # Execute
    trigger_exit_handlers()

    # Assert: Handlers should be called in LIFO order (3, 2, 1)
    assert execution_order == [3, 2, 1]


def test_trigger_exit_handlers_clears_list() -> None:
    """Test that trigger_exit_handlers clears the registered handlers."""
    # Prepare
    execution_count = []

    def handler() -> None:
        execution_count.append(1)

    add_exit_handler(handler)

    # Execute & assert
    trigger_exit_handlers()
    assert len(execution_count) == 1

    # Trigger again. The handler should not be called again
    trigger_exit_handlers()
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
    trigger_exit_handlers()

    # Assert - all handlers should have been called in LIFO order
    assert execution_order == [3, 2, 1]
