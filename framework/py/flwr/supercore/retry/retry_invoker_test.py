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
"""Tests for `RetryInvoker`."""


from collections.abc import Generator
from unittest.mock import MagicMock, Mock, patch

import pytest

from flwr.supercore.retry import RetryInvoker, constant


def successful_function() -> str:
    """."""
    return "success"


def failing_function() -> None:
    """."""
    raise ValueError("failed")


@pytest.fixture(name="mock_time")
def fixture_mock_time() -> Generator[MagicMock, None, None]:
    """Mock time.monotonic for controlled testing."""
    with patch("time.monotonic") as mock_time:
        yield mock_time


@pytest.fixture(name="mock_sleep")
def fixture_mock_sleep() -> Generator[MagicMock, None, None]:
    """Mock sleep to prevent actual waiting during testing."""
    with patch("time.sleep") as mock_sleep:
        yield mock_sleep


def test_successful_invocation() -> None:
    """Ensure successful function invocation."""
    # Prepare
    success_handler = Mock()
    backoff_handler = Mock()
    giveup_handler = Mock()
    invoker = RetryInvoker(
        lambda: constant(0.1),
        ValueError,
        max_tries=None,
        max_time=None,
        on_success=success_handler,
        on_backoff=backoff_handler,
        on_giveup=giveup_handler,
    )

    # Execute
    result = invoker.invoke(successful_function)

    # Assert
    assert result == "success"
    success_handler.assert_called_once()
    backoff_handler.assert_not_called()
    giveup_handler.assert_not_called()


def test_failure() -> None:
    """Check termination when unexpected exception is raised."""
    # Prepare
    # `constant([0.1])` generator will raise `StopIteration` after one iteration.
    invoker = RetryInvoker(lambda: constant(0.1), TypeError, None, None)

    # Execute and Assert
    with pytest.raises(ValueError):
        invoker.invoke(failing_function)


def test_failure_two_exceptions(mock_sleep: MagicMock) -> None:
    """Verify one retry on a specified iterable of exceptions."""
    # Prepare
    invoker = RetryInvoker(
        lambda: constant(0.1),
        (TypeError, ValueError),
        max_tries=2,
        max_time=None,
        jitter=None,
    )

    # Execute and Assert
    with pytest.raises(ValueError):
        invoker.invoke(failing_function)
    mock_sleep.assert_called_once_with(0.1)


def test_backoff_on_failure(mock_sleep: MagicMock) -> None:
    """Verify one retry on specified exception."""
    # Prepare
    # `constant([0.1])` generator will raise `StopIteration` after one iteration.
    invoker = RetryInvoker(lambda: constant([0.1]), ValueError, None, None, jitter=None)

    # Execute and Assert
    with pytest.raises(ValueError):
        invoker.invoke(failing_function)
    mock_sleep.assert_called_once_with(0.1)


def test_max_tries(mock_sleep: MagicMock) -> None:
    """Check termination after `max_tries`."""
    # Prepare
    # Disable `jitter` to ensure 0.1s wait time.
    invoker = RetryInvoker(
        lambda: constant(0.1), ValueError, max_tries=2, max_time=None, jitter=None
    )

    # Execute and Assert
    with pytest.raises(ValueError):
        invoker.invoke(failing_function)
    # Assert 1 sleep call due to the max_tries being set to 2
    mock_sleep.assert_called_once_with(0.1)


def test_max_time(mock_time: MagicMock, mock_sleep: MagicMock) -> None:
    """Check termination after `max_time`."""
    # Prepare
    # Simulate the passage of time using mock
    mock_time.side_effect = [
        0.0,
        3.0,
    ]
    invoker = RetryInvoker(
        lambda: constant(2), ValueError, max_tries=None, max_time=2.5
    )

    # Execute and Assert
    with pytest.raises(ValueError):
        invoker.invoke(failing_function)
    # Assert no wait because `max_time` is exceeded before the first retry.
    mock_sleep.assert_not_called()


def test_event_handlers() -> None:
    """Test `on_backoff` and `on_giveup` triggers."""
    # Prepare
    success_handler = Mock()
    backoff_handler = Mock()
    giveup_handler = Mock()
    invoker = RetryInvoker(
        lambda: constant(0.1),
        ValueError,
        max_tries=2,
        max_time=None,
        on_success=success_handler,
        on_backoff=backoff_handler,
        on_giveup=giveup_handler,
    )

    # Execute and Assert
    with pytest.raises(ValueError):
        invoker.invoke(failing_function)
    backoff_handler.assert_called_once()
    giveup_handler.assert_called_once()
    success_handler.assert_not_called()


def test_giveup_condition() -> None:
    """Verify custom giveup termination."""

    # Prepare
    def should_give_up(exc: Exception) -> bool:
        return isinstance(exc, ValueError)

    invoker = RetryInvoker(
        lambda: constant(0.1), ValueError, None, None, should_giveup=should_give_up
    )

    # Execute and Assert
    with pytest.raises(ValueError):
        invoker.invoke(failing_function)
