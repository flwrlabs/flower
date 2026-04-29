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
"""Tests for app process utilities."""


import os
import signal
import time
from unittest.mock import Mock, patch

import pytest

from .app_utils import start_lifeline_fd_monitor

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="Lifeline FD monitoring depends on POSIX FD behavior.",
)


def test_lifeline_fd_monitor_raises_sigint_when_writer_closes() -> None:
    """The lifeline monitor should raise SIGINT when the writer closes."""
    read_fd, write_fd = os.pipe()
    raise_signal = Mock()

    with patch("flwr.supercore.app_utils.signal.raise_signal", raise_signal):
        start_lifeline_fd_monitor(read_fd)
        os.close(write_fd)
        _wait_for_signal(raise_signal)

    raise_signal.assert_called_once_with(signal.SIGINT)


def test_lifeline_fd_monitor_does_not_raise_while_writer_is_open() -> None:
    """The lifeline monitor should stay quiet while the writer remains open."""
    read_fd, write_fd = os.pipe()
    raise_signal = Mock()

    with patch("flwr.supercore.app_utils.signal.raise_signal", raise_signal):
        start_lifeline_fd_monitor(read_fd)
        time.sleep(0.05)
        raise_signal.assert_not_called()
        os.close(write_fd)
        _wait_for_signal(raise_signal)


def test_lifeline_fd_monitor_marks_fd_non_inheritable() -> None:
    """The app-side lifeline FD should not be inherited by app grandchildren."""
    read_fd, write_fd = os.pipe()

    try:
        os.set_inheritable(read_fd, True)
        with patch("flwr.supercore.app_utils.signal.raise_signal") as raise_signal:
            start_lifeline_fd_monitor(read_fd)
            assert os.get_inheritable(read_fd) is False
            os.close(write_fd)
            _wait_for_signal(raise_signal)
    finally:
        _close_fd(write_fd)


def test_lifeline_fd_monitor_tolerates_already_closed_fd() -> None:
    """Starting the lifeline monitor on a closed FD should be a no-op."""
    read_fd, write_fd = os.pipe()
    os.close(read_fd)

    try:
        with patch("flwr.supercore.app_utils.signal.raise_signal") as raise_signal:
            start_lifeline_fd_monitor(read_fd)
            time.sleep(0.05)
    finally:
        _close_fd(write_fd)

    raise_signal.assert_not_called()


def _wait_for_signal(raise_signal: Mock) -> None:
    """Wait briefly for the monitor thread to raise the mocked signal."""
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if raise_signal.called:
            return
        time.sleep(0.01)
    raise AssertionError("lifeline monitor did not raise SIGINT")


def _close_fd(fd: int) -> None:
    """Close an FD, ignoring already-closed descriptors."""
    try:
        os.close(fd)
    except OSError:
        pass
