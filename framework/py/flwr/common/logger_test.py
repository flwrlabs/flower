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
"""Flower Logger tests."""


import sys
import threading
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from queue import Queue

from .logger import (
    FLOWER_LOGGER,
    configure_superlink_log_file,
    console_handler,
    flush_logs,
    mirror_output_to_queue,
    restore_output,
)


def test_mirror_output_to_queue() -> None:
    """Test that stdout and stderr are mirrored to the provided queue."""
    # Prepare
    log_queue: Queue[str | None] = Queue()

    # Execute
    mirror_output_to_queue(log_queue)
    print("Test message")
    sys.stderr.write("Error message\n")

    # Assert
    assert not log_queue.empty()
    assert log_queue.get() == "Test message"
    assert log_queue.get() == "\n"
    assert log_queue.get() == "Error message\n"


def test_restore_output() -> None:
    """Test that stdout and stderr are restored after calling restore_output."""
    # Prepare
    log_queue: Queue[str | None] = Queue()

    # Execute
    mirror_output_to_queue(log_queue)
    print("Test message before restore")
    restore_output()
    print("Test message after restore")
    sys.stderr.write("Error message after restore\n")

    # Assert
    assert log_queue.get() == "Test message before restore"
    assert log_queue.get() == "\n"
    assert log_queue.empty()


def test_flush_logs_returns_true_when_queue_drains() -> None:
    """Test that flushing succeeds once all queued logs are consumed."""
    # Prepare
    log_queue: Queue[str | None] = Queue()
    log_queue.put("Test message")

    def drain_queue() -> None:
        time.sleep(0.05)
        log_queue.get()

    thread = threading.Thread(target=drain_queue)
    thread.start()

    # Execute
    result = flush_logs(log_queue, timeout=1.0)
    thread.join()

    # Assert
    assert result
    assert log_queue.empty()


def test_flush_logs_returns_true_when_queue_is_empty() -> None:
    """Test that flushing succeeds when there are no queued logs."""
    # Prepare
    log_queue: Queue[str | None] = Queue()

    # Execute
    result = flush_logs(log_queue, timeout=0.01)

    # Assert
    assert result
    assert log_queue.empty()


def test_flush_logs_returns_false_when_queue_does_not_drain() -> None:
    """Test that flushing times out if queued logs are not consumed."""
    # Prepare
    log_queue: Queue[str | None] = Queue()
    log_queue.put("Test message")

    # Execute
    result = flush_logs(log_queue, timeout=0.01)

    # Assert
    assert not result
    assert not log_queue.empty()


def test_configure_superlink_log_file(tmp_path: Path) -> None:
    """Test configuring timed file rotation for SuperLink logs."""
    # Prepare
    file_name = tmp_path / "test-superlink.log"
    path = file_name.resolve()
    before = list(FLOWER_LOGGER.handlers)

    try:
        # Execute
        configure_superlink_log_file(
            filename=str(file_name),
            interval_hours=24,
            backup_count=7,
        )

        # Assert
        rotating_handler = next(
            (
                h
                for h in FLOWER_LOGGER.handlers
                if isinstance(h, TimedRotatingFileHandler)
                and Path(h.baseFilename).resolve() == path  # pylint: disable=no-member
            ),
            None,
        )
        assert rotating_handler is not None
        assert rotating_handler.level == console_handler.level
        assert rotating_handler.backupCount == 7
        assert rotating_handler.interval == 24 * 60 * 60
    finally:
        # Clean up any handlers introduced by this test
        for cleanup_handler in list(FLOWER_LOGGER.handlers):
            if cleanup_handler in before:
                continue
            FLOWER_LOGGER.removeHandler(cleanup_handler)
            cleanup_handler.close()


def test_configure_superlink_log_file_idempotent(tmp_path: Path) -> None:
    """Test configuring SuperLink rotation twice does not duplicate handlers."""
    # Prepare
    file_name = tmp_path / "test-superlink-idempotent.log"
    path = file_name.resolve()
    before = list(FLOWER_LOGGER.handlers)

    try:
        # Execute
        configure_superlink_log_file(
            filename=str(file_name),
            interval_hours=24,
            backup_count=7,
        )
        configure_superlink_log_file(
            filename=str(file_name),
            interval_hours=24,
            backup_count=7,
        )

        # Assert
        handlers = [
            h
            for h in FLOWER_LOGGER.handlers
            if isinstance(h, TimedRotatingFileHandler)
            and Path(h.baseFilename).resolve() == path  # pylint: disable=no-member
        ]
        assert len(handlers) == 1
    finally:
        for handler in list(FLOWER_LOGGER.handlers):
            if handler in before:
                continue
            FLOWER_LOGGER.removeHandler(handler)
            handler.close()
