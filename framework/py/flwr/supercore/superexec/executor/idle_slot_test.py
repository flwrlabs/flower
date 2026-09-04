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
"""Tests for the inert TaskExecutor idle-slot process."""

from pathlib import Path
from threading import Event
from typing import cast
from unittest.mock import Mock

from .idle_slot import _run_idle_slot


def test_run_idle_slot_reports_ready_while_waiting_and_cleans_up(
    tmp_path: Path,
) -> None:
    """Test the marker represents a live process waiting for termination."""
    ready_file = tmp_path / "ready"
    stop_event = Mock()

    def assert_ready() -> None:
        assert ready_file.is_file()

    stop_event.wait.side_effect = assert_ready

    _run_idle_slot(ready_file, cast(Event, stop_event))

    stop_event.wait.assert_called_once_with()
    assert not ready_file.exists()
