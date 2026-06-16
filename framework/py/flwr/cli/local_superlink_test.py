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
"""Tests for local SuperLink runtime helpers."""


import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from flwr.cli.constant import (
    LOCAL_CONTROL_API_ADDRESS,
    LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE,
    LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE_IN_MEMORY,
)
from flwr.cli.typing import SuperLinkConnection, SuperLinkSimulationOptions
from flwr.supercore.constant import FLWR_DISABLE_UPDATE_CHECK

from .local_superlink import _start_local_superlink, ensure_local_superlink

_IS_STARTED_PATH = "flwr.cli.local_superlink._is_local_superlink_started"
_START_PATH = "flwr.cli.local_superlink._start_local_superlink"


def test_magic_address_connection_uses_local_superlink() -> None:
    """Magic-address connections are mapped to the managed local SuperLink."""
    connection = SuperLinkConnection(
        name="local",
        address=LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE,
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch(_IS_STARTED_PATH, return_value=True) as mock_is_started:
        with patch(_START_PATH) as mock_start:
            resolved = ensure_local_superlink(connection)

    assert resolved.address == LOCAL_CONTROL_API_ADDRESS
    assert resolved.insecure is True
    assert resolved.root_certificates is None
    mock_is_started.assert_called_once()
    mock_start.assert_not_called()


def test_magic_address_connection_starts_local_superlink_when_unavailable() -> None:
    """Local SuperLink is started when it is unavailable."""
    connection = SuperLinkConnection(
        name="local",
        address=LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE,
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch(_IS_STARTED_PATH, return_value=False):
        with patch(_START_PATH) as mock_start:
            resolved = ensure_local_superlink(connection)

    assert resolved.address == LOCAL_CONTROL_API_ADDRESS
    assert resolved.insecure is True
    mock_start.assert_called_once_with(False)


def test_in_memory_magic_address_starts_local_superlink_in_memory() -> None:
    """The in-memory magic address starts the managed local SuperLink in memory."""
    connection = SuperLinkConnection(
        name="local",
        address=LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE_IN_MEMORY,
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch(_IS_STARTED_PATH, return_value=False):
        with patch(_START_PATH) as mock_start:
            resolved = ensure_local_superlink(connection)

    assert resolved.address == LOCAL_CONTROL_API_ADDRESS
    assert resolved.insecure is True
    mock_start.assert_called_once_with(True)


def test_local_config_is_preserved_when_endpoint_available() -> None:
    """Keep explicit-address connection unchanged."""
    connection = SuperLinkConnection(
        name="local-dev",
        address="127.0.0.1:9093",
        insecure=False,
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch(_IS_STARTED_PATH) as mock_is_started:
        with patch(_START_PATH) as mock_start:
            resolved = ensure_local_superlink(connection)

    assert resolved is connection
    mock_is_started.assert_not_called()
    mock_start.assert_not_called()


def test_options_only_connection_warns_and_uses_local_magic_address() -> None:
    """Options-only local simulation config is deprecated and mapped to `:local:`."""
    connection = SuperLinkConnection(
        name="local",
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch(_IS_STARTED_PATH, return_value=True) as mock_is_started:
        with patch(_START_PATH) as mock_start:
            resolved = ensure_local_superlink(connection)

    assert resolved.address == LOCAL_CONTROL_API_ADDRESS
    assert resolved.insecure is True
    assert resolved.root_certificates is None
    mock_is_started.assert_called_once()
    mock_start.assert_not_called()


def test_start_local_superlink_uses_builtin_log_rotation(tmp_path: Path) -> None:
    """Start command should include built-in SuperLink log rotation flags."""
    # Prepare
    database = tmp_path / "flwr-local-superlink-state.db"
    log_file = tmp_path / "flwr-local-superlink.log"
    process = MagicMock()
    process.poll.return_value = None

    # Execute
    with (
        patch(
            "flwr.cli.local_superlink._get_local_superlink_paths",
            return_value=(database, log_file),
        ),
        patch(_IS_STARTED_PATH, return_value=True),
        patch(
            "flwr.cli.local_superlink.subprocess.Popen", return_value=process
        ) as popen,
    ):
        _start_local_superlink()

    # Assert
    cmd = popen.call_args.args[0]
    assert "--log-file" in cmd
    assert str(log_file) in cmd
    assert "--serverappio-api-address" in cmd
    assert "127.0.0.1:0" in cmd
    assert "--database" in cmd
    assert str(database) in cmd
    assert "--log-rotation-interval-hours" in cmd
    assert "24" in cmd
    assert "--log-rotation-backup-count" in cmd
    assert "7" in cmd
    assert "--storage-dir" not in cmd
    assert popen.call_args.kwargs["stdout"] is subprocess.DEVNULL
    assert popen.call_args.kwargs["stderr"] is subprocess.DEVNULL
    assert popen.call_args.kwargs["env"][FLWR_DISABLE_UPDATE_CHECK] == "1"


def test_start_local_superlink_in_memory_skips_database_flag(tmp_path: Path) -> None:
    """In-memory local SuperLink startup should not configure a database path."""
    database = tmp_path / "flwr-local-superlink-state.db"
    log_file = tmp_path / "flwr-local-superlink.log"
    process = MagicMock()
    process.poll.return_value = None

    with (
        patch(
            "flwr.cli.local_superlink._get_local_superlink_paths",
            return_value=(database, log_file),
        ),
        patch(_IS_STARTED_PATH, return_value=True),
        patch(
            "flwr.cli.local_superlink.subprocess.Popen", return_value=process
        ) as popen,
    ):
        _start_local_superlink(in_memory=True)

    cmd = popen.call_args.args[0]
    assert "--database" not in cmd
