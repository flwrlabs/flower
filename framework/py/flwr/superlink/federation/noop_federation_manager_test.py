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
"""Tests for NoOpFederationManager."""


from unittest.mock import Mock

import pytest
from parameterized import parameterized

from flwr.common.constant import NOOP_ACCOUNT_NAME, NOOP_FLWR_AID
from flwr.common.typing import Federation, Run, RunStatus
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.federation_pb2 import Account, Member  # pylint: disable=E0611
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.supercore.constant import (
    DEFAULT_SIMULATION_CONFIG,
    NOOP_FEDERATION,
    NOOP_FEDERATION_DESCRIPTION,
    ActionType,
)
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.typing import ActionContext

from .noop_federation_manager import NoOpFederationManager


def test_get_details_with_valid_federation() -> None:
    """Test get_details returns correct Federation details."""
    # Prepare
    manager = NoOpFederationManager(simulation=True)
    mock_linkstate = Mock()
    manager.linkstate = mock_linkstate
    config = SimulationConfig(
        num_supernodes=12,
        client_resources_num_cpus=3,
        client_resources_num_gpus=1.0,
        backend="ray",
        verbose=True,
    )
    expected_config = SimulationConfig(
        num_supernodes=12,
        client_resources_num_cpus=3,
        client_resources_num_gpus=1.0,
        backend="ray",
        verbose=True,
        init_args_logging_level=DEFAULT_SIMULATION_CONFIG.init_args_logging_level,
        init_args_log_to_driver=DEFAULT_SIMULATION_CONFIG.init_args_log_to_driver,
    )
    manager.set_simulation_config(NOOP_FLWR_AID, NOOP_FEDERATION, config)

    # Mock data
    run_id_1 = 123
    run_id_2 = 456
    mock_run_1 = Run(
        run_id=run_id_1,
        fab_id="test_fab_1",
        fab_version="1.0.0",
        fab_hash="hash123",
        override_config={},
        pending_at="2025-01-01T00:00:00",
        starting_at="2025-01-01T00:01:00",
        running_at="2025-01-01T00:02:00",
        finished_at="",
        status=RunStatus(status="running", sub_status="", details=""),
        flwr_aid=NOOP_FLWR_AID,
        federation=NOOP_FEDERATION,
        bytes_sent=1024,
        bytes_recv=512,
        clientapp_runtime=1.1,
    )
    mock_run_2 = Run(
        run_id=run_id_2,
        fab_id="test_fab_2",
        fab_version="2.0.0",
        fab_hash="hash456",
        override_config={},
        pending_at="2025-01-02T00:00:00",
        starting_at="2025-01-02T00:01:00",
        running_at="2025-01-02T00:02:00",
        finished_at="2025-01-02T00:10:00",
        status=RunStatus(status="finished", sub_status="", details=""),
        flwr_aid=NOOP_FLWR_AID,
        federation=NOOP_FEDERATION,
        bytes_sent=2048,
        bytes_recv=1024,
        clientapp_runtime=1.2,
    )
    mock_node_1 = NodeInfo(
        node_id=1,
        owner_aid=NOOP_FLWR_AID,
        owner_name="test_owner_1",
        status="registered",
        registered_at="2025-01-01T00:00:00",
        heartbeat_interval=30.0,
        public_key=b"public_key_1",
    )
    mock_node_2 = NodeInfo(
        node_id=2,
        owner_aid=NOOP_FLWR_AID,
        owner_name="test_owner_2",
        status="registered",
        registered_at="2025-01-02T00:00:00",
        heartbeat_interval=30.0,
        public_key=b"public_key_2",
    )

    # Configure mocks
    mock_linkstate.get_run_info.return_value = [mock_run_1, mock_run_2]
    mock_linkstate.get_node_info.return_value = [mock_node_1, mock_node_2]

    # Execute
    result = manager.get_details(NOOP_FEDERATION)

    # Assert
    assert isinstance(result, Federation)
    assert result.name == NOOP_FEDERATION
    assert result.description == NOOP_FEDERATION_DESCRIPTION
    assert len(result.members) == 1
    assert result.members[0] == Member(
        account=Account(id=NOOP_FLWR_AID, name=NOOP_ACCOUNT_NAME),
        role="owner",
    )
    assert len(result.nodes) == 2
    assert mock_node_1 in result.nodes and mock_node_2 in result.nodes
    assert len(result.runs) == 2
    assert mock_run_1 in result.runs and mock_run_2 in result.runs
    assert result.archived is False
    assert result.simulation is True
    assert result.config == expected_config


def test_get_details_with_invalid_federation() -> None:
    """Test get_details raises ValueError for invalid federation."""
    # Prepare
    manager = NoOpFederationManager()
    mock_linkstate = Mock()
    manager.linkstate = mock_linkstate
    invalid_federation = "invalid_federation"

    # Execute & Assert
    with pytest.raises(ValueError):
        manager.get_details(invalid_federation)


def test_get_details_with_no_runs() -> None:
    """Test get_details returns empty runs list when no runs exist."""
    # Prepare
    manager = NoOpFederationManager()
    mock_linkstate = Mock()
    manager.linkstate = mock_linkstate

    # Configure mocks for empty runs
    mock_linkstate.get_run_info.return_value = []
    mock_linkstate.get_node_info.return_value = []

    # Execute
    result = manager.get_details(NOOP_FEDERATION)

    # Assert
    assert result.name == NOOP_FEDERATION
    assert len(result.members) == 1
    assert result.members[0] == Member(
        account=Account(id=NOOP_FLWR_AID, name=NOOP_ACCOUNT_NAME),
        role="owner",
    )
    assert len(result.nodes) == 0
    assert len(result.runs) == 0
    assert result.archived is False
    assert result.simulation is False


def test_exists() -> None:
    """Test exists method returns True only for NOOP_FEDERATION."""
    # Prepare
    manager = NoOpFederationManager()

    # Execute & Assert
    assert manager.exists(NOOP_FEDERATION)
    assert not manager.exists("other_federation")


def test_has_member() -> None:
    """Test has_member method returns True only for NOOP_FLWR_AID."""
    # Prepare
    manager = NoOpFederationManager()

    # Execute & Assert
    assert manager.has_member(NOOP_FLWR_AID, NOOP_FEDERATION) is True
    assert manager.has_member("any_aid", NOOP_FEDERATION) is False

    # Test that it raises ValueError for non-existent federation
    with pytest.raises(ValueError):
        manager.has_member("any_aid", "other_federation")


def test_filter_nodes() -> None:
    """Test filter_nodes method returns all provided node IDs."""
    # Prepare
    manager = NoOpFederationManager()
    node_ids = {1, 2, 3, 4, 5}

    # Execute
    result = manager.filter_nodes(node_ids, NOOP_FEDERATION)

    # Assert
    assert result == node_ids

    # Test that it raises ValueError for non-existent federation
    with pytest.raises(ValueError):
        manager.filter_nodes(node_ids, "other_federation")


def test_has_node() -> None:
    """Test has_node method returns True for NOOP_FEDERATION."""
    # Prepare
    manager = NoOpFederationManager()

    # Execute & Assert
    assert manager.has_node(1, NOOP_FEDERATION) is True
    assert manager.has_node(999, NOOP_FEDERATION) is True

    # Test that it raises ValueError for non-existent federation
    with pytest.raises(ValueError):
        manager.has_node(999, "any_federation")


@parameterized.expand(
    [
        (ActionType.START_RUN),
        (ActionType.CREATE_FEDERATION),
        (ActionType.CREATE_INVITATION),
    ]
)  # type: ignore
def test_can_execute(action: ActionType) -> None:
    """Test can_execute method returns True for NOOP_FEDERATION."""
    manager = NoOpFederationManager()

    allowed = manager.can_execute(NOOP_FLWR_AID, action, ActionContext())
    assert allowed is True


def test_get_federations() -> None:
    """Test get_federations method returns NOOP_FEDERATION."""
    # Prepare
    manager = NoOpFederationManager()

    # Execute
    result = manager.get_federations("any_aid")
    result2 = manager.get_federations(NOOP_FLWR_AID)

    # Assert
    assert len(result) == 0
    assert len(result2) == 1
    assert result2[0].name == NOOP_FEDERATION
    assert result2[0].description == NOOP_FEDERATION_DESCRIPTION
    assert result2[0].archived is False
    assert result2[0].simulation is False


def test_simulation_runtime_flag_is_reflected() -> None:
    """Test simulation flag is reflected by NoOpFederationManager responses."""
    manager = NoOpFederationManager(simulation=True)
    mock_linkstate = Mock()
    mock_linkstate.get_run_info.return_value = []
    mock_linkstate.get_node_info.return_value = []
    manager.linkstate = mock_linkstate

    federations = manager.get_federations(NOOP_FLWR_AID)
    details = manager.get_details(NOOP_FEDERATION)

    assert federations[0].simulation is True
    assert details.simulation is True
    assert federations[0].config == DEFAULT_SIMULATION_CONFIG
    assert details.config == DEFAULT_SIMULATION_CONFIG


def test_get_simulation_config_returns_defaults_when_unset() -> None:
    """Test get_simulation_config returns shared defaults when unset."""
    manager = NoOpFederationManager(simulation=True)

    stored = manager.get_simulation_config(NOOP_FEDERATION)

    assert stored == DEFAULT_SIMULATION_CONFIG
    assert stored.num_supernodes == DEFAULT_SIMULATION_CONFIG.num_supernodes
    assert (
        stored.client_resources_num_cpus
        == DEFAULT_SIMULATION_CONFIG.client_resources_num_cpus
    )
    assert (
        stored.client_resources_num_gpus
        == DEFAULT_SIMULATION_CONFIG.client_resources_num_gpus
    )
    assert stored.backend == DEFAULT_SIMULATION_CONFIG.backend
    assert stored.verbose is DEFAULT_SIMULATION_CONFIG.verbose
    assert (
        stored.init_args_log_to_driver
        is DEFAULT_SIMULATION_CONFIG.init_args_log_to_driver
    )


def test_simulation_config_returns_none_when_simulation_is_disabled() -> None:
    """Test simulation config reads return None outside simulation mode."""
    manager = NoOpFederationManager()

    assert manager._simulation_config is None  # pylint: disable=protected-access

    assert manager.get_simulation_config(NOOP_FEDERATION) is None

    with pytest.raises(FlowerError) as set_err:
        manager.set_simulation_config(
            NOOP_FLWR_AID, NOOP_FEDERATION, SimulationConfig()
        )
    assert set_err.value.code == ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PERMISSION


def test_get_simulation_config_fails_for_invalid_federation() -> None:
    """Test get_simulation_config fails for invalid federation names."""
    manager = NoOpFederationManager(simulation=True)

    with pytest.raises(FlowerError) as err:
        manager.get_simulation_config("invalid_federation")

    assert err.value.code == ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PERMISSION


def test_get_federations_returns_stored_simulation_config() -> None:
    """Test get_federations returns the stored simulation config."""
    manager = NoOpFederationManager(simulation=True)
    config = SimulationConfig(
        num_supernodes=12,
        client_resources_num_cpus=3,
        client_resources_num_gpus=1.0,
        backend="ray",
        verbose=True,
    )
    expected_config = SimulationConfig(
        num_supernodes=12,
        client_resources_num_cpus=3,
        client_resources_num_gpus=1.0,
        backend="ray",
        verbose=True,
        init_args_logging_level=DEFAULT_SIMULATION_CONFIG.init_args_logging_level,
        init_args_log_to_driver=DEFAULT_SIMULATION_CONFIG.init_args_log_to_driver,
    )

    manager.set_simulation_config(NOOP_FLWR_AID, NOOP_FEDERATION, config)

    federations = manager.get_federations(NOOP_FLWR_AID)

    assert federations[0].config == expected_config


def test_set_simulation_config_does_not_override_unset_fields() -> None:
    """Test that partial updates don't clear fields not included in the update.

    Setting only `num_supernodes` must not reset other fields (e.g.
    `client_resources_num_cpus`, `backend`) back to their unset/default values.
    """
    manager = NoOpFederationManager(simulation=True)

    # Establish a fully-specified baseline
    full_config = SimulationConfig(
        num_supernodes=10,
        client_resources_num_cpus=4,
        client_resources_num_gpus=1.0,
        backend="ray",
        verbose=True,
    )
    manager.set_simulation_config(NOOP_FLWR_AID, NOOP_FEDERATION, full_config)

    # Apply a partial update that only changes num_supernodes
    partial_config = SimulationConfig(num_supernodes=99)
    manager.set_simulation_config(NOOP_FLWR_AID, NOOP_FEDERATION, partial_config)

    result = manager.get_simulation_config(NOOP_FEDERATION)
    assert result is not None

    # The updated field should reflect the new value
    assert result.num_supernodes == 99

    # All other fields must be preserved from the previous full_config
    assert result.client_resources_num_cpus == 4
    assert result.client_resources_num_gpus == 1.0
    assert result.backend == "ray"
    assert result.verbose is True
