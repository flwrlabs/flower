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
"""Tests for automation creation from a run template."""

from typing import cast
from unittest.mock import Mock, patch

from flwr.common.serde import user_config_from_proto
from flwr.proto.appio_pb2 import StartAutomationFromTaskRequest  # pylint: disable=E0611
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationResponse,
    StartRunRequest,
)
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.fab import Fab
from flwr.supercore.run import Run

from .automation import derive_start_run_request, start_automation_from_run


def test_derive_start_run_request() -> None:
    """Derive a complete start request from the source run."""
    state_mock = Mock(spec=LinkState)
    fab = Fab("fab-hash", b"fab-content", {"signature": "verified"})
    federation_config = SimulationConfig(num_supernodes=2)
    state_mock.get_fab.return_value = fab
    state_mock.get_federation_config.return_value = federation_config
    state_mock.get_run_connector_refs.return_value = ["calendar", "email"]
    run = Run.create_empty(run_id=123)
    run.fab_hash = fab.hash_str
    run.override_config = {"learning-rate": 0.1}
    run.federation_id = "federation-a"
    run.series_id = 456

    request = derive_start_run_request(
        cast(LinkState, state_mock), run, "Train a model"
    )

    assert request.fab.hash_str == fab.hash_str
    assert request.fab.content == fab.content
    assert dict(request.fab.verifications) == fab.verifications
    assert request.override_federation_config == federation_config
    assert request.federation == run.federation_id
    assert request.series_id == run.series_id
    assert list(request.connector_refs) == ["calendar", "email"]
    assert user_config_from_proto(request.override_config) == {
        "learning-rate": 0.1,
        "agent.input": "Train a model",
    }


@patch("flwr.superlink.servicer.automation.start_automation")
def test_start_automation_from_run_delegates_to_control(
    start_automation_mock: Mock,
) -> None:
    """Delegate the derived request to the Control implementation."""
    state_mock = Mock(spec=LinkState)
    state_mock.get_fab.return_value = None
    state_mock.get_federation_config.return_value = None
    state_mock.get_run_connector_refs.return_value = []
    start_automation_mock.return_value = StartAutomationResponse(
        automation_id=1,
        series_id=456,
        next_run_at="2026-07-28T12:00:00Z",
    )
    run = Run.create_empty(run_id=123)
    run.flwr_aid = "account-a"
    run.federation_id = "federation-a"
    run.series_id = 456

    response = start_automation_from_run(
        cast(LinkState, state_mock),
        run,
        StartAutomationFromTaskRequest(
            task="Train a model",
            start_at="2026-07-28T12:00:00Z",
            fixed_interval=60,
            max_runs=3,
        ),
    )

    assert response.automation_id == 1
    control_request = cast(
        StartRunRequest,
        start_automation_mock.call_args.args[0].start_run_request,
    )
    assert control_request.federation == run.federation_id
    assert control_request.series_id == run.series_id
    assert user_config_from_proto(control_request.override_config)["agent.input"] == (
        "Train a model"
    )
    assert start_automation_mock.call_args.args[0].start_at == ("2026-07-28T12:00:00Z")
    assert start_automation_mock.call_args.args[0].fixed_interval == 60
    assert start_automation_mock.call_args.args[0].max_runs == 3
    assert start_automation_mock.call_args.args[1].flwr_aid == run.flwr_aid
    assert start_automation_mock.call_args.args[2] is state_mock
