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
"""Automation creation from an authenticated run template."""

from flwr.common.serde import fab_to_proto, user_config_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    StartAutomationFromTaskRequest,
    StartAutomationFromTaskResponse,
)
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartRunRequest,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.error import FlowerError
from flwr.supercore.run import Run
from flwr.superlink.servicer.control.control_handlers import start_automation


def derive_start_run_request(
    state: LinkState,
    run: Run,
    automation_task: str,
) -> StartRunRequest:
    """Derive a start run request from an authoritative run template."""
    override_config = dict(run.override_config)
    override_config["agent.input"] = automation_task
    fab = state.get_fab(run.fab_hash)
    federation_config = state.get_federation_config(run.run_id)
    return StartRunRequest(
        fab=fab_to_proto(fab) if fab is not None else None,
        override_config=user_config_to_proto(override_config),
        override_federation_config=federation_config,
        federation=run.federation_id,
        series_id=run.series_id,
        connector_refs=state.get_run_connector_refs(run_id=run.run_id),
    )


def start_automation_from_run(
    state: LinkState,
    run: Run,
    request: StartAutomationFromTaskRequest,
) -> StartAutomationFromTaskResponse:
    """Store an automation derived from an authoritative run template."""
    automation_task = request.task.strip()
    if not automation_task:
        raise ValueError("`task` must be a non-empty string.")

    control_request = StartAutomationRequest(
        start_run_request=derive_start_run_request(state, run, automation_task),
    )
    if request.HasField("start_at"):
        control_request.start_at = request.start_at
    if request.HasField("fixed_interval"):
        control_request.fixed_interval = request.fixed_interval
    if request.HasField("max_runs"):
        control_request.max_runs = request.max_runs

    try:
        response = start_automation(
            control_request,
            AccountInfo(flwr_aid=run.flwr_aid, account_name=""),
            state,
        )
    except FlowerError as exc:
        raise ValueError(exc.public_details or exc.message) from exc

    return StartAutomationFromTaskResponse(
        automation_id=response.automation_id,
        series_id=response.series_id,
        next_run_at=response.next_run_at,
    )
