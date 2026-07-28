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

from datetime import UTC, datetime

from flwr.common.serde import fab_to_proto, user_config_to_proto
from flwr.proto.appio_pb2 import (  # pylint: disable=E0611
    StartAutomationFromTaskRequest,
    StartAutomationFromTaskResponse,
)
from flwr.proto.control_pb2 import StartRunRequest  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.date import isoformat8601_utc, now
from flwr.supercore.run import Run


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

    next_run_at = now()
    if request.HasField("start_at"):
        try:
            next_run_at = datetime.fromisoformat(request.start_at)
        except ValueError as exc:
            raise ValueError("`start_at` must be an RFC 3339 timestamp.") from exc
        if next_run_at.tzinfo is None:
            raise ValueError("`start_at` must include a timezone.")
        next_run_at = next_run_at.astimezone(UTC)
    fixed_interval = (
        request.fixed_interval if request.HasField("fixed_interval") else None
    )
    max_runs = request.max_runs if request.HasField("max_runs") else None
    if fixed_interval == 0:
        raise ValueError("`fixed_interval` must be greater than zero.")
    if max_runs == 0:
        raise ValueError("`max_runs` must be greater than zero.")
    if fixed_interval is None and max_runs is not None:
        raise ValueError("`max_runs` requires `fixed_interval`.")

    automation = state.store_automation(
        federation_id=run.federation_id,
        flwr_aid=run.flwr_aid,
        start_run_request=derive_start_run_request(state, run, automation_task),
        series_id=run.series_id,
        next_run_at=isoformat8601_utc(next_run_at),
        fixed_interval=fixed_interval,
        max_runs=max_runs if fixed_interval is not None else 1,
    )
    return StartAutomationFromTaskResponse(
        automation_id=automation.automation_id,
        series_id=automation.series_id,
        next_run_at=automation.next_run_at,
    )
