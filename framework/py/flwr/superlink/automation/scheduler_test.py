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
"""Tests for the automation scheduler worker."""


from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from flwr.proto.control_pb2 import Automation  # pylint: disable=E0611
from flwr.server.superlink.linkstate import InMemoryLinkState, LinkState
from flwr.supercore.constant import (
    ActionType,
    FLWR_IN_MEMORY_DB_NAME,
    AutomationStatus,
    RunTime,
    TaskType,
)
from flwr.supercore.error import EntitlementError
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.typing import StartRunContext
from flwr.superlink.federation import NoOpFederationManager

from .scheduler import process_due_automations


def test_process_due_automations_completes_one_shot_schedule() -> None:
    """A one-shot automation should dispatch once and complete immediately."""
    state = _create_state()
    current = datetime(2026, 1, 1, tzinfo=UTC)
    automation = _store_due_automation(state, current=current, max_runs=1)

    dispatched = process_due_automations(state, current_time=current)

    assert dispatched == 1
    completed = state.list_automations(
        statuses=[AutomationStatus.COMPLETED],
        order_by="updated_at",
    )
    assert [item.automation_id for item in completed] == [automation.automation_id]
    assert len(state.get_run_info(flwr_aids=["aid-a"])) == 1


def test_process_due_automations_advances_recurring_schedule() -> None:
    """A recurring automation should stay active until the final occurrence."""
    state = _create_state()
    current = datetime(2026, 1, 1, tzinfo=UTC)
    automation = _store_due_automation(
        state,
        current=current,
        fixed_interval=60,
        max_runs=2,
    )
    expected_next_run_at = (current + timedelta(seconds=30)).isoformat()

    dispatched = process_due_automations(state, current_time=current)

    assert dispatched == 1
    active = state.list_automations(
        statuses=[AutomationStatus.ACTIVE],
        order_by="updated_at",
    )
    assert [item.automation_id for item in active] == [automation.automation_id]
    assert active[0].remaining_runs == 1
    assert active[0].next_run_at == expected_next_run_at

    dispatched = process_due_automations(
        state,
        current_time=current + timedelta(seconds=31),
    )

    assert dispatched == 1
    completed = state.list_automations(
        statuses=[AutomationStatus.COMPLETED],
        order_by="updated_at",
    )
    assert [item.automation_id for item in completed] == [automation.automation_id]
    assert len(state.get_run_info(flwr_aids=["aid-a"])) == 2


def test_process_due_automations_keeps_unbounded_schedule_active() -> None:
    """An unbounded automation should keep scheduling future occurrences."""
    state = _create_state()
    current = datetime(2026, 1, 1, tzinfo=UTC)
    automation = _store_due_automation(
        state,
        current=current,
        fixed_interval=60,
        max_runs=None,
    )

    dispatched = process_due_automations(state, current_time=current)

    assert dispatched == 1
    active = state.list_automations(
        statuses=[AutomationStatus.ACTIVE],
        order_by="updated_at",
    )
    assert [item.automation_id for item in active] == [automation.automation_id]
    assert not active[0].HasField("remaining_runs")


def test_process_due_automations_fails_invalid_recurring_schedule() -> None:
    """A recurring automation without an interval should fail instead of spinning."""
    state = _create_state()
    current = datetime(2026, 1, 1, tzinfo=UTC)
    automation = _store_due_automation(state, current=current, max_runs=2)

    dispatched = process_due_automations(state, current_time=current)

    assert dispatched == 0
    failed = state.list_automations(
        statuses=[AutomationStatus.FAILED],
        order_by="updated_at",
    )
    assert [item.automation_id for item in failed] == [automation.automation_id]
    assert len(state.get_run_info(flwr_aids=["aid-a"])) == 0


def test_process_due_automations_fails_entitlement_denial() -> None:
    """A denied automation should fail without creating a run."""
    state = _create_state()
    current = datetime(2026, 1, 1, tzinfo=UTC)
    automation = _store_due_automation(state, current=current, max_runs=1)

    with patch.object(
        state.federation_manager,
        "can_execute",
        side_effect=EntitlementError(
            "Start run denied for this account.",
            public_details="Start run not permitted.",
            entitlement_code=101,
        ),
    ) as mock_can_execute:
        dispatched = process_due_automations(state, current_time=current)

    assert dispatched == 0
    mock_can_execute.assert_called_once_with(
        "aid-a",
        ActionType.START_RUN,
        StartRunContext(federation_id="@me/fed-a", runtime=RunTime.DEPLOYMENT),
    )
    failed = state.list_automations(
        statuses=[AutomationStatus.FAILED],
        order_by="updated_at",
    )
    assert [item.automation_id for item in failed] == [automation.automation_id]
    assert len(state.get_run_info(flwr_aids=["aid-a"])) == 0


def _create_state() -> LinkState:
    """Create an in-memory LinkState for scheduler tests."""
    return InMemoryLinkState(
        NoOpFederationManager(),
        ObjectStoreFactory(FLWR_IN_MEMORY_DB_NAME).store(),
    )


def _store_due_automation(
    state: LinkState,
    *,
    current: datetime,
    fixed_interval: int | None = None,
    max_runs: int | None,
) -> Automation:
    """Store a due automation with a valid run series."""
    initial_run_id = state.create_run(
        fab_id="bootstrap-fab",
        fab_version="1.0.0",
        fab_hash="bootstrap-hash",
        override_config={},
        federation_id="@me/fed-a",
        federation_config=None,
        flwr_aid="bootstrap-aid",
        primary_task_type=TaskType.SERVER_APP,
    )
    series_id = state.get_run_info(run_ids=[initial_run_id])[0].series_id
    return state.store_automation(
        federation_id="@me/fed-a",
        flwr_aid="aid-a",
        fab_id="fab-id",
        fab_version="1.0.0",
        fab_hash="fab-hash",
        override_config={},
        federation_config=None,
        primary_task_type=TaskType.SERVER_APP,
        series_id=series_id,
        next_run_at=(current - timedelta(seconds=30)).isoformat(),
        fixed_interval=fixed_interval,
        max_runs=max_runs,
    )
