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
"""Automation scheduler worker."""


from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from logging import ERROR, INFO, WARNING

from flwr.common.logger import log
from flwr.proto.control_pb2 import Automation  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.constant import AutomationStatus
from flwr.supercore.date import now
from flwr.supercore.error import EntitlementError

DEFAULT_AUTOMATION_SCHEDULER_BATCH_LIMIT = 100
DEFAULT_AUTOMATION_SCHEDULER_POLL_INTERVAL = 1.0


def process_due_automations(
    state: LinkState,
    *,
    current_time: datetime | None = None,
    limit: int = DEFAULT_AUTOMATION_SCHEDULER_BATCH_LIMIT,
) -> int:
    """Dispatch due automations and return the number of created runs."""
    timestamp = current_time or now()
    due_automations = state.list_automations(
        statuses=[AutomationStatus.ACTIVE],
        due_before=timestamp,
        order_by="next_run_at",
        limit=limit,
    )

    dispatched = 0
    for automation in due_automations:
        try:
            next_run_at = _calculate_next_run_at(automation)
        except ValueError as exc:
            log(
                ERROR,
                "Failing automation %d: %s",
                automation.automation_id,
                exc,
            )
            state.finish_automation(
                automation.automation_id,
                status=AutomationStatus.FAILED,
            )
            continue

        try:
            run_id = state.dispatch_automation(
                automation.automation_id,
                previous_next_run_at=automation.next_run_at,
                next_run_at=next_run_at,
            )
        except EntitlementError as exc:
            log(
                ERROR,
                "Failing automation %d: %s",
                automation.automation_id,
                exc,
            )
            state.finish_automation(
                automation.automation_id,
                status=AutomationStatus.FAILED,
            )
            continue
        if run_id is None:
            continue

        dispatched += 1
        if next_run_at is None and not state.finish_automation(
            automation.automation_id,
            status=AutomationStatus.COMPLETED,
        ):
            log(
                WARNING,
                "Automation %d was dispatched but could not be completed.",
                automation.automation_id,
            )

    return dispatched


def run_automation_scheduler_worker(
    state_factory: LinkStateFactory,
    stop_event: threading.Event,
    *,
    poll_interval: float = DEFAULT_AUTOMATION_SCHEDULER_POLL_INTERVAL,
    batch_limit: int = DEFAULT_AUTOMATION_SCHEDULER_BATCH_LIMIT,
) -> None:
    """Poll for due automations until `stop_event` is set."""
    log(INFO, "Automation scheduler worker started")
    while not stop_event.is_set():
        try:
            process_due_automations(state_factory.state(), limit=batch_limit)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log(ERROR, "Automation scheduler worker failed: %s", exc)

        stop_event.wait(poll_interval)
    log(INFO, "Automation scheduler worker stopped")


def _calculate_next_run_at(automation: Automation) -> str | None:
    """Return the next scheduled run time, or `None` for the final occurrence."""
    if automation.HasField("remaining_runs") and automation.remaining_runs <= 1:
        return None

    if not automation.HasField("fixed_interval") or automation.fixed_interval <= 0:
        raise ValueError("recurring automation requires a positive fixed interval")

    previous_next_run_at = _parse_timestamp(automation.next_run_at)
    return (
        previous_next_run_at + timedelta(seconds=automation.fixed_interval)
    ).isoformat()


def _parse_timestamp(value: str) -> datetime:
    """Parse an automation timestamp string."""
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp
