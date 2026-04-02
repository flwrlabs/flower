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
"""Flower command line interface utils."""


from dataclasses import dataclass
from datetime import datetime, timedelta

from flwr.common.typing import Run
from flwr.supercore.date import isoformat8601_utc


@dataclass
class RunRow:  # pylint: disable=too-many-instance-attributes
    """Represents a single run's data for display.

    Attributes
    ----------
    run_id : int
        The unique identifier for the run.
    federation : str
        The federation name.
    fab_id : str
        The Flower App Bundle identifier.
    fab_version : str
        The FAB version string.
    fab_hash : str
        The SHA-256 hash of the FAB.
    status_text : str
        The formatted status text.
    elapsed : float
        The elapsed time in seconds.
    pending_at : str
        Timestamp when run entered pending state.
    starting_at : str
        Timestamp when run entered starting state.
    running_at : str
        Timestamp when run entered running state.
    finished_at : str
        Timestamp when run finished.
    network_traffic_inbound : int
        The total inbound network traffic (in bytes) used during the run.
        It includes the traffic from SuperNodes to SuperLink.
    network_traffic_outbound : int
        The total outbound network traffic (in bytes) used during the run.
        It includes the traffic from SuperLink to SuperNodes.
    compute_time_serverapp : float
        The total compute time (in seconds) of the ServerApp during the run.
    compute_time_clientapp : float
        The total compute time (in seconds) of all ClientApps during the run.
    """

    run_id: int
    federation: str
    fab_id: str
    fab_version: str
    fab_hash: str
    status_text: str
    elapsed: float
    pending_at: str
    starting_at: str
    running_at: str
    finished_at: str
    network_traffic_inbound: int
    network_traffic_outbound: int
    compute_time_serverapp: float
    compute_time_clientapp: float


def format_runs(runs: list[Run], now_isoformat: str) -> list[RunRow]:
    """Format runs to a list of RunRow objects.

    Parameters
    ----------
    runs : list[Run]
        List of Run objects to format.
    now_isoformat : str
        Current timestamp in ISO format for calculating elapsed time.

    Returns
    -------
    list[RunRow]
        List of formatted RunRow objects sorted by pending_at timestamp.
    """

    def _format_datetime(dt: datetime | None) -> str:
        return isoformat8601_utc(dt).replace("T", " ") if dt else "N/A"

    run_list: list[RunRow] = []

    # Add rows
    for run in sorted(runs, key=lambda r: r.pending_at, reverse=True):
        # Combine status and sub-status into a single string
        if run.status.sub_status == "":
            status_text = run.status.status
        else:
            status_text = f"{run.status.status}:{run.status.sub_status}"

        # Convert isoformat to datetime
        pending_at = datetime.fromisoformat(run.pending_at) if run.pending_at else None
        starting_at = (
            datetime.fromisoformat(run.starting_at) if run.starting_at else None
        )
        running_at = datetime.fromisoformat(run.running_at) if run.running_at else None
        finished_at = (
            datetime.fromisoformat(run.finished_at) if run.finished_at else None
        )

        # Calculate elapsed time
        elapsed_time = timedelta()
        if running_at:
            if finished_at:
                end_time = finished_at
            else:
                end_time = datetime.fromisoformat(now_isoformat)
            elapsed_time = end_time - running_at

        row = RunRow(
            run_id=run.run_id,
            federation=run.federation,
            fab_id=run.fab_id,
            fab_version=run.fab_version,
            fab_hash=run.fab_hash,
            status_text=status_text,
            elapsed=elapsed_time.total_seconds(),
            pending_at=_format_datetime(pending_at),
            starting_at=_format_datetime(starting_at),
            running_at=_format_datetime(running_at),
            finished_at=_format_datetime(finished_at),
            network_traffic_inbound=run.bytes_recv,
            network_traffic_outbound=run.bytes_sent,
            compute_time_serverapp=elapsed_time.total_seconds(),
            compute_time_clientapp=run.clientapp_runtime,
        )
        run_list.append(row)
    return run_list
