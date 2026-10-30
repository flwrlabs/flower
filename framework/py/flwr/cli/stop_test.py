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
"""Tests for the Flower CLI stop command."""


from unittest.mock import Mock

from flwr.cli.stop import _list_run_ids
from flwr.common.constant import Status
from flwr.proto.control_pb2 import ListRunsResponse
from flwr.proto.run_pb2 import Run, RunStatus


def _run(run_id: int, status: str, pending_at: str) -> Run:
    return Run(
        run_id=run_id,
        status=RunStatus(status=status),
        pending_at=pending_at,
    )


def test_list_run_ids_latest() -> None:
    """The latest option should select the newest run."""
    stub = Mock()
    stub.ListRuns.return_value = ListRunsResponse(
        run_dict={
            1: _run(1, Status.RUNNING, "2026-01-01T00:00:00+00:00"),
            2: _run(2, Status.FINISHED, "2026-01-02T00:00:00+00:00"),
        }
    )

    assert _list_run_ids(stub, latest_only=True) == [2]
    request = stub.ListRuns.call_args.kwargs["request"]
    assert request.limit == 20
    assert request.skip == 0


def test_list_run_ids_all_paginates_and_filters_active() -> None:
    """The all option should include active runs across every page."""
    stub = Mock()
    first_page = {
        run_id: _run(
            run_id,
            Status.RUNNING if run_id % 2 else Status.FINISHED,
            f"2026-01-{(run_id % 28) + 1:02d}T00:00:00+00:00",
        )
        for run_id in range(20, 0, -1)
    }
    second_page = {
        run_id: _run(run_id, Status.PENDING, f"2025-12-{run_id:02d}T00:00:00+00:00")
        for run_id in range(21, 23)
    }
    stub.ListRuns.side_effect = [
        ListRunsResponse(run_dict=first_page),
        ListRunsResponse(run_dict=second_page),
    ]

    run_ids = _list_run_ids(stub, active_only=True)

    assert set(run_ids) == set(range(1, 22, 2)) | {21, 22}
    assert stub.ListRuns.call_count == 2
    assert stub.ListRuns.call_args_list[1].kwargs["request"].skip == 20
