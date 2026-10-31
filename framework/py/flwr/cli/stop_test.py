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

import click
import pytest

import flwr.cli.stop as stop_module
from flwr.cli.stop import _list_run_ids, _stop_all_runs
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



def test_list_run_ids_all_deduplicates_runs_across_pages() -> None:
    """Runs repeated across offset-paginated pages should be returned once."""
    stub = Mock()
    first_page = {
        run_id: _run(run_id, Status.RUNNING, f"2026-01-{run_id:02d}T00:00:00+00:00")
        for run_id in range(20, 0, -1)
    }
    second_page = {
        19: _run(19, Status.RUNNING, "2026-01-19T00:00:00+00:00"),
        21: _run(21, Status.PENDING, "2026-01-21T00:00:00+00:00"),
    }
    stub.ListRuns.side_effect = [
        ListRunsResponse(run_dict=first_page),
        ListRunsResponse(run_dict=second_page),
    ]

    run_ids = _list_run_ids(stub, active_only=True)

    assert run_ids.count(19) == 1
    assert run_ids == list(range(20, 0, -1)) + [21]


def test_stop_all_runs_continues_after_a_run_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure for one run must not prevent attempts to stop later runs."""
    stub = Mock()
    attempted: list[int] = []

    def stop_run(*, stub: Mock, run_id: int, is_json: bool) -> None:
        attempted.append(run_id)
        if run_id == 2:
            raise click.ClickException("Run 2 already finished.")

    monkeypatch.setattr(stop_module, "_stop_run", stop_run)

    failures = _stop_all_runs(stub, [1, 2, 3])

    assert attempted == [1, 2, 3]
    assert failures == ["run 2: Run 2 already finished."]
