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
"""Tests for the `ls` command."""


from unittest.mock import Mock, call

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
)
from flwr.proto.run_pb2 import Run as ProtoRun  # pylint: disable=E0611
from flwr.supercore.control import ControlHttpClient

from .ls import _list_runs

_NOW = "2026-01-01T00:00:00+00:00"


def test_list_runs_fetches_all_pages_without_limit() -> None:
    """The CLI's default listing includes runs beyond the HTTP first page."""
    stub = Mock(spec=ControlHttpClient)
    stub.ListRuns.side_effect = [
        ListRunsResponse(
            run_dict={run_id: ProtoRun(run_id=run_id) for run_id in range(1, 21)},
            now=_NOW,
        ),
        ListRunsResponse(run_dict={21: ProtoRun(run_id=21)}, now=_NOW),
    ]

    rows = _list_runs(stub)

    assert {row.run_id for row in rows} == set(range(1, 22))
    assert stub.ListRuns.call_args_list == [
        call(ListRunsRequest(skip=0)),
        call(ListRunsRequest(skip=20)),
    ]


def test_list_runs_accepts_unbounded_legacy_response() -> None:
    """Default listing stays complete with a server that ignores `skip`."""
    stub = Mock(spec=ControlHttpClient)
    stub.ListRuns.return_value = ListRunsResponse(
        run_dict={run_id: ProtoRun(run_id=run_id) for run_id in range(1, 22)},
        now=_NOW,
    )

    rows = _list_runs(stub)

    assert {row.run_id for row in rows} == set(range(1, 22))
    stub.ListRuns.assert_called_once_with(ListRunsRequest(skip=0))


def test_list_runs_stops_when_legacy_server_repeats_page() -> None:
    """An older server with exactly 20 runs cannot loop forever."""
    stub = Mock(spec=ControlHttpClient)
    stub.ListRuns.return_value = ListRunsResponse(
        run_dict={run_id: ProtoRun(run_id=run_id) for run_id in range(1, 21)},
        now=_NOW,
    )

    rows = _list_runs(stub)

    assert {row.run_id for row in rows} == set(range(1, 21))
    assert stub.ListRuns.call_args_list == [
        call(ListRunsRequest(skip=0)),
        call(ListRunsRequest(skip=20)),
        call(ListRunsRequest(limit=21)),
    ]


def test_list_runs_continues_after_overlapping_page() -> None:
    """New runs shifting an offset page do not hide older runs."""
    stub = Mock(spec=ControlHttpClient)
    stub.ListRuns.side_effect = [
        ListRunsResponse(
            run_dict={run_id: ProtoRun(run_id=run_id) for run_id in range(21, 41)},
            now=_NOW,
        ),
        ListRunsResponse(
            run_dict={run_id: ProtoRun(run_id=run_id) for run_id in range(21, 41)},
            now=_NOW,
        ),
        ListRunsResponse(
            run_dict={run_id: ProtoRun(run_id=run_id) for run_id in range(40, 61)},
            now=_NOW,
        ),
        ListRunsResponse(run_dict={1: ProtoRun(run_id=1)}, now=_NOW),
    ]

    rows = _list_runs(stub)

    assert {row.run_id for row in rows} == set(range(21, 61)) | {1}
    assert stub.ListRuns.call_args_list == [
        call(ListRunsRequest(skip=0)),
        call(ListRunsRequest(skip=20)),
        call(ListRunsRequest(limit=21)),
        call(ListRunsRequest(skip=40)),
    ]


def test_list_runs_with_limit_fetches_one_page() -> None:
    """An explicit CLI limit requests only the selected number of runs."""
    stub = Mock(spec=ControlHttpClient)
    stub.ListRuns.return_value = ListRunsResponse(
        run_dict={1: ProtoRun(run_id=1)}, now=_NOW
    )

    rows = _list_runs(stub, limit=1)

    assert [row.run_id for row in rows] == [1]
    stub.ListRuns.assert_called_once_with(ListRunsRequest(limit=1))
