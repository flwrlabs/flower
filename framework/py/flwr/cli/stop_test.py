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
"""Tests for Flower command line interface `stop` command."""


import json
from unittest.mock import MagicMock, call, patch

import click
import grpc
import pytest
from typer.testing import CliRunner

from flwr.common.constant import CliOutputFormat, Status
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
    StopRunRequest,
    StopRunResponse,
)
from flwr.proto.run_pb2 import Run, RunStatus  # pylint: disable=E0611
from flwr.supercore.error import ApiErrorCode

from .app import app
from .stop import _parse_run_id, _resolve_run_ids, _stop_runs, stop

runner = CliRunner()


class _AlreadyFinishedRpcError(grpc.RpcError):  # type: ignore[misc]
    """Represent an already-finished API error from the Control API."""

    def details(self) -> str:
        """Return the serialized Flower error."""
        return json.dumps(
            {
                "code": ApiErrorCode.RUN_ALREADY_FINISHED,
                "detail": "Run already finished.",
            }
        )


def _run(run_id: int, status: str, pending_at: str) -> Run:
    return Run(
        run_id=run_id,
        status=RunStatus(status=status),
        pending_at=pending_at,
    )


def test_resolve_run_ids_returns_numeric_id_without_listing_runs() -> None:
    """A numeric run ID should not require a ListRuns request."""
    stub = MagicMock()

    assert _resolve_run_ids(stub, 123) == [123]
    stub.ListRuns.assert_not_called()


def test_resolve_run_ids_returns_latest_active_run() -> None:
    """The latest selector should skip newer finished runs."""
    stub = MagicMock()
    stub.ListRuns.return_value = ListRunsResponse(
        run_dict={
            1: _run(1, Status.RUNNING, "2026-08-20T10:00:00+00:00"),
            2: _run(2, Status.FINISHED, "2026-08-20T12:00:00+00:00"),
            3: _run(3, Status.PENDING, "2026-08-20T11:00:00+00:00"),
        }
    )

    assert _resolve_run_ids(stub, "latest") == [3]


def test_resolve_run_ids_returns_all_active_runs_newest_first() -> None:
    """The all selector should return active runs ordered by creation time."""
    stub = MagicMock()
    stub.ListRuns.return_value = ListRunsResponse(
        run_dict={
            1: _run(1, Status.RUNNING, "2026-08-20T10:00:00+00:00"),
            2: _run(2, Status.FINISHED, "2026-08-20T12:00:00+00:00"),
            3: _run(3, Status.STARTING, "2026-08-20T11:00:00+00:00"),
        }
    )

    assert _resolve_run_ids(stub, "all") == [3, 1]


@pytest.mark.parametrize("run_id", ["newest", "-1"])
def test_parse_run_id_rejects_invalid_run_id(run_id: str) -> None:
    """Unknown selectors and negative run IDs should be rejected clearly."""
    with pytest.raises(click.ClickException):
        _parse_run_id(run_id)


def test_stop_command_rejects_invalid_selector_before_setup() -> None:
    """Invalid selectors should fail before migration or connection setup."""
    with (
        patch("flwr.cli.app.warn_if_flwr_update_available"),
        patch("flwr.cli.stop.migrate") as migrate,
        patch("flwr.cli.stop.init_channel_from_connection") as init_channel,
    ):
        result = runner.invoke(app, ["stop", "lates"])

    assert result.exit_code == 1
    assert "RUN_ID must be an integer, 'latest', or 'all'" in result.output
    migrate.assert_not_called()
    init_channel.assert_not_called()


def test_resolve_run_ids_rejects_selector_without_active_runs() -> None:
    """Selectors should fail clearly if all runs have finished."""
    stub = MagicMock()
    stub.ListRuns.return_value = ListRunsResponse(
        run_dict={
            1: _run(1, Status.FINISHED, "2026-08-20T10:00:00+00:00"),
        }
    )

    with pytest.raises(click.ClickException, match="No active runs found"):
        _resolve_run_ids(stub, "all")


def test_stop_all_calls_each_run_and_prints_one_json_response() -> None:
    """The all selector should stop each active run and emit one JSON document."""
    stub = MagicMock()
    stub.ListRuns.return_value = ListRunsResponse(
        run_dict={
            1: _run(1, Status.RUNNING, "2026-08-20T10:00:00+00:00"),
            3: _run(3, Status.STARTING, "2026-08-20T11:00:00+00:00"),
        }
    )
    stub.StopRun.return_value = StopRunResponse(success=True)
    channel = MagicMock()

    with (
        patch("flwr.cli.stop.warn_if_federation_config_overrides"),
        patch("flwr.cli.stop.migrate"),
        patch("flwr.cli.stop.read_superlink_connection"),
        patch("flwr.cli.stop.init_channel_from_connection", return_value=channel),
        patch("flwr.cli.stop.ControlStub", return_value=stub),
        patch("flwr.cli.stop.print_json_to_stdout") as print_json,
    ):
        stop(MagicMock(args=[]), "all", output_format=CliOutputFormat.JSON)

    assert stub.StopRun.call_args_list == [
        call(request=StopRunRequest(run_id=3)),
        call(request=StopRunRequest(run_id=1)),
    ]
    print_json.assert_called_once_with({"success": True, "run-ids": ["3", "1"]})
    channel.close.assert_called_once()


def test_stop_all_continues_after_failure() -> None:
    """A failed stop should not prevent attempts for later runs in the batch."""
    stub = MagicMock()
    with patch(
        "flwr.cli.stop._stop_run",
        side_effect=[click.ClickException("already finished"), None],
    ) as stop_run:
        with pytest.raises(click.ClickException, match="already finished"):
            _stop_runs(stub, [3, 1], is_json=False, selector="all")

    assert stop_run.call_args_list == [
        call(stub=stub, run_id=3, is_json=False, ignore_finished=True),
        call(stub=stub, run_id=1, is_json=False, ignore_finished=True),
    ]


def test_stop_all_ignores_run_that_finished_after_selection() -> None:
    """A concurrently finished run should not make a batch stop fail."""
    stub = MagicMock()
    stub.StopRun.side_effect = [
        _AlreadyFinishedRpcError(),
        StopRunResponse(success=True),
    ]

    _stop_runs(stub, [3, 1], is_json=False, selector="all")

    assert stub.StopRun.call_args_list == [
        call(request=StopRunRequest(run_id=3)),
        call(request=StopRunRequest(run_id=1)),
    ]


def test_stop_latest_ignores_run_that_finished_after_selection() -> None:
    """The latest selector should accept a target finishing before its stop RPC."""
    stub = MagicMock()
    stub.StopRun.side_effect = _AlreadyFinishedRpcError()

    _stop_runs(stub, [3], is_json=False, selector="latest")


def test_stop_selector_rechecks_unsuccessful_response() -> None:
    """A false stop response should succeed if the selected run is now finished."""
    stub = MagicMock()
    stub.StopRun.return_value = StopRunResponse(success=False)
    stub.ListRuns.return_value = ListRunsResponse(
        run_dict={
            3: _run(3, Status.FINISHED, "2026-08-20T11:00:00+00:00"),
        }
    )

    _stop_runs(stub, [3], is_json=False, selector="all")

    stub.ListRuns.assert_called_once_with(ListRunsRequest(run_id=3))
