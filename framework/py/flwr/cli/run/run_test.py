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
"""Tests for the capabilities-file input of ``flwr run``."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import click
import pytest
from typer.testing import CliRunner

from flwr.cli.app import app
from flwr.proto.control_pb2 import StartRunResponse  # pylint: disable=E0611

from .run import _load_capability_packages

PARTICIPANT_ID = "flwr-p384-spki-pem-sha256:" + "a" * 64
run_module = importlib.import_module("flwr.cli.run.run")


def test_load_capability_packages(tmp_path: Path) -> None:
    """Load opaque capability strings from the versioned file."""
    path = tmp_path / "capabilities.json"
    path.write_text(
        json.dumps({"version": "v1", "capabilities": {PARTICIPANT_ID: "opaque-data"}}),
        encoding="utf-8",
    )

    assert _load_capability_packages(path) == {PARTICIPANT_ID: b"opaque-data"}


@pytest.mark.parametrize(
    "document",
    [
        {"version": "v2", "capabilities": {PARTICIPANT_ID: "data"}},
        {"version": "v1", "capabilities": {}},
        {"version": "v1", "capabilities": {"node-1": "data"}},
        {"version": "v1", "capabilities": {PARTICIPANT_ID: ""}},
    ],
)
def test_reject_invalid_capabilities_file(tmp_path: Path, document: object) -> None:
    """Reject malformed or unsupported capabilities-file contracts."""
    path = tmp_path / "capabilities.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(click.ClickException):
        _load_capability_packages(path)


def test_capabilities_file_reaches_start_run_request(tmp_path: Path) -> None:
    """Propagate the user-facing option into StartRunRequest packages."""
    path = tmp_path / "capabilities.json"
    path.write_text(
        json.dumps({"version": "v1", "capabilities": {PARTICIPANT_ID: "opaque"}}),
        encoding="utf-8",
    )
    control_client = MagicMock()
    control_client.StartRun.return_value = StartRunResponse(
        run_id=1, federation="local"
    )
    connection = SimpleNamespace(
        federation="local",
        name="local",
        address="unused",
        options=None,
    )

    with (
        patch("flwr.cli.app.warn_if_flwr_update_available"),
        patch.object(run_module, "read_superlink_connection", return_value=connection),
        patch.object(
            run_module,
            "init_http_client_from_connection",
            return_value=control_client,
        ),
    ):
        result = CliRunner().invoke(
            app,
            [
                "run",
                "@flwrlabs/quickstart-numpy",
                "--capabilities-file",
                str(path),
            ],
        )

    assert result.exit_code == 0, result.output
    request = control_client.StartRun.call_args.args[0]
    assert dict(request.capability_packages) == {PARTICIPANT_ID: b"opaque"}
    control_client.close.assert_called_once()


def test_run_help_uses_accepted_capabilities_file_option() -> None:
    """Expose only the accepted plural option name."""
    result = CliRunner().invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "--capabilities-file" in result.output
    assert "--capability-file" not in result.output
