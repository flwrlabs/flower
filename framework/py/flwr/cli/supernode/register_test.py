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
"""Tests for the Flower CLI SuperNode register command."""

from unittest.mock import Mock, patch

from typer.testing import CliRunner

from flwr.cli.app import app
from flwr.proto.control_pb2 import RegisterNodeResponse  # pylint: disable=E0611


def test_register_command_sends_name() -> None:
    """Parse and send the optional name in the registration request."""
    client = Mock()
    client.RegisterNode.return_value = RegisterNodeResponse(node_id=1)

    with (
        patch("flwr.cli.app.warn_if_flwr_update_available"),
        patch(
            "flwr.cli.supernode.register.try_load_public_key",
            return_value=b"public-key",
        ),
        patch("flwr.cli.supernode.register.migrate"),
        patch("flwr.cli.supernode.register.read_superlink_connection"),
        patch(
            "flwr.cli.supernode.register.init_http_client_from_connection",
            return_value=client,
        ),
    ):
        result = CliRunner().invoke(
            app,
            [
                "supernode",
                "register",
                "public-key.pub",
                "test-superlink",
                "--name",
                "London SuperNode",
            ],
        )

    assert result.exit_code == 0
    request = client.RegisterNode.call_args.kwargs["request"]
    assert request.name == "London SuperNode"
