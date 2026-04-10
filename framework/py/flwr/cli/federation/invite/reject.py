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
"""Flower command line interface `federation invite reject` command."""


from typing import Annotated, Literal

import typer

from flwr.cli.utils import (
    cli_output_control_stub,
    flwr_cli_grpc_exc_handler,
    print_json_to_stdout,
)
from flwr.common.constant import CliOutputFormat
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    RejectInvitationRequest,
    RejectInvitationResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub

from ..error_handlers import handle_invite_grpc_error


def reject(
    federation: Annotated[
        str,
        typer.Argument(help="Name of the federation."),
    ],
    superlink: Annotated[
        str | None,
        typer.Argument(help="Name of the SuperLink connection."),
    ] = None,
    output_format: Annotated[
        Literal["default", "json"],
        typer.Option(
            "--format",
            case_sensitive=False,
            help="Format output using 'default' view or 'json'",
        ),
    ] = CliOutputFormat.DEFAULT,
) -> None:
    """Reject an invitation to join a federation."""
    with cli_output_control_stub(superlink, output_format) as (stub, is_json):
        request = RejectInvitationRequest(federation_name=federation)
        _reject_invitation(stub=stub, request=request, is_json=is_json)


def _reject_invitation(
    stub: ControlStub,
    request: RejectInvitationRequest,
    is_json: bool,
) -> None:
    """Send a reject invitation request."""
    with flwr_cli_grpc_exc_handler(handle_invite_grpc_error):
        _: RejectInvitationResponse = stub.RejectInvitation(request)

    if is_json:
        print_json_to_stdout({"success": True})
    else:
        typer.secho(f"✅ Rejected invitation to join '{request.federation_name}'.")
