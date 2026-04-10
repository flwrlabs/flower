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
"""Flower command line interface `federation invite create` command."""


from typing import Annotated, Literal

import typer

from flwr.cli.utils import (
    cli_output_control_stub,
    flwr_cli_grpc_exc_handler,
    print_json_to_stdout,
)
from flwr.common.constant import CliOutputFormat
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    CreateInvitationRequest,
    CreateInvitationResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub

from ..error_handlers import handle_invite_grpc_error


def create(
    account: Annotated[
        str,
        typer.Argument(help="Name of the Flower account to invite."),
    ],
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
    """Create an invitation to join a federation."""
    with cli_output_control_stub(superlink, output_format) as (stub, is_json):
        request = CreateInvitationRequest(
            invitee_account_name=account,
            federation_name=federation,
        )
        _create_invitation(stub=stub, request=request, is_json=is_json)


def _create_invitation(
    stub: ControlStub,
    request: CreateInvitationRequest,
    is_json: bool,
) -> None:
    """Send a create invitation request."""
    with flwr_cli_grpc_exc_handler(handle_invite_grpc_error):
        _: CreateInvitationResponse = stub.CreateInvitation(request)

    if is_json:
        print_json_to_stdout({"success": True})
    else:
        typer.secho(
            f"✅ Created invitation for '{request.invitee_account_name}' to join "
            f"'{request.federation_name}'."
        )
