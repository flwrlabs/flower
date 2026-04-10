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
"""Flower command line interface `federation invite revoke` command."""


from typing import Annotated, Literal

import typer

from flwr.cli.utils import (
    cli_output_control_stub,
    flwr_cli_grpc_exc_handler,
    print_json_to_stdout,
)
from flwr.common.constant import CliOutputFormat
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    RevokeInvitationRequest,
    RevokeInvitationResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub

from ..error_handlers import handle_invite_grpc_error


def revoke(
    account: Annotated[
        str,
        typer.Argument(
            help="Name of the Flower account whose invitation should be revoked."
        ),
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
    """Revoke a pending invitation."""
    with cli_output_control_stub(superlink, output_format) as (stub, is_json):
        request = RevokeInvitationRequest(
            invitee_account_name=account,
            federation_name=federation,
        )
        _revoke_invitation(stub=stub, request=request, is_json=is_json)


def _revoke_invitation(
    stub: ControlStub,
    request: RevokeInvitationRequest,
    is_json: bool,
) -> None:
    """Send a revoke invitation request."""
    with flwr_cli_grpc_exc_handler(handle_invite_grpc_error):
        _: RevokeInvitationResponse = stub.RevokeInvitation(request)

    if is_json:
        print_json_to_stdout({"success": True})
    else:
        typer.secho(
            f"✅ Revoked invitation for '{request.invitee_account_name}' to join "
            f"'{request.federation_name}'."
        )
