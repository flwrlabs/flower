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
"""Flower command line interface `federation invite list` command."""


from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from flwr.cli.utils import (
    cli_output_control_stub,
    flwr_cli_grpc_exc_handler,
    print_json_to_stdout,
)
from flwr.common.constant import CliOutputFormat
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListInvitationsRequest,
    ListInvitationsResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub
from flwr.proto.federation_pb2 import Invitation  # pylint: disable=E0611
from flwr.supercore.constant import InvitationStatus
from flwr.supercore.date import isoformat8601_utc

from ..error_handlers import handle_invite_grpc_error

_STATUS_TO_COLOR: dict[str, str] = {
    InvitationStatus.PENDING: "yellow",
    InvitationStatus.ACCEPTED: "green",
    InvitationStatus.REJECTED: "red",
    InvitationStatus.REVOKED: "bright_black",
    InvitationStatus.EXPIRED: "bright_black",
}


def ls(
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
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show non-pending invitations.",
        ),
    ] = False,
) -> None:
    """List invitations addressed to you and invitations created by you (alias: ls)."""
    with cli_output_control_stub(superlink, output_format) as (stub, is_json):
        request = ListInvitationsRequest()
        _list_invitations(
            stub=stub,
            request=request,
            is_json=is_json,
            verbose=verbose,
        )


def _list_invitations(
    stub: ControlStub,
    request: ListInvitationsRequest,
    is_json: bool,
    verbose: bool,
) -> None:
    """Send a list invitations request."""
    with flwr_cli_grpc_exc_handler(handle_invite_grpc_error):
        response: ListInvitationsResponse = stub.ListInvitations(request)

    created_invitations = _filter_invitations(response.created_invitations, verbose)
    received_invitations = _filter_invitations(response.received_invitations, verbose)

    if is_json:
        print_json_to_stdout(_to_json(created_invitations, received_invitations))
    else:
        created_table = _to_invitations_table(
            created_invitations, is_received=False, verbose=verbose
        )
        received_table = _to_invitations_table(
            received_invitations, is_received=True, verbose=verbose
        )
        console = Console()
        console.print()
        console.print(created_table)
        console.print()
        console.print(received_table)
        if not verbose and not received_invitations and not created_invitations:
            console.print(
                "[dim]No pending invitations. Use --verbose to show all statuses.[/dim]"
            )


def _filter_invitations(
    invitations: Sequence[Invitation], verbose: bool
) -> Sequence[Invitation]:
    """Filter invitations for presentation."""
    if verbose:
        return invitations
    return [iv for iv in invitations if iv.status == InvitationStatus.PENDING]


def _format_datetime(dt_str: str) -> str:
    """Format ISO 8601 timestamp as `YYYY-MM-DD HH:MM:SSZ`."""
    dt = datetime.fromisoformat(dt_str) if dt_str else None
    return isoformat8601_utc(dt).replace("T", " ") if dt else "N/A"


def _to_invitations_table(
    invitations: Sequence[Invitation],
    is_received: bool,
    verbose: bool,
) -> Table:
    """Render an invitations table."""
    title = "Received Invitations" if is_received else "Created Invitations"
    actor_column = "Invited By" if is_received else "Invited"

    table = Table(
        title=title,
        header_style="bold cyan",
        show_lines=True,
    )
    table.add_column(Text("Federation", justify="center"), no_wrap=True)
    table.add_column(Text(actor_column, justify="center"), no_wrap=True)
    table.add_column(Text("Status", justify="center"), no_wrap=True)
    if verbose:
        table.add_column(Text("Created @", justify="center"))
        table.add_column(Text("Status Changed @", justify="center"))

    for invitation in invitations:
        color = _STATUS_TO_COLOR.get(invitation.status, "bright_black")
        row = [
            invitation.federation_name,
            invitation.inviter.name if is_received else invitation.invitee.name,
            f"[{color}]{invitation.status}[/{color}]",
        ]
        if verbose:
            row += [
                _format_datetime(invitation.created_at),
                _format_datetime(invitation.status_changed_at),
            ]
        table.add_row(*row)

    return table


def _to_json(
    created_invitations: Sequence[Invitation],
    received_invitations: Sequence[Invitation],
) -> dict[str, Any]:
    """Convert invitations to JSON serializable structure."""
    return {
        "success": True,
        "created-invitations": [
            {
                "federation-name": invitation.federation_name,
                "invited": invitation.invitee.name,
                "status": invitation.status,
                "created-at": _format_datetime(invitation.created_at),
                "status-changed-at": _format_datetime(invitation.status_changed_at),
            }
            for invitation in created_invitations
        ],
        "received-invitations": [
            {
                "federation-name": invitation.federation_name,
                "invited-by": invitation.inviter.name,
                "status": invitation.status,
                "created-at": _format_datetime(invitation.created_at),
                "status-changed-at": _format_datetime(invitation.status_changed_at),
            }
            for invitation in received_invitations
        ],
    }
