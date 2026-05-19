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
"""Flower command line interface `federation remove-account` command."""


from typing import Annotated, Literal

import typer

from flwr.common.constant import CliOutputFormat
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    RemoveAccountFromFederationRequest,
    RemoveAccountFromFederationResponse,
)
from flwr.proto.control_pb2_grpc import ControlStub

from ..utils import (
    cli_output_control_stub,
    flwr_cli_grpc_exc_handler,
    print_json_to_stdout,
)


def remove_account(
    federation: Annotated[
        str,
        typer.Argument(help="Name of the federation to remove an account from."),
    ],
    account_name: Annotated[
        str | None,
        typer.Argument(help="Name of the account to remove from the federation."),
    ] = None,
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
    """Remove an account from an existing federation."""
    with cli_output_control_stub(superlink, output_format) as (stub, is_json):
        request = RemoveAccountFromFederationRequest(
            federation_name=federation, account_name=account_name
        )
        _remove_account_from_federation(
            stub=stub,
            request=request,
            is_json=is_json,
        )


def _remove_account_from_federation(  # pylint: disable=W0613
    stub: ControlStub,
    request: RemoveAccountFromFederationRequest,
    is_json: bool,
) -> None:
    """Remove an account from a federation."""
    with flwr_cli_grpc_exc_handler():
        _: RemoveAccountFromFederationResponse = stub.RemoveAccountFromFederation(
            request
        )

    if is_json:
        print_json_to_stdout({"success": True})
    else:
        message = (
            f"✅ Removed account '{request.account_name}' from federation "
            f"'{request.federation_name}'."
            if request.account_name
            else f"✅ Your account was removed from federation "
            f"'{request.federation_name}'."
        )
        typer.secho(message)
