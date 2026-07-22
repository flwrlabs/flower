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
"""Control API router."""

from typing import Annotated

from fastapi import APIRouter, Depends

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AcceptInvitationRequest,
    AcceptInvitationResponse,
    AddNodeToFederationRequest,
    AddNodeToFederationResponse,
    ArchiveFederationRequest,
    ArchiveFederationResponse,
    ConfigureSimulationFederationRequest,
    ConfigureSimulationFederationResponse,
    CreateFederationRequest,
    CreateFederationResponse,
    CreateInvitationRequest,
    CreateInvitationResponse,
    GetAuthTokensRequest,
    GetAuthTokensResponse,
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
    GetRunSeriesRequest,
    GetRunSeriesResponse,
    ListFederationsRequest,
    ListFederationsResponse,
    ListInvitationsRequest,
    ListInvitationsResponse,
    ListNodesRequest,
    ListNodesResponse,
    ListRunSeriesRequest,
    ListRunSeriesResponse,
    ListRunsRequest,
    ListRunsResponse,
    RegisterNodeRequest,
    RegisterNodeResponse,
    RejectInvitationRequest,
    RejectInvitationResponse,
    RemoveAccountFromFederationRequest,
    RemoveAccountFromFederationResponse,
    RemoveNodeFromFederationRequest,
    RemoveNodeFromFederationResponse,
    RevokeInvitationRequest,
    RevokeInvitationResponse,
    ShowFederationRequest,
    ShowFederationResponse,
    StartRunRequest,
    StartRunResponse,
    StopRunRequest,
    StopRunResponse,
    UnregisterNodeRequest,
    UnregisterNodeResponse,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.protobuf.routing import ProtobufRoute
from flwr.supercore.protobuf.translation import get_protobuf_request
from flwr.superlink.auth_plugin import ControlAuthnPlugin
from flwr.superlink.dependencies.account import get_account, get_authn_plugin
from flwr.superlink.dependencies.linkstate import get_linkstate
from flwr.superlink.servicer.control import control_handlers

router = APIRouter(prefix="/control", tags=["control"], route_class=ProtobufRoute)

LinkStateDependency = Annotated[LinkState, Depends(get_linkstate)]
AccountDependency = Annotated[AccountInfo, Depends(get_account)]
AuthnPluginDependency = Annotated[ControlAuthnPlugin, Depends(get_authn_plugin)]


@router.post("/start-run")
def start_run(
    request: Annotated[StartRunRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> StartRunResponse:
    """Start a run."""
    # Temporary: pass an empty Fleet API type
    return control_handlers.start_run(request, account, linkstate, "")


@router.post("/list-runs")
def list_runs(
    request: Annotated[ListRunsRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> ListRunsResponse:
    """List runs."""
    return control_handlers.list_runs(request, account, linkstate)


@router.post("/list-run-series")
def list_run_series(
    request: Annotated[ListRunSeriesRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> ListRunSeriesResponse:
    """List run series."""
    return control_handlers.list_run_series(request, account, linkstate)


@router.post("/get-run-series")
def get_run_series(
    request: Annotated[GetRunSeriesRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> GetRunSeriesResponse:
    """Get a run series."""
    return control_handlers.get_run_series(request, account, linkstate)


@router.post("/stop-run")
def stop_run(
    request: Annotated[StopRunRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> StopRunResponse:
    """Stop a run."""
    return control_handlers.stop_run(request, account, linkstate)


@router.post("/get-login-details")
def get_login_details(
    request: Annotated[GetLoginDetailsRequest, Depends(get_protobuf_request)],
    authn_plugin: AuthnPluginDependency,
) -> GetLoginDetailsResponse:
    """Get login details."""
    return control_handlers.get_login_details(request, authn_plugin)


@router.post("/get-auth-tokens")
def get_auth_tokens(
    request: Annotated[GetAuthTokensRequest, Depends(get_protobuf_request)],
    authn_plugin: AuthnPluginDependency,
) -> GetAuthTokensResponse:
    """Get authentication tokens."""
    return control_handlers.get_auth_tokens(request, authn_plugin)


@router.post("/register-node")
def register_node(
    request: Annotated[RegisterNodeRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> RegisterNodeResponse:
    """Register a SuperNode."""
    return control_handlers.register_node(request, account, linkstate)


@router.post("/unregister-node")
def unregister_node(
    request: Annotated[UnregisterNodeRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> UnregisterNodeResponse:
    """Unregister a SuperNode."""
    return control_handlers.unregister_node(request, account, linkstate)


@router.post("/list-nodes")
def list_nodes(
    request: Annotated[ListNodesRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> ListNodesResponse:
    """List SuperNodes."""
    return control_handlers.list_nodes(request, account, linkstate)


@router.post("/list-federations")
def list_federations(
    request: Annotated[ListFederationsRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> ListFederationsResponse:
    """List federations."""
    return control_handlers.list_federations(request, account, linkstate)


@router.post("/show-federation")
def show_federation(
    request: Annotated[ShowFederationRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> ShowFederationResponse:
    """Show a federation."""
    return control_handlers.show_federation(request, account, linkstate)


@router.post("/create-federation")
def create_federation(
    request: Annotated[CreateFederationRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> CreateFederationResponse:
    """Create a federation."""
    return control_handlers.create_federation(request, account, linkstate)


@router.post("/archive-federation")
def archive_federation(
    request: Annotated[ArchiveFederationRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> ArchiveFederationResponse:
    """Archive a federation."""
    return control_handlers.archive_federation(request, account, linkstate)


@router.post("/add-node-to-federation")
def add_node_to_federation(
    request: Annotated[AddNodeToFederationRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> AddNodeToFederationResponse:
    """Add a SuperNode to a federation."""
    return control_handlers.add_node_to_federation(request, account, linkstate)


@router.post("/remove-node-from-federation")
def remove_node_from_federation(
    request: Annotated[RemoveNodeFromFederationRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> RemoveNodeFromFederationResponse:
    """Remove a SuperNode from a federation."""
    return control_handlers.remove_node_from_federation(request, account, linkstate)


@router.post("/remove-account-from-federation")
def remove_account_from_federation(
    request: Annotated[
        RemoveAccountFromFederationRequest, Depends(get_protobuf_request)
    ],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> RemoveAccountFromFederationResponse:
    """Remove an account from a federation."""
    return control_handlers.remove_account_from_federation(request, account, linkstate)


@router.post("/create-invitation")
def create_invitation(
    request: Annotated[CreateInvitationRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> CreateInvitationResponse:
    """Create a federation invitation."""
    return control_handlers.create_invitation(request, account, linkstate)


@router.post("/list-invitations")
def list_invitations(
    request: Annotated[ListInvitationsRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> ListInvitationsResponse:
    """List federation invitations."""
    return control_handlers.list_invitations(request, account, linkstate)


@router.post("/accept-invitation")
def accept_invitation(
    request: Annotated[AcceptInvitationRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> AcceptInvitationResponse:
    """Accept a federation invitation."""
    return control_handlers.accept_invitation(request, account, linkstate)


@router.post("/reject-invitation")
def reject_invitation(
    request: Annotated[RejectInvitationRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> RejectInvitationResponse:
    """Reject a federation invitation."""
    return control_handlers.reject_invitation(request, account, linkstate)


@router.post("/revoke-invitation")
def revoke_invitation(
    request: Annotated[RevokeInvitationRequest, Depends(get_protobuf_request)],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> RevokeInvitationResponse:
    """Revoke a federation invitation."""
    return control_handlers.revoke_invitation(request, account, linkstate)


@router.post("/configure-simulation-federation")
def configure_simulation_federation(
    request: Annotated[
        ConfigureSimulationFederationRequest, Depends(get_protobuf_request)
    ],
    linkstate: LinkStateDependency,
    account: AccountDependency,
) -> ConfigureSimulationFederationResponse:
    """Configure a federation for simulation."""
    return control_handlers.configure_simulation_federation(request, account, linkstate)
