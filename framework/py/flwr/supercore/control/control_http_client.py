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
"""HTTP client for the Control API."""

from collections.abc import Generator

import httpx

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AcceptInvitationRequest,
    AcceptInvitationResponse,
    AddAppRequest,
    AddAppResponse,
    AddNodeToFederationRequest,
    AddNodeToFederationResponse,
    ArchiveFederationRequest,
    ArchiveFederationResponse,
    BeginConnectorOAuthRequest,
    BeginConnectorOAuthResponse,
    BootstrapRequest,
    BootstrapResponse,
    CompleteConnectorOAuthRequest,
    CompleteConnectorOAuthResponse,
    ConfigureSimulationFederationRequest,
    ConfigureSimulationFederationResponse,
    CreateAgentRequest,
    CreateAgentResponse,
    CreateFederationRequest,
    CreateFederationResponse,
    CreateInvitationRequest,
    CreateInvitationResponse,
    DeleteAgentRequest,
    DeleteAgentResponse,
    DisconnectConnectorRequest,
    DisconnectConnectorResponse,
    GetAuthTokensRequest,
    GetAuthTokensResponse,
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
    GetRunSeriesRequest,
    GetRunSeriesResponse,
    ListAgentsRequest,
    ListAgentsResponse,
    ListAppAssociationsRequest,
    ListAppAssociationsResponse,
    ListAppsRequest,
    ListAppsResponse,
    ListAutomationsRequest,
    ListAutomationsResponse,
    ListConnectorsRequest,
    ListConnectorsResponse,
    ListFederationsRequest,
    ListFederationsResponse,
    ListInvitationsRequest,
    ListInvitationsResponse,
    ListNodesRequest,
    ListNodesResponse,
    ListRunSeriesEventsRequest,
    ListRunSeriesEventsResponse,
    ListRunSeriesRequest,
    ListRunSeriesResponse,
    ListRunsRequest,
    ListRunsResponse,
    PullArtifactsRequest,
    PullArtifactsResponse,
    RefreshAuthTokensRequest,
    RefreshAuthTokensResponse,
    RegisterNodeRequest,
    RegisterNodeResponse,
    RejectInvitationRequest,
    RejectInvitationResponse,
    RemoveAccountFromFederationRequest,
    RemoveAccountFromFederationResponse,
    RemoveAppRequest,
    RemoveAppResponse,
    RemoveNodeFromFederationRequest,
    RemoveNodeFromFederationResponse,
    RevokeInvitationRequest,
    RevokeInvitationResponse,
    SetFederationIconRequest,
    SetFederationIconResponse,
    ShowFederationRequest,
    ShowFederationResponse,
    StartAutomationRequest,
    StartAutomationResponse,
    StartRunRequest,
    StartRunResponse,
    StopAutomationRequest,
    StopAutomationResponse,
    StopRunRequest,
    StopRunResponse,
    StreamEventsRequest,
    StreamEventsResponse,
    StreamLogsRequest,
    StreamLogsResponse,
    StreamRunEventsRequest,
    StreamRunEventsResponse,
    UnregisterNodeRequest,
    UnregisterNodeResponse,
    UpdateAgentRequest,
    UpdateAgentResponse,
    UpdateRunSeriesDescriptionRequest,
    UpdateRunSeriesDescriptionResponse,
)
from flwr.proto.space_pb2 import (  # pylint: disable=E0611
    CreateSpaceRequest,
    CreateSpaceResponse,
    GetSpaceItemsRequest,
    GetSpaceItemsResponse,
    ListSpacesRequest,
    ListSpacesResponse,
    SendSpaceItemRequest,
    SendSpaceItemResponse,
)
from flwr.supercore.protobuf.client import ProtobufClient
from flwr.supercore.protobuf.constants import (
    CREATE_FEDERATION_NAMESPACE_MEDIA_TYPE,
    PROTOBUF_MEDIA_TYPE,
)

_CREATE_FEDERATION_PATH = "/v1/control/create-federation"
_NAMESPACE_UNSUPPORTED = (
    "The SuperLink did not confirm support for explicit federation namespaces. "
    "Use an updated SuperLink that exposes its OpenAPI schema."
)


# Match the Control API method names, including HTTP-only token refresh.
# pylint: disable=invalid-name
class ControlHttpClient(ProtobufClient):  # pylint: disable=too-many-public-methods
    """Protobuf-over-HTTP client for the Control API."""

    def Bootstrap(self, request: BootstrapRequest) -> BootstrapResponse:
        """Read the user feed cursor and visible Agents and Spaces."""
        return self._unary_unary(
            path="/v1/control/bootstrap",
            rpc_method="/flwr.proto.Control/Bootstrap",
            request=request,
            response_type=BootstrapResponse,
        )

    def StreamEvents(
        self, request: StreamEventsRequest
    ) -> Generator[StreamEventsResponse, None, None]:
        """Replay application events after a recipient-specific sequence."""
        return self._unary_stream(
            path="/v1/control/stream-events",
            rpc_method="/flwr.proto.Control/StreamEvents",
            request=request,
            response_type=StreamEventsResponse,
        )

    def CreateAgent(self, request: CreateAgentRequest) -> CreateAgentResponse:
        """Create an agent."""
        return self._unary_unary(
            path="/v1/control/create-agent",
            rpc_method="/flwr.proto.Control/CreateAgent",
            request=request,
            response_type=CreateAgentResponse,
        )

    def DeleteAgent(self, request: DeleteAgentRequest) -> DeleteAgentResponse:
        """Delete an agent."""
        return self._unary_unary(
            path="/v1/control/delete-agent",
            rpc_method="/flwr.proto.Control/DeleteAgent",
            request=request,
            response_type=DeleteAgentResponse,
        )

    def ListAgents(self, request: ListAgentsRequest) -> ListAgentsResponse:
        """List agents."""
        return self._unary_unary(
            path="/v1/control/list-agents",
            rpc_method="/flwr.proto.Control/ListAgents",
            request=request,
            response_type=ListAgentsResponse,
        )

    def UpdateAgent(self, request: UpdateAgentRequest) -> UpdateAgentResponse:
        """Replace an agent's configuration using its current version."""
        return self._unary_unary(
            path="/v1/control/update-agent",
            rpc_method="/flwr.proto.Control/UpdateAgent",
            request=request,
            response_type=UpdateAgentResponse,
        )

    def StartRun(self, request: StartRunRequest) -> StartRunResponse:
        """Start a run."""
        return self._unary_unary(
            path="/v1/control/start-run",
            rpc_method="/flwr.proto.Control/StartRun",
            request=request,
            response_type=StartRunResponse,
        )

    def StopRun(self, request: StopRunRequest) -> StopRunResponse:
        """Stop a run."""
        return self._unary_unary(
            path="/v1/control/stop-run",
            rpc_method="/flwr.proto.Control/StopRun",
            request=request,
            response_type=StopRunResponse,
        )

    def StartAutomation(
        self, request: StartAutomationRequest
    ) -> StartAutomationResponse:
        """Start an automation."""
        return self._unary_unary(
            path="/v1/control/start-automation",
            rpc_method="/flwr.proto.Control/StartAutomation",
            request=request,
            response_type=StartAutomationResponse,
        )

    def ListAutomations(
        self, request: ListAutomationsRequest
    ) -> ListAutomationsResponse:
        """List automations."""
        return self._unary_unary(
            path="/v1/control/list-automations",
            rpc_method="/flwr.proto.Control/ListAutomations",
            request=request,
            response_type=ListAutomationsResponse,
        )

    def StopAutomation(self, request: StopAutomationRequest) -> StopAutomationResponse:
        """Stop an automation."""
        return self._unary_unary(
            path="/v1/control/stop-automation",
            rpc_method="/flwr.proto.Control/StopAutomation",
            request=request,
            response_type=StopAutomationResponse,
        )

    def StreamLogs(
        self, request: StreamLogsRequest
    ) -> Generator[StreamLogsResponse, None, None]:
        """Stream logs for a run."""
        return self._unary_stream(
            path="/v1/control/stream-logs",
            rpc_method="/flwr.proto.Control/StreamLogs",
            request=request,
            response_type=StreamLogsResponse,
        )

    def ListRuns(self, request: ListRunsRequest) -> ListRunsResponse:
        """List runs."""
        return self._unary_unary(
            path="/v1/control/list-runs",
            rpc_method="/flwr.proto.Control/ListRuns",
            request=request,
            response_type=ListRunsResponse,
        )

    def CreateSpace(self, request: CreateSpaceRequest) -> CreateSpaceResponse:
        """Create space."""
        return self._unary_unary(
            path="/v1/control/create-space",
            rpc_method="/flwr.proto.Control/CreateSpace",
            request=request,
            response_type=CreateSpaceResponse,
        )

    def ListSpaces(self, request: ListSpacesRequest) -> ListSpacesResponse:
        """List spaces."""
        return self._unary_unary(
            path="/v1/control/list-spaces",
            rpc_method="/flwr.proto.Control/ListSpaces",
            request=request,
            response_type=ListSpacesResponse,
        )

    def GetSpaceItems(self, request: GetSpaceItemsRequest) -> GetSpaceItemsResponse:
        """Get space items."""
        return self._unary_unary(
            path="/v1/control/get-space-items",
            rpc_method="/flwr.proto.Control/GetSpaceItems",
            request=request,
            response_type=GetSpaceItemsResponse,
        )

    def SendSpaceItem(self, request: SendSpaceItemRequest) -> SendSpaceItemResponse:
        """Send a message item to a Space."""
        return self._unary_unary(
            path="/v1/control/send-space-item",
            rpc_method="/flwr.proto.Control/SendSpaceItem",
            request=request,
            response_type=SendSpaceItemResponse,
        )

    def ListRunSeries(self, request: ListRunSeriesRequest) -> ListRunSeriesResponse:
        """List run series."""
        return self._unary_unary(
            path="/v1/control/list-run-series",
            rpc_method="/flwr.proto.Control/ListRunSeries",
            request=request,
            response_type=ListRunSeriesResponse,
        )

    def GetRunSeries(self, request: GetRunSeriesRequest) -> GetRunSeriesResponse:
        """Get a run series."""
        return self._unary_unary(
            path="/v1/control/get-run-series",
            rpc_method="/flwr.proto.Control/GetRunSeries",
            request=request,
            response_type=GetRunSeriesResponse,
        )

    def UpdateRunSeriesDescription(
        self, request: UpdateRunSeriesDescriptionRequest
    ) -> UpdateRunSeriesDescriptionResponse:
        """Update a run series description."""
        return self._unary_unary(
            path="/v1/control/update-run-series-description",
            rpc_method="/flwr.proto.Control/UpdateRunSeriesDescription",
            request=request,
            response_type=UpdateRunSeriesDescriptionResponse,
        )

    def ListRunSeriesEvents(
        self, request: ListRunSeriesEventsRequest
    ) -> ListRunSeriesEventsResponse:
        """List events for all runs in a run series."""
        return self._unary_unary(
            path="/v1/control/list-run-series-events",
            rpc_method="/flwr.proto.Control/ListRunSeriesEvents",
            request=request,
            response_type=ListRunSeriesEventsResponse,
        )

    def GetLoginDetails(
        self, request: GetLoginDetailsRequest
    ) -> GetLoginDetailsResponse:
        """Get account login details."""
        return self._unary_unary(
            path="/v1/control/get-login-details",
            rpc_method="/flwr.proto.Control/GetLoginDetails",
            request=request,
            response_type=GetLoginDetailsResponse,
        )

    def GetAuthTokens(self, request: GetAuthTokensRequest) -> GetAuthTokensResponse:
        """Get account authentication tokens."""
        return self._unary_unary(
            path="/v1/control/get-auth-tokens",
            rpc_method="/flwr.proto.Control/GetAuthTokens",
            request=request,
            response_type=GetAuthTokensResponse,
        )

    def RefreshAuthTokens(
        self, request: RefreshAuthTokensRequest
    ) -> RefreshAuthTokensResponse:
        """Refresh account authentication tokens."""
        return self._unary_unary(
            path="/v1/control/refresh-auth-tokens",
            rpc_method="/flwr.proto.Control/RefreshAuthTokens",
            request=request,
            response_type=RefreshAuthTokensResponse,
        )

    def ListConnectors(self, request: ListConnectorsRequest) -> ListConnectorsResponse:
        """List OAuth connectors available to the authenticated account."""
        return self._unary_unary(
            path="/v1/control/list-connectors",
            rpc_method="/flwr.proto.Control/ListConnectors",
            request=request,
            response_type=ListConnectorsResponse,
        )

    def DisconnectConnector(
        self, request: DisconnectConnectorRequest
    ) -> DisconnectConnectorResponse:
        """Disconnect connector credentials for the authenticated account."""
        return self._unary_unary(
            path="/v1/control/disconnect-connector",
            rpc_method="/flwr.proto.Control/DisconnectConnector",
            request=request,
            response_type=DisconnectConnectorResponse,
        )

    def BeginConnectorOAuth(
        self, request: BeginConnectorOAuthRequest
    ) -> BeginConnectorOAuthResponse:
        """Begin an OAuth connector authorization flow."""
        return self._unary_unary(
            path="/v1/control/begin-connector-oauth",
            rpc_method="/flwr.proto.Control/BeginConnectorOAuth",
            request=request,
            response_type=BeginConnectorOAuthResponse,
        )

    def CompleteConnectorOAuth(
        self, request: CompleteConnectorOAuthRequest
    ) -> CompleteConnectorOAuthResponse:
        """Complete an OAuth connector authorization flow."""
        return self._unary_unary(
            path="/v1/control/complete-connector-oauth",
            rpc_method="/flwr.proto.Control/CompleteConnectorOAuth",
            request=request,
            response_type=CompleteConnectorOAuthResponse,
        )

    def PullArtifacts(self, request: PullArtifactsRequest) -> PullArtifactsResponse:
        """Pull artifacts generated during a run."""
        return self._unary_unary(
            path="/v1/control/pull-artifacts",
            rpc_method="/flwr.proto.Control/PullArtifacts",
            request=request,
            response_type=PullArtifactsResponse,
        )

    def RegisterNode(self, request: RegisterNodeRequest) -> RegisterNodeResponse:
        """Register a SuperNode."""
        return self._unary_unary(
            path="/v1/control/register-node",
            rpc_method="/flwr.proto.Control/RegisterNode",
            request=request,
            response_type=RegisterNodeResponse,
        )

    def UnregisterNode(self, request: UnregisterNodeRequest) -> UnregisterNodeResponse:
        """Unregister a SuperNode."""
        return self._unary_unary(
            path="/v1/control/unregister-node",
            rpc_method="/flwr.proto.Control/UnregisterNode",
            request=request,
            response_type=UnregisterNodeResponse,
        )

    def ListNodes(self, request: ListNodesRequest) -> ListNodesResponse:
        """List SuperNodes."""
        return self._unary_unary(
            path="/v1/control/list-nodes",
            rpc_method="/flwr.proto.Control/ListNodes",
            request=request,
            response_type=ListNodesResponse,
        )

    def ListFederations(
        self, request: ListFederationsRequest
    ) -> ListFederationsResponse:
        """List federations."""
        return self._unary_unary(
            path="/v1/control/list-federations",
            rpc_method="/flwr.proto.Control/ListFederations",
            request=request,
            response_type=ListFederationsResponse,
        )

    def ListApps(self, request: ListAppsRequest) -> ListAppsResponse:
        """List apps in a federation."""
        return self._unary_unary(
            path="/v1/control/list-apps",
            rpc_method="/flwr.proto.Control/ListApps",
            request=request,
            response_type=ListAppsResponse,
        )

    def ListAppAssociations(
        self, request: ListAppAssociationsRequest
    ) -> ListAppAssociationsResponse:
        """List federations associated with an app."""
        return self._unary_unary(
            path="/v1/control/list-app-associations",
            rpc_method="/flwr.proto.Control/ListAppAssociations",
            request=request,
            response_type=ListAppAssociationsResponse,
        )

    def AddApp(self, request: AddAppRequest) -> AddAppResponse:
        """Add an app to a federation."""
        return self._unary_unary(
            path="/v1/control/add-app",
            rpc_method="/flwr.proto.Control/AddApp",
            request=request,
            response_type=AddAppResponse,
        )

    def RemoveApp(self, request: RemoveAppRequest) -> RemoveAppResponse:
        """Remove an app from a federation."""
        return self._unary_unary(
            path="/v1/control/remove-app",
            rpc_method="/flwr.proto.Control/RemoveApp",
            request=request,
            response_type=RemoveAppResponse,
        )

    def ShowFederation(self, request: ShowFederationRequest) -> ShowFederationResponse:
        """Show a federation."""
        return self._unary_unary(
            path="/v1/control/show-federation",
            rpc_method="/flwr.proto.Control/ShowFederation",
            request=request,
            response_type=ShowFederationResponse,
        )

    def SetFederationIcon(
        self, request: SetFederationIconRequest
    ) -> SetFederationIconResponse:
        """Set or clear a federation icon."""
        return self._unary_unary(
            path="/v1/control/set-federation-icon",
            rpc_method="/flwr.proto.Control/SetFederationIcon",
            request=request,
            response_type=SetFederationIconResponse,
        )

    def CreateFederation(
        self,
        request: CreateFederationRequest,
        *,
        require_namespace_support: bool = False,
    ) -> CreateFederationResponse:
        """Create a federation."""
        content_type = PROTOBUF_MEDIA_TYPE
        if require_namespace_support:
            if not self._supports_create_federation_namespace():
                raise ValueError(_NAMESPACE_UNSUPPORTED)
            # An older server may receive the POST during a rolling upgrade.
            content_type = CREATE_FEDERATION_NAMESPACE_MEDIA_TYPE

        try:
            return self._unary_unary(
                path=_CREATE_FEDERATION_PATH,
                rpc_method="/flwr.proto.Control/CreateFederation",
                request=request,
                response_type=CreateFederationResponse,
                content_type=content_type,
            )
        except httpx.HTTPStatusError as exc:
            if (
                require_namespace_support
                and exc.response.status_code == httpx.codes.UNSUPPORTED_MEDIA_TYPE
            ):
                raise ValueError(_NAMESPACE_UNSUPPORTED) from exc
            raise

    def _supports_create_federation_namespace(self) -> bool:
        response = self._client.get(f"{self._base_url}/openapi.json")
        if not response.is_success:
            return False
        try:
            operation = response.json()["paths"][_CREATE_FEDERATION_PATH]["post"]
            content = operation["requestBody"]["content"]
        except (KeyError, TypeError, ValueError):
            return False
        return (
            isinstance(content, dict)
            and CREATE_FEDERATION_NAMESPACE_MEDIA_TYPE in content
        )

    def ArchiveFederation(
        self, request: ArchiveFederationRequest
    ) -> ArchiveFederationResponse:
        """Archive a federation."""
        return self._unary_unary(
            path="/v1/control/archive-federation",
            rpc_method="/flwr.proto.Control/ArchiveFederation",
            request=request,
            response_type=ArchiveFederationResponse,
        )

    def AddNodeToFederation(
        self, request: AddNodeToFederationRequest
    ) -> AddNodeToFederationResponse:
        """Add a SuperNode to a federation."""
        return self._unary_unary(
            path="/v1/control/add-node-to-federation",
            rpc_method="/flwr.proto.Control/AddNodeToFederation",
            request=request,
            response_type=AddNodeToFederationResponse,
        )

    def RemoveNodeFromFederation(
        self, request: RemoveNodeFromFederationRequest
    ) -> RemoveNodeFromFederationResponse:
        """Remove a SuperNode from a federation."""
        return self._unary_unary(
            path="/v1/control/remove-node-from-federation",
            rpc_method="/flwr.proto.Control/RemoveNodeFromFederation",
            request=request,
            response_type=RemoveNodeFromFederationResponse,
        )

    def RemoveAccountFromFederation(
        self, request: RemoveAccountFromFederationRequest
    ) -> RemoveAccountFromFederationResponse:
        """Remove an account from a federation."""
        return self._unary_unary(
            path="/v1/control/remove-account-from-federation",
            rpc_method="/flwr.proto.Control/RemoveAccountFromFederation",
            request=request,
            response_type=RemoveAccountFromFederationResponse,
        )

    def CreateInvitation(
        self, request: CreateInvitationRequest
    ) -> CreateInvitationResponse:
        """Create a federation invitation."""
        return self._unary_unary(
            path="/v1/control/create-invitation",
            rpc_method="/flwr.proto.Control/CreateInvitation",
            request=request,
            response_type=CreateInvitationResponse,
        )

    def ListInvitations(
        self, request: ListInvitationsRequest
    ) -> ListInvitationsResponse:
        """List federation invitations."""
        return self._unary_unary(
            path="/v1/control/list-invitations",
            rpc_method="/flwr.proto.Control/ListInvitations",
            request=request,
            response_type=ListInvitationsResponse,
        )

    def AcceptInvitation(
        self, request: AcceptInvitationRequest
    ) -> AcceptInvitationResponse:
        """Accept a federation invitation."""
        return self._unary_unary(
            path="/v1/control/accept-invitation",
            rpc_method="/flwr.proto.Control/AcceptInvitation",
            request=request,
            response_type=AcceptInvitationResponse,
        )

    def RejectInvitation(
        self, request: RejectInvitationRequest
    ) -> RejectInvitationResponse:
        """Reject a federation invitation."""
        return self._unary_unary(
            path="/v1/control/reject-invitation",
            rpc_method="/flwr.proto.Control/RejectInvitation",
            request=request,
            response_type=RejectInvitationResponse,
        )

    def RevokeInvitation(
        self, request: RevokeInvitationRequest
    ) -> RevokeInvitationResponse:
        """Revoke a federation invitation."""
        return self._unary_unary(
            path="/v1/control/revoke-invitation",
            rpc_method="/flwr.proto.Control/RevokeInvitation",
            request=request,
            response_type=RevokeInvitationResponse,
        )

    def ConfigureSimulationFederation(
        self, request: ConfigureSimulationFederationRequest
    ) -> ConfigureSimulationFederationResponse:
        """Configure a federation for simulation."""
        return self._unary_unary(
            path="/v1/control/configure-simulation-federation",
            rpc_method="/flwr.proto.Control/ConfigureSimulationFederation",
            request=request,
            response_type=ConfigureSimulationFederationResponse,
        )

    def StreamRunEvents(
        self, request: StreamRunEventsRequest
    ) -> Generator[StreamRunEventsResponse, None, None]:
        """Stream task events for a run."""
        return self._unary_stream(
            path="/v1/control/stream-run-events",
            rpc_method="/flwr.proto.Control/StreamRunEvents",
            request=request,
            response_type=StreamRunEventsResponse,
        )
