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

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AcceptInvitationRequest,
    AcceptInvitationResponse,
    AddAppRequest,
    AddAppResponse,
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
    GetRunSeriesRequest,
    GetRunSeriesResponse,
    ListAppsRequest,
    ListAppsResponse,
    ListAutomationsRequest,
    ListAutomationsResponse,
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
    PullArtifactsRequest,
    PullArtifactsResponse,
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
    StreamLogsRequest,
    StreamLogsResponse,
    StreamRunEventsRequest,
    StreamRunEventsResponse,
    UnregisterNodeRequest,
    UnregisterNodeResponse,
)
from flwr.supercore.protobuf.client import ProtobufClient


# Match the method names defined by the Control protobuf service.
# pylint: disable=invalid-name
class ControlHttpClient(ProtobufClient):  # pylint: disable=too-many-public-methods
    """Protobuf-over-HTTP client for the Control API."""

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

    def CreateFederation(
        self, request: CreateFederationRequest
    ) -> CreateFederationResponse:
        """Create a federation."""
        return self._unary_unary(
            path="/v1/control/create-federation",
            rpc_method="/flwr.proto.Control/CreateFederation",
            request=request,
            response_type=CreateFederationResponse,
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
