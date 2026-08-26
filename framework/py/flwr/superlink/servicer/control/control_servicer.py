# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Control API servicer."""

from collections.abc import Generator

import grpc

from flwr.proto import control_pb2_grpc  # pylint: disable=E0611
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
    CompleteConnectorOAuthRequest,
    CompleteConnectorOAuthResponse,
    ConfigureSimulationFederationRequest,
    ConfigureSimulationFederationResponse,
    CreateFederationRequest,
    CreateFederationResponse,
    CreateInvitationRequest,
    CreateInvitationResponse,
    DisconnectConnectorRequest,
    DisconnectConnectorResponse,
    GetAuthTokensRequest,
    GetAuthTokensResponse,
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
    GetRunSeriesRequest,
    GetRunSeriesResponse,
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
from flwr.server.superlink.linkstate import LinkStateFactory
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.utils import get_metadata_str
from flwr.superlink.artifact_provider import ArtifactProvider
from flwr.superlink.auth_plugin import ControlAuthnPlugin
from flwr.superlink.run_source import (
    RUN_SOURCE_METADATA_KEY,
    resolve_run_start_source,
)

from . import control_handlers
from .control_account_auth_interceptor import get_current_account_info
from .control_handlers import _resolve_federation_id


# pylint: disable=too-many-public-methods
class ControlServicer(control_pb2_grpc.ControlServicer):
    """Control API servicer."""

    def __init__(  # pylint: disable=R0913, R0917
        self,
        linkstate_factory: LinkStateFactory,
        objectstore_factory: ObjectStoreFactory,
        authn_plugin: ControlAuthnPlugin,
        artifact_provider: ArtifactProvider | None = None,
        fleet_api_type: str | None = None,
    ) -> None:
        self.linkstate_factory = linkstate_factory
        self.objectstore_factory = objectstore_factory
        self.authn_plugin = authn_plugin
        self.artifact_provider = artifact_provider
        self.fleet_api_type = fleet_api_type

    def StartRun(
        self, request: StartRunRequest, context: grpc.ServicerContext
    ) -> StartRunResponse:
        """Create run ID."""
        # Best-effort analytics attribution only;
        # trust caller-provided labels and default missing sources to unknown.
        metadata = context.invocation_metadata()
        run_source = get_metadata_str(metadata, RUN_SOURCE_METADATA_KEY)

        return control_handlers.start_run(
            request,
            _get_account(),
            self.linkstate_factory.state(),
            self.fleet_api_type,
            source=resolve_run_start_source(run_source),
        )

    def StreamLogs(  # pylint: disable=C0103
        self, request: StreamLogsRequest, context: grpc.ServicerContext
    ) -> Generator[StreamLogsResponse, None, None]:
        """Get logs."""
        return control_handlers.stream_logs(
            request,
            _get_account(),
            self.linkstate_factory.state(),
            context.is_active,
        )

    def ListRuns(
        self, request: ListRunsRequest, context: grpc.ServicerContext
    ) -> ListRunsResponse:
        """Handle `flwr ls` command."""
        return control_handlers.list_runs(
            request,
            _get_account(),
            self.linkstate_factory.state(),
        )

    def ListRunSeries(
        self, request: ListRunSeriesRequest, context: grpc.ServicerContext
    ) -> ListRunSeriesResponse:
        """List run series."""
        return control_handlers.list_run_series(
            request, _get_account(), self.linkstate_factory.state()
        )

    def GetRunSeries(
        self, request: GetRunSeriesRequest, context: grpc.ServicerContext
    ) -> GetRunSeriesResponse:
        """Get run series."""
        return control_handlers.get_run_series(
            request, _get_account(), self.linkstate_factory.state()
        )

    def StopRun(
        self, request: StopRunRequest, context: grpc.ServicerContext
    ) -> StopRunResponse:
        """Stop a given run ID."""
        return control_handlers.stop_run(
            request, _get_account(), self.linkstate_factory.state()
        )

    def StartAutomation(
        self, request: StartAutomationRequest, context: grpc.ServicerContext
    ) -> StartAutomationResponse:
        """Start an automation."""
        return control_handlers.start_automation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def ListAutomations(
        self, request: ListAutomationsRequest, context: grpc.ServicerContext
    ) -> ListAutomationsResponse:
        """List automations."""
        return control_handlers.list_automations(
            request, _get_account(), self.linkstate_factory.state()
        )

    def StopAutomation(
        self, request: StopAutomationRequest, context: grpc.ServicerContext
    ) -> StopAutomationResponse:
        """Stop an automation."""
        return control_handlers.stop_automation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def GetLoginDetails(
        self, request: GetLoginDetailsRequest, context: grpc.ServicerContext
    ) -> GetLoginDetailsResponse:
        """Start login."""
        return control_handlers.get_login_details(request, self.authn_plugin)

    def GetAuthTokens(
        self, request: GetAuthTokensRequest, context: grpc.ServicerContext
    ) -> GetAuthTokensResponse:
        """Get auth token."""
        return control_handlers.get_auth_tokens(request, self.authn_plugin)

    def ListConnectors(
        self, request: ListConnectorsRequest, context: grpc.ServicerContext
    ) -> ListConnectorsResponse:
        """List OAuth connectors available to the authenticated account."""
        return control_handlers.list_connectors(
            request,
            _get_account(),
            self.linkstate_factory.state(),
        )

    def DisconnectConnector(
        self, request: DisconnectConnectorRequest, context: grpc.ServicerContext
    ) -> DisconnectConnectorResponse:
        """Disconnect connector credentials for the authenticated account."""
        return control_handlers.disconnect_connector(
            request,
            _get_account(),
            self.linkstate_factory.state(),
        )

    def BeginConnectorOAuth(
        self, request: BeginConnectorOAuthRequest, context: grpc.ServicerContext
    ) -> BeginConnectorOAuthResponse:
        """Begin OAuth connector authorization flow."""
        return control_handlers.begin_connector_oauth(
            request,
            _get_account(),
            self.linkstate_factory.state(),
        )

    def CompleteConnectorOAuth(
        self, request: CompleteConnectorOAuthRequest, context: grpc.ServicerContext
    ) -> CompleteConnectorOAuthResponse:
        """Complete OAuth connector authorization flow."""
        return control_handlers.complete_connector_oauth(
            request,
            _get_account(),
            self.linkstate_factory.state(),
        )

    def PullArtifacts(
        self, request: PullArtifactsRequest, context: grpc.ServicerContext
    ) -> PullArtifactsResponse:
        """Pull artifacts for a given run ID."""
        return control_handlers.pull_artifacts(
            request,
            _get_account(),
            self.linkstate_factory.state(),
            self.artifact_provider,
        )

    def RegisterNode(
        self, request: RegisterNodeRequest, context: grpc.ServicerContext
    ) -> RegisterNodeResponse:
        """Add a SuperNode."""
        return control_handlers.register_node(
            request, _get_account(), self.linkstate_factory.state()
        )

    def UnregisterNode(
        self, request: UnregisterNodeRequest, context: grpc.ServicerContext
    ) -> UnregisterNodeResponse:
        """Remove a SuperNode."""
        return control_handlers.unregister_node(
            request, _get_account(), self.linkstate_factory.state()
        )

    def ListNodes(
        self, request: ListNodesRequest, context: grpc.ServicerContext
    ) -> ListNodesResponse:
        """List all SuperNodes."""
        return control_handlers.list_nodes(
            request, _get_account(), self.linkstate_factory.state()
        )

    def ListFederations(
        self, request: ListFederationsRequest, context: grpc.ServicerContext
    ) -> ListFederationsResponse:
        """List all SuperNodes."""
        return control_handlers.list_federations(
            request, _get_account(), self.linkstate_factory.state()
        )

    def ListApps(
        self, request: ListAppsRequest, context: grpc.ServicerContext
    ) -> ListAppsResponse:
        """List apps in a federation."""
        return control_handlers.list_apps(
            request, _get_account(), self.linkstate_factory.state()
        )

    def AddApp(
        self, request: AddAppRequest, context: grpc.ServicerContext
    ) -> AddAppResponse:
        """Add an app to a federation."""
        return control_handlers.add_app(
            request,
            _get_account(),
            self.linkstate_factory.state(),
            self.fleet_api_type,
        )

    def RemoveApp(
        self, request: RemoveAppRequest, context: grpc.ServicerContext
    ) -> RemoveAppResponse:
        """Remove an app from a federation."""
        return control_handlers.remove_app(
            request, _get_account(), self.linkstate_factory.state()
        )

    def ShowFederation(
        self, request: ShowFederationRequest, context: grpc.ServicerContext
    ) -> ShowFederationResponse:
        """Show details of a specific Federation."""
        return control_handlers.show_federation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def CreateFederation(
        self, request: CreateFederationRequest, context: grpc.ServicerContext
    ) -> CreateFederationResponse:
        """Create a new Federation."""
        return control_handlers.create_federation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def ArchiveFederation(
        self, request: ArchiveFederationRequest, context: grpc.ServicerContext
    ) -> ArchiveFederationResponse:
        """Archive a Federation."""
        return control_handlers.archive_federation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def AddNodeToFederation(
        self, request: AddNodeToFederationRequest, context: grpc.ServicerContext
    ) -> AddNodeToFederationResponse:
        """Add a node to a Federation."""
        return control_handlers.add_node_to_federation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def RemoveNodeFromFederation(
        self, request: RemoveNodeFromFederationRequest, context: grpc.ServicerContext
    ) -> RemoveNodeFromFederationResponse:
        """Remove a node from a Federation."""
        return control_handlers.remove_node_from_federation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def RemoveAccountFromFederation(
        self, request: RemoveAccountFromFederationRequest, context: grpc.ServicerContext
    ) -> RemoveAccountFromFederationResponse:
        """Remove an account from a Federation."""
        return control_handlers.remove_account_from_federation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def CreateInvitation(
        self, request: CreateInvitationRequest, context: grpc.ServicerContext
    ) -> CreateInvitationResponse:
        """Create an invitation."""
        return control_handlers.create_invitation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def ListInvitations(
        self, request: ListInvitationsRequest, context: grpc.ServicerContext
    ) -> ListInvitationsResponse:
        """List invitations."""
        return control_handlers.list_invitations(
            request, _get_account(), self.linkstate_factory.state()
        )

    def AcceptInvitation(
        self, request: AcceptInvitationRequest, context: grpc.ServicerContext
    ) -> AcceptInvitationResponse:
        """Accept an invitation."""
        return control_handlers.accept_invitation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def RejectInvitation(
        self, request: RejectInvitationRequest, context: grpc.ServicerContext
    ) -> RejectInvitationResponse:
        """Reject an invitation."""
        return control_handlers.reject_invitation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def RevokeInvitation(
        self, request: RevokeInvitationRequest, context: grpc.ServicerContext
    ) -> RevokeInvitationResponse:
        """Revoke an invitation."""
        return control_handlers.revoke_invitation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def ConfigureSimulationFederation(
        self,
        request: ConfigureSimulationFederationRequest,
        context: grpc.ServicerContext,
    ) -> ConfigureSimulationFederationResponse:
        """Configure a federation for simulation."""
        return control_handlers.configure_simulation_federation(
            request, _get_account(), self.linkstate_factory.state()
        )

    def StreamRunEvents(
        self, request: StreamRunEventsRequest, context: grpc.ServicerContext
    ) -> Generator[StreamRunEventsResponse, None, None]:
        """Start run event stream."""
        return control_handlers.stream_run_events(
            request,
            _get_account(),
            self.linkstate_factory.state(),
            context.is_active,
        )

    def _resolve_federation_id(self, account_name: str, federation_id: str) -> str:
        """Return the requested federation ID or derive the default federation ID."""
        return _resolve_federation_id(
            self.linkstate_factory.state(), account_name, federation_id
        )


def _get_account() -> AccountInfo:
    """Guard clause to check if account information exists."""
    account = get_current_account_info()
    if not account.flwr_aid:
        raise FlowerError(
            ApiErrorCode.ACCOUNT_INFO_NOT_FOUND,
            "Failed to fetch the account information.",
        )
    return account
