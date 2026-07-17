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

import time
from collections.abc import Generator
from logging import INFO
from typing import Any, NoReturn, cast

import grpc

from flwr.common.constant import LOG_STREAM_INTERVAL, RUN_EVENTS_STREAM_INTERVAL, Status
from flwr.common.logger import log
from flwr.proto import control_pb2_grpc  # pylint: disable=E0611
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AcceptInvitationRequest,
    AcceptInvitationResponse,
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
from flwr.superlink.artifact_provider import ArtifactProvider
from flwr.superlink.auth_plugin import ControlAuthnPlugin

from . import control_handlers
from .control_account_auth_interceptor import get_current_account_info
from .control_handlers import (
    _resolve_federation_id,
    _validate_federation_membership_in_request,
)


def _abort_automations_unimplemented(context: grpc.ServicerContext) -> NoReturn:
    """Abort an automation RPC that has no implementation yet."""
    context.abort(
        grpc.StatusCode.UNIMPLEMENTED,
        "Automations are not implemented.",
    )
    raise NotImplementedError("Automations are not implemented.")


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
        return control_handlers.start_run(
            request, _get_account(), self.linkstate_factory.state(), self.fleet_api_type
        )

    def StreamLogs(  # pylint: disable=C0103
        self, request: StreamLogsRequest, context: grpc.ServicerContext
    ) -> Generator[StreamLogsResponse, Any, None]:
        """Get logs."""
        log(INFO, self.StreamLogs.__qualname__)

        # Init link state
        state = self.linkstate_factory.state()

        # Retrieve run ID and run
        run_id = request.run_id
        runs = state.get_run_info(run_ids=[run_id])

        # Exit if `run_id` not found
        if not runs:
            raise FlowerError(
                ApiErrorCode.RUN_ID_NOT_FOUND,
                f"Run {run_id} not found while streaming logs.",
            )
        run = runs[0]
        task_id = cast(int, run.primary_task_id)

        account = _get_account()
        flwr_aid = account.flwr_aid
        _validate_federation_membership_in_request(state, flwr_aid, run.federation_id)

        after_timestamp = request.after_timestamp + 1e-6
        while context.is_active():
            log_msg, latest_timestamp = state.get_task_log(task_id, after_timestamp)
            if log_msg:
                yield StreamLogsResponse(
                    log_output=log_msg,
                    latest_timestamp=latest_timestamp,
                )
                # Add a small epsilon to the latest timestamp to avoid getting
                # the same log
                after_timestamp = max(latest_timestamp + 1e-6, after_timestamp)

            # Wait for and continue to yield more log responses only if the
            # run isn't completed yet. If the run is finished, the entire log
            # is returned at this point and the server ends the stream.
            run = state.get_run_info(run_ids=[run_id])[0]
            if run.status.status == Status.FINISHED:
                log(INFO, "All logs for run ID `%s` returned", run_id)

                state.cleanup_run(run_id)
                break

            time.sleep(LOG_STREAM_INTERVAL)  # Sleep briefly to avoid busy waiting

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
        _abort_automations_unimplemented(context)

    def ListAutomations(
        self, request: ListAutomationsRequest, context: grpc.ServicerContext
    ) -> ListAutomationsResponse:
        """List automations."""
        _abort_automations_unimplemented(context)

    def StopAutomation(
        self, request: StopAutomationRequest, context: grpc.ServicerContext
    ) -> StopAutomationResponse:
        """Stop an automation."""
        _abort_automations_unimplemented(context)

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
    ) -> Generator[StreamRunEventsResponse, Any, None]:
        """Start run event stream."""
        log(INFO, self.StreamRunEvents.__qualname__)

        # Init link state
        state = self.linkstate_factory.state()

        # Retrieve run ID and run
        run_id = request.run_id
        runs = state.get_run_info(run_ids=[run_id])

        # Exit if `run_id` not found
        if not runs:
            raise FlowerError(
                ApiErrorCode.RUN_ID_NOT_FOUND,
                f"Run {run_id} not found while streaming run events.",
            )
        run = runs[0]

        account = _get_account()
        flwr_aid = account.flwr_aid
        _validate_federation_membership_in_request(state, flwr_aid, run.federation_id)

        after_task_event_id = None
        if request.HasField("after_task_event_id"):
            after_task_event_id = request.after_task_event_id
        while context.is_active():
            should_break = run.status.status == Status.FINISHED

            # Retrieve and yield all task events generated after the latest
            # streamed task event
            events = state.get_task_events(
                run_id=run_id,
                after_task_event_id=after_task_event_id,
            )
            for event in events:
                after_task_event_id = event.id
                yield StreamRunEventsResponse(task_event=event)

            # If the run was already finished before fetching this batch, all
            # events are returned at this point and the server ends the stream.
            if should_break:
                log(INFO, "All events for run ID `%s` returned", run_id)
                break

            # Refresh status after yielding. If streaming this batch raced with
            # run completion, continue immediately and fetch one final batch.
            run = state.get_run_info(run_ids=[run_id])[0]
            if run.status.status == Status.FINISHED:
                continue

            # Sleep briefly to avoid busy waiting
            time.sleep(RUN_EVENTS_STREAM_INTERVAL)

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
