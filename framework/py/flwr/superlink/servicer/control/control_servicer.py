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

# pylint: disable=too-many-lines

import hashlib
import json
import time
from collections.abc import Generator, Sequence
from logging import ERROR, INFO
from typing import Any, cast

import grpc
import requests

from flwr.agentapp.builtin import try_resolve_builtin_agent_fab
from flwr.cli.utils import validate_federation_name
from flwr.common.config import (
    flatten_dict,
    fuse_dicts,
    get_fab_config,
    get_metadata_from_config,
)
from flwr.common.constant import (
    FAB_MAX_SIZE,
    FEDERATION_NOT_FOUND_MESSAGE,
    HEARTBEAT_DEFAULT_INTERVAL,
    LOG_STREAM_INTERVAL,
    NO_ACCOUNT_AUTH_MESSAGE,
    NO_ARTIFACT_PROVIDER_MESSAGE,
    NODE_NOT_FOUND_MESSAGE,
    PUBLIC_KEY_ALREADY_IN_USE_MESSAGE,
    PUBLIC_KEY_NOT_VALID,
    PULL_UNFINISHED_RUN_MESSAGE,
    RUN_EVENTS_STREAM_INTERVAL,
    RUN_ID_NOT_FOUND_MESSAGE,
    TRANSPORT_TYPE_GRPC_ADAPTER,
    Status,
)
from flwr.common.logger import log
from flwr.common.serde import (
    context_to_proto,
    run_status_to_proto,
    run_to_proto,
    user_config_from_proto,
)
from flwr.proto import control_pb2_grpc  # pylint: disable=E0611
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
    StartRunRequest,
    StartRunResponse,
    StopRunRequest,
    StopRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
    StreamRunEventsRequest,
    StreamRunEventsResponse,
    UnregisterNodeRequest,
    UnregisterNodeResponse,
)
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.constant import (
    DEFAULT_FEDERATION_SIMULATION,
    NOOP_FEDERATION,
    PLATFORM_API_URL,
    ActionType,
    RunTime,
    TaskType,
)
from flwr.supercore.date import now
from flwr.supercore.error import ApiErrorCode, FlowerError, rpc_error_translator
from flwr.supercore.fab import Fab
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.primitives.asymmetric import bytes_to_public_key, uses_nist_ec_curve
from flwr.supercore.run import Run
from flwr.supercore.typing import (
    AcceptInvitationContext,
    CreateFederationContext,
    CreateInvitationContext,
    RegisterSupernodeContext,
    StartRunContext,
)
from flwr.supercore.utils import (
    parse_app_spec,
    request_download_link,
    resolve_account_ids,
)
from flwr.superlink.artifact_provider import ArtifactProvider
from flwr.superlink.auth_plugin import ControlAuthnPlugin
from flwr.superlink.federation.noop_federation_manager import NoOpFederationManager

from .control_account_auth_interceptor import get_current_account_info


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

    def StartRun(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        self, request: StartRunRequest, context: grpc.ServicerContext
    ) -> StartRunResponse:
        """Create run ID."""
        log(INFO, rpc_name := self.StartRun.__qualname__)
        state = self.linkstate_factory.state()

        verification_dict: dict[str, str] = {}
        note: str | None = None

        builtin_agent_fab = try_resolve_builtin_agent_fab(request.app_spec)
        if builtin_agent_fab is not None:
            fab_file, verification_dict = builtin_agent_fab
        elif request.app_spec:
            fab_file, verification_dict, note = _get_remote_fab(
                self.fleet_api_type, request.app_spec, context
            )
        else:
            fab_file = request.fab.content

        if len(fab_file) > FAB_MAX_SIZE:
            log(
                ERROR,
                "FAB size exceeds maximum allowed size of %d bytes.",
                FAB_MAX_SIZE,
            )
            return StartRunResponse()

        account = _get_account(context)
        flwr_aid = cast(str, account.flwr_aid)
        account_name = cast(str, account.account_name)
        override_config = user_config_from_proto(request.override_config)

        with rpc_error_translator(context, rpc_name):
            state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)

            # Check (1) federation exists and (2) the flwr_aid is a member
            federation = self._resolve_federation(account_name, request.federation)
            if not state.federation_manager.exists(federation):
                if request.federation:
                    raise FlowerError(
                        ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PERMISSION,
                        f"Federation '{federation}' not found or has been archived.",
                    )
                raise FlowerError(
                    ApiErrorCode.FEDERATION_NOT_SPECIFIED, "No federation specified."
                )

            if not state.federation_manager.has_member(flwr_aid, federation):
                raise FlowerError(
                    ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PERMISSION,
                    f"Account with ID '{flwr_aid}' is not a member of the "
                    f"federation '{federation}'.",
                )

        try:
            # Validate user config overrides matches keys in run config in FAB
            fab_config = get_fab_config(fab_file)
            run_config = flatten_dict(fab_config["tool"]["flwr"]["app"].get("config"))
            _ = fuse_dicts(run_config, override_config)

            # Derive primary task type from the submitted FAB. AgentApp-only FABs can
            # be bundled locally and submitted through the regular `flwr run` path.
            components = fab_config["tool"]["flwr"]["app"].get("components", {})
            is_agentapp_bundle = "agentapp" in components
            primary_task_type = (
                TaskType.AGENT_APP if is_agentapp_bundle else TaskType.SERVER_APP
            )
            resolved_federation_config = None
            runtime = RunTime.DEPLOYMENT
            with rpc_error_translator(context, rpc_name):
                sim_cfg = state.federation_manager.get_simulation_config(federation)
                if sim_cfg and not is_agentapp_bundle:
                    primary_task_type = TaskType.SIMULATION
                    runtime = RunTime.SIMULATION
                    resolved_federation_config = SimulationConfig()
                    resolved_federation_config.CopyFrom(sim_cfg)
                    resolved_federation_config.MergeFrom(
                        request.override_federation_config
                    )

                state.federation_manager.can_execute(
                    flwr_aid,
                    ActionType.START_RUN,
                    StartRunContext(federation_name=federation, runtime=runtime),
                )

            # Create run
            fab = Fab(
                hashlib.sha256(fab_file).hexdigest(),
                fab_file,
                verification_dict,
            )
            fab_hash = state.store_fab(fab)

            if fab_hash != fab.hash_str:
                raise ValueError(
                    f"FAB ({fab.hash_str}) hash from request doesn't match contents"
                )
            fab_id, fab_version = get_metadata_from_config(fab_config)

            run_id = state.create_run(
                fab_id,
                fab_version,
                fab_hash,
                override_config,
                federation,
                resolved_federation_config,
                flwr_aid,
                primary_task_type,
                request.series_id if request.HasField("series_id") else None,
            )

            if run_id == 0:
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    "Failed to create or initialize the run.",
                )

            run = state.get_run_info(run_ids=[run_id])[0]
            series_id = run.series_id

        except ValueError as e:
            log(ERROR, "Could not start run: %s", str(e))
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))

        log_msg = f"Created run {run_id} in federation {run.federation}"
        log(INFO, log_msg)
        return StartRunResponse(
            run_id=run_id, note=note, series_id=series_id, federation=run.federation
        )

    def StreamLogs(  # pylint: disable=C0103
        self, request: StreamLogsRequest, context: grpc.ServicerContext
    ) -> Generator[StreamLogsResponse, Any, None]:
        """Get logs."""
        log(INFO, rpc_name := self.StreamLogs.__qualname__)

        # Init link state
        state = self.linkstate_factory.state()

        # Retrieve run ID and run
        run_id = request.run_id
        runs = state.get_run_info(run_ids=[run_id])

        # Exit if `run_id` not found
        if not runs:
            context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)
        run = runs[0]
        task_id = cast(int, run.primary_task_id)

        with rpc_error_translator(context, rpc_name):
            flwr_aid = _get_flwr_aid(context)
            _validate_federation_membership_in_request(
                state, flwr_aid, run.federation, context
            )

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

                # Delete objects of the run from the object store
                self.objectstore_factory.store().delete_objects_in_run(run_id)
                break

            time.sleep(LOG_STREAM_INTERVAL)  # Sleep briefly to avoid busy waiting

    def ListRuns(
        self, request: ListRunsRequest, context: grpc.ServicerContext
    ) -> ListRunsResponse:
        """Handle `flwr ls` command."""
        log(INFO, rpc_name := self.ListRuns.__qualname__)

        # Init link state
        state = self.linkstate_factory.state()

        account = _get_account(context)
        flwr_aid = cast(str, account.flwr_aid)
        account_name = cast(str, account.account_name)
        # Build a set of run IDs for `flwr ls --runs`
        if not request.HasField("run_id"):
            # If no `run_id` is specified and account auth is enabled,
            # return run IDs for the authenticated account
            limit = request.limit if request.HasField("limit") else None
            runs = state.get_run_info(
                flwr_aids=[flwr_aid],
                order_by="pending_at",
                ascending=False,
                limit=limit,
            )
        # Build a set of run IDs for `flwr ls --run-id <run_id>`
        else:
            # Retrieve run ID and run
            runs = state.get_run_info(run_ids=[request.run_id])

            # Exit if `run_id` not found
            if not runs:
                context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)
                raise grpc.RpcError()  # This line is unreachable

            # Check if requester is a member of the federation
            # that the run belongs to
            with rpc_error_translator(context, rpc_name):
                _validate_federation_membership_in_request(
                    state, flwr_aid, runs[0].federation, context
                )

        # Clear objects of finished runs
        store = self.objectstore_factory.store()
        # Resolve only non-caller run owners; caller-owned runs use `account_name`.
        account_names = resolve_account_ids(
            {run.flwr_aid for run in runs if run.flwr_aid != flwr_aid}
        )
        account_names[flwr_aid] = account_name
        for run in runs:
            run.account_name = account_names[run.flwr_aid]
            if run.status.status == Status.FINISHED:
                store.delete_objects_in_run(run.run_id)

        # Construct and return response
        return ListRunsResponse(
            run_dict={run.run_id: run_to_proto(run) for run in runs},
            now=now().isoformat(),
        )

    def ListRunSeries(
        self, request: ListRunSeriesRequest, context: grpc.ServicerContext
    ) -> ListRunSeriesResponse:
        """List run series."""
        log(INFO, rpc_name := self.ListRunSeries.__qualname__)

        state = self.linkstate_factory.state()
        flwr_aid = _get_flwr_aid(context)
        updated_before = (
            request.updated_before if request.HasField("updated_before") else None
        )
        limit = request.limit if request.HasField("limit") else None
        federation_id = (
            request.federation_id if request.HasField("federation_id") else None
        )

        with rpc_error_translator(context, rpc_name):
            if federation_id is not None:
                _validate_federation_membership_in_request(
                    state, flwr_aid, federation_id, context
                )
                federation_ids = [federation_id]
            else:
                federations = state.federation_manager.get_federations(flwr_aid)
                federation_ids = [federation.name for federation in federations]
            entries = state.get_run_series(
                federations=federation_ids,
                updated_before=updated_before,
                limit=limit,
            )

        return ListRunSeriesResponse(entries=_with_last_run_statuses(state, entries))

    def GetRunSeries(
        self, request: GetRunSeriesRequest, context: grpc.ServicerContext
    ) -> GetRunSeriesResponse:
        """Get run series."""
        log(INFO, self.GetRunSeries.__qualname__)

        state = self.linkstate_factory.state()
        flwr_aid = _get_flwr_aid(context)
        with rpc_error_translator(context, self.GetRunSeries.__qualname__):
            series_matches = state.get_run_series(series_ids=[request.series_id])

            # The caller must be a member of the federation
            if not series_matches or not state.federation_manager.has_member(
                flwr_aid, series_matches[0].federation
            ):
                context.abort(grpc.StatusCode.NOT_FOUND, "Run series ID not found.")
                raise grpc.RpcError()  # This line is unreachable

        # Get the run series context and construct the response
        # Run series context is created atomically by LinkState.create_run(...)
        # and should never be None.
        series_context = state.get_run_series_context(request.series_id)
        response = GetRunSeriesResponse(
            series=_with_last_run_statuses(state, series_matches)[0],
            context=context_to_proto(series_context) if series_context else None,
        )
        return response

    def StopRun(
        self, request: StopRunRequest, context: grpc.ServicerContext
    ) -> StopRunResponse:
        """Stop a given run ID."""
        log(INFO, rpc_name := self.StopRun.__qualname__)

        # Init link state
        state = self.linkstate_factory.state()

        # Retrieve run ID and run
        run_id = request.run_id
        runs = state.get_run_info(run_ids=[run_id])

        # Exit if `run_id` not found
        if not runs:
            context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)
            raise grpc.RpcError()  # This line is unreachable
        run = runs[0]

        with rpc_error_translator(context, rpc_name):
            flwr_aid = _get_flwr_aid(context)
            _validate_federation_membership_in_request(
                state, flwr_aid, run.federation, context
            )

        if run.status.status == Status.FINISHED:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"Run ID {run_id} is already finished",
            )

        return StopRunResponse(success=state.stop_run(run_id))

    def GetLoginDetails(
        self, request: GetLoginDetailsRequest, context: grpc.ServicerContext
    ) -> GetLoginDetailsResponse:
        """Start login."""
        log(INFO, "ControlServicer.GetLoginDetails")
        if self.authn_plugin is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                NO_ACCOUNT_AUTH_MESSAGE,
            )
            raise grpc.RpcError()  # This line is unreachable

        # Get login details
        details = self.authn_plugin.get_login_details()

        # Return empty response if details is None
        if details is None:
            return GetLoginDetailsResponse()

        return GetLoginDetailsResponse(
            authn_type=details.authn_type,
            device_code=details.device_code,
            verification_uri_complete=details.verification_uri_complete,
            expires_in=details.expires_in,
            interval=details.interval,
        )

    def GetAuthTokens(
        self, request: GetAuthTokensRequest, context: grpc.ServicerContext
    ) -> GetAuthTokensResponse:
        """Get auth token."""
        log(INFO, "ControlServicer.GetAuthTokens")
        if self.authn_plugin is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                NO_ACCOUNT_AUTH_MESSAGE,
            )
            raise grpc.RpcError()  # This line is unreachable

        # Get auth tokens
        credentials = self.authn_plugin.get_auth_tokens(request.device_code)

        # Return empty response if credentials is None
        if credentials is None:
            return GetAuthTokensResponse()

        return GetAuthTokensResponse(
            access_token=credentials.access_token,
            refresh_token=credentials.refresh_token,
        )

    def PullArtifacts(
        self, request: PullArtifactsRequest, context: grpc.ServicerContext
    ) -> PullArtifactsResponse:
        """Pull artifacts for a given run ID."""
        log(INFO, "ControlServicer.PullArtifacts")

        # Check if artifact provider is configured
        if self.artifact_provider is None:
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                NO_ARTIFACT_PROVIDER_MESSAGE,
            )
            raise grpc.RpcError()  # This line is unreachable

        # Init link state
        state = self.linkstate_factory.state()

        # Retrieve run ID and run
        run_id = request.run_id
        runs = state.get_run_info(run_ids=[run_id])

        # Exit if `run_id` not found
        if not runs:
            context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)
            raise grpc.RpcError()  # This line is unreachable
        run = runs[0]

        # Exit if the run is not finished yet
        if run.status.status != Status.FINISHED:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, PULL_UNFINISHED_RUN_MESSAGE
            )

        # Check if `flwr_aid` matches the run's `flwr_aid`
        flwr_aid = get_current_account_info().flwr_aid
        _check_flwr_aid_in_run(flwr_aid=flwr_aid, run=run, context=context)

        # Call artifact provider
        download_url = self.artifact_provider.get_url(run_id)
        return PullArtifactsResponse(url=download_url)

    def RegisterNode(
        self, request: RegisterNodeRequest, context: grpc.ServicerContext
    ) -> RegisterNodeResponse:
        """Add a SuperNode."""
        log(INFO, "ControlServicer.RegisterNode")

        # Verify public key
        try:
            # Attempt to deserialize public key
            pub_key = bytes_to_public_key(request.public_key)
            # Check if it's a NIST EC curve public key
            if not uses_nist_ec_curve(pub_key):
                err_msg = "The provided public key is not a NIST EC curve public key."
                log(ERROR, "%s", err_msg)
                raise ValueError(err_msg)
        except (ValueError, AttributeError) as err:
            log(ERROR, "%s", err)
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, PUBLIC_KEY_NOT_VALID)

        # Init link state
        state = self.linkstate_factory.state()
        node_id = 0

        flwr_aid = _get_flwr_aid(context)
        with rpc_error_translator(context, self.RegisterNode.__qualname__):
            state.federation_manager.can_execute(
                flwr_aid,
                ActionType.REGISTER_SUPERNODE,
                RegisterSupernodeContext(),
            )

        # Account name exists if `flwr_aid` exists
        account_name = cast(str, get_current_account_info().account_name)
        try:
            node_id = state.create_node(
                owner_aid=flwr_aid,
                owner_name=account_name,
                public_key=request.public_key,
                heartbeat_interval=HEARTBEAT_DEFAULT_INTERVAL,
            )

        except ValueError:
            # Public key already in use
            log(ERROR, PUBLIC_KEY_ALREADY_IN_USE_MESSAGE)
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, PUBLIC_KEY_ALREADY_IN_USE_MESSAGE
            )
        log(INFO, "[ControlServicer.RegisterNode] Created node_id=%s", node_id)

        return RegisterNodeResponse(node_id=node_id)

    def UnregisterNode(
        self, request: UnregisterNodeRequest, context: grpc.ServicerContext
    ) -> UnregisterNodeResponse:
        """Remove a SuperNode."""
        log(INFO, "ControlServicer.UnregisterNode")

        # Init link state
        state = self.linkstate_factory.state()

        flwr_aid = _get_flwr_aid(context)
        try:
            state.delete_node(owner_aid=flwr_aid, node_id=request.node_id)
        except ValueError:
            log(ERROR, NODE_NOT_FOUND_MESSAGE)
            context.abort(grpc.StatusCode.NOT_FOUND, NODE_NOT_FOUND_MESSAGE)

        return UnregisterNodeResponse()

    def ListNodes(
        self, request: ListNodesRequest, context: grpc.ServicerContext
    ) -> ListNodesResponse:
        """List all SuperNodes."""
        log(INFO, "ControlServicer.ListNodes")

        nodes_info: Sequence[NodeInfo] = []
        # Init link state
        state = self.linkstate_factory.state()

        # Retrieve all nodes for the account
        nodes_info = state.get_node_info(owner_aids=[_get_flwr_aid(context)])

        return ListNodesResponse(nodes_info=nodes_info, now=now().isoformat())

    def ListFederations(
        self, request: ListFederationsRequest, context: grpc.ServicerContext
    ) -> ListFederationsResponse:
        """List all SuperNodes."""
        log(INFO, rpc_name := self.ListFederations.__qualname__)

        # Init link state
        state = self.linkstate_factory.state()
        flwr_aid = _get_flwr_aid(context)

        # Get federations the account is a member of
        with rpc_error_translator(context, rpc_name):
            state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)
            federations = state.federation_manager.get_federations(flwr_aid)

        return ListFederationsResponse(
            federations=[
                Federation(
                    name=fed.name,
                    description=fed.description,
                    archived=fed.archived,
                    simulation=fed.simulation,
                )
                for fed in federations
            ]
        )

    def ShowFederation(
        self, request: ShowFederationRequest, context: grpc.ServicerContext
    ) -> ShowFederationResponse:
        """Show details of a specific Federation."""
        log(INFO, rpc_name := self.ShowFederation.__qualname__)

        # Init link state
        state = self.linkstate_factory.state()

        # Ensure flwr_aid is a member of the requested federation
        federation = request.federation_name
        flwr_aid = _get_flwr_aid(context)
        with rpc_error_translator(context, rpc_name):
            state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)
            if not state.federation_manager.has_member(flwr_aid, federation):
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"Federation '{federation}' does not exist or you are "
                    "not a member of it.",
                )

            # Fetch federation details
            details = state.federation_manager.get_details(federation)

        # Build Federation proto object
        federation_proto = Federation(
            name=federation,
            description=details.description,
            members=details.members,
            nodes=details.nodes,
            runs=[run_to_proto(run) for run in details.runs],
            archived=details.archived,
            simulation=details.simulation,
            config=details.config,
        )
        return ShowFederationResponse(
            federation=federation_proto, now=now().isoformat()
        )

    def CreateFederation(
        self, request: CreateFederationRequest, context: grpc.ServicerContext
    ) -> CreateFederationResponse:
        """Create a new Federation."""
        log(INFO, rpc_name := self.CreateFederation.__qualname__)

        with rpc_error_translator(context, rpc_name):
            # Check that a federation is specified
            if not request.federation_name:
                raise FederationNotSpecified()

            # Ensure valid federation name is provided
            success, err_msg = validate_federation_name(request.federation_name)
            if not success:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    f"Invalid federation name: '{request.federation_name}'. {err_msg}",
                )

            # Init link state
            state = self.linkstate_factory.state()

            # Construct federation name
            account = _get_account(context)
            flwr_aid = cast(str, account.flwr_aid)
            state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)
            federation_name = f"@{account.account_name}/{request.federation_name}"

            runtime = RunTime.SIMULATION if request.simulation else RunTime.DEPLOYMENT
            state.federation_manager.can_execute(
                flwr_aid,
                ActionType.CREATE_FEDERATION,
                CreateFederationContext(
                    federation_name=federation_name,
                    runtime=runtime,
                    visibility="private",
                ),
            )

            # Create federation
            federation = state.federation_manager.create_federation(
                name=federation_name,
                description=request.description,
                flwr_aid=flwr_aid,
                simulation=request.simulation,
            )

        return CreateFederationResponse(
            federation=Federation(
                name=federation.name,
                description=federation.description,
                members=federation.members,
                simulation=federation.simulation,
            )
        )

    def ArchiveFederation(
        self, request: ArchiveFederationRequest, context: grpc.ServicerContext
    ) -> ArchiveFederationResponse:
        """Archive a Federation."""
        log(INFO, rpc_name := self.ArchiveFederation.__qualname__)

        with rpc_error_translator(context, rpc_name):
            # Check that a federation is specified
            if not request.federation_name:
                raise FederationNotSpecified()

            # Init link state
            state = self.linkstate_factory.state()

            # Archive federation
            state.federation_manager.archive_federation(
                flwr_aid=_get_flwr_aid(context),
                name=request.federation_name,
            )
            for run in state.get_run_info(federations=[request.federation_name]):
                if run.status.status != Status.FINISHED:
                    state.stop_run(run.run_id)

        return ArchiveFederationResponse()

    def AddNodeToFederation(
        self, request: AddNodeToFederationRequest, context: grpc.ServicerContext
    ) -> AddNodeToFederationResponse:
        """Add a node to a Federation."""
        log(INFO, rpc_name := self.AddNodeToFederation.__qualname__)

        # Init link state
        state = self.linkstate_factory.state()

        with rpc_error_translator(context, rpc_name):
            # Validate federation, node ID, and ownership
            flwr_aid = _get_flwr_aid(context)
            state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)
            _validate_federation_and_node_in_request(
                state, flwr_aid, request.federation_name, request.node_id, context
            )

            # Add node to the federation
            state.federation_manager.add_supernode(
                flwr_aid=flwr_aid,
                federation=request.federation_name,
                node_id=request.node_id,
            )

        return AddNodeToFederationResponse()

    def RemoveNodeFromFederation(
        self, request: RemoveNodeFromFederationRequest, context: grpc.ServicerContext
    ) -> RemoveNodeFromFederationResponse:
        """Remove a node from a Federation."""
        log(INFO, rpc_name := self.RemoveNodeFromFederation.__qualname__)

        # Init link state
        state = self.linkstate_factory.state()

        with rpc_error_translator(context, rpc_name):
            # Validate federation, node ID, and ownership
            flwr_aid = _get_flwr_aid(context)
            _validate_federation_and_node_in_request(
                state, flwr_aid, request.federation_name, request.node_id, context
            )

            # Remove node from the federation
            state.federation_manager.remove_supernode(
                flwr_aid=flwr_aid,
                federation=request.federation_name,
                node_id=request.node_id,
            )

        return RemoveNodeFromFederationResponse()

    def RemoveAccountFromFederation(
        self, request: RemoveAccountFromFederationRequest, context: grpc.ServicerContext
    ) -> RemoveAccountFromFederationResponse:
        """Remove an account from a Federation."""
        log(INFO, rpc_name := self.RemoveAccountFromFederation.__qualname__)

        state = self.linkstate_factory.state()

        target_account = None if not request.account_name else request.account_name

        with rpc_error_translator(context, rpc_name):
            removed_flwr_aid = state.federation_manager.remove_account(
                flwr_aid=_get_flwr_aid(context),
                federation=request.federation_name,
                target_account_name=target_account,
            )
            # Get runs from account that was removed
            # and stop them.
            for run in state.get_run_info(
                federations=[request.federation_name],
                flwr_aids=[removed_flwr_aid],
                statuses=[Status.PENDING, Status.STARTING, Status.RUNNING],
            ):
                state.stop_run(run.run_id)
        return RemoveAccountFromFederationResponse()

    def CreateInvitation(
        self, request: CreateInvitationRequest, context: grpc.ServicerContext
    ) -> CreateInvitationResponse:
        """Create an invitation."""
        log(INFO, rpc_name := self.CreateInvitation.__qualname__)

        state = self.linkstate_factory.state()

        with rpc_error_translator(context, rpc_name):
            flwr_aid = _get_flwr_aid(context)
            state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)
            federation = request.federation_name
            invitee_account_name = request.invitee_account_name

            runtime = (
                RunTime.SIMULATION
                if state.federation_manager.get_simulation_config(federation)
                else RunTime.DEPLOYMENT
            )

            state.federation_manager.can_execute(
                flwr_aid=flwr_aid,
                action=ActionType.CREATE_INVITATION,
                context=CreateInvitationContext(
                    federation_name=federation,
                    invitee_account_name=invitee_account_name,
                    runtime=runtime,
                ),
            )

            state.federation_manager.create_invitation(
                flwr_aid=flwr_aid,
                federation=federation,
                invitee_account_name=invitee_account_name,
            )
        return CreateInvitationResponse()

    def ListInvitations(
        self, request: ListInvitationsRequest, context: grpc.ServicerContext
    ) -> ListInvitationsResponse:
        """List invitations."""
        log(INFO, rpc_name := self.ListInvitations.__qualname__)

        state = self.linkstate_factory.state()

        with rpc_error_translator(context, rpc_name):
            created_invitations, received_invitations = (
                state.federation_manager.list_invitations(_get_flwr_aid(context))
            )
        return ListInvitationsResponse(
            created_invitations=created_invitations,
            received_invitations=received_invitations,
        )

    def AcceptInvitation(
        self, request: AcceptInvitationRequest, context: grpc.ServicerContext
    ) -> AcceptInvitationResponse:
        """Accept an invitation."""
        log(INFO, rpc_name := self.AcceptInvitation.__qualname__)

        state = self.linkstate_factory.state()

        with rpc_error_translator(context, rpc_name):
            flwr_aid = _get_flwr_aid(context)
            federation = request.federation_name

            runtime = (
                RunTime.SIMULATION
                if state.federation_manager.get_simulation_config(federation)
                else RunTime.DEPLOYMENT
            )

            state.federation_manager.can_execute(
                flwr_aid=flwr_aid,
                action=ActionType.ACCEPT_INVITATION,
                context=AcceptInvitationContext(
                    federation_name=federation,
                    runtime=runtime,
                ),
            )

            state.federation_manager.accept_invitation(
                flwr_aid=_get_flwr_aid(context),
                federation=request.federation_name,
            )
        return AcceptInvitationResponse()

    def RejectInvitation(
        self, request: RejectInvitationRequest, context: grpc.ServicerContext
    ) -> RejectInvitationResponse:
        """Reject an invitation."""
        log(INFO, rpc_name := self.RejectInvitation.__qualname__)

        state = self.linkstate_factory.state()

        with rpc_error_translator(context, rpc_name):
            state.federation_manager.reject_invitation(
                flwr_aid=_get_flwr_aid(context),
                federation=request.federation_name,
            )
        return RejectInvitationResponse()

    def RevokeInvitation(
        self, request: RevokeInvitationRequest, context: grpc.ServicerContext
    ) -> RevokeInvitationResponse:
        """Revoke an invitation."""
        log(INFO, rpc_name := self.RevokeInvitation.__qualname__)

        state = self.linkstate_factory.state()

        with rpc_error_translator(context, rpc_name):
            state.federation_manager.revoke_invitation(
                flwr_aid=_get_flwr_aid(context),
                federation=request.federation_name,
                invitee_account_name=request.invitee_account_name,
            )
        return RevokeInvitationResponse()

    def ConfigureSimulationFederation(
        self,
        request: ConfigureSimulationFederationRequest,
        context: grpc.ServicerContext,
    ) -> ConfigureSimulationFederationResponse:
        """Configure a federation for simulation."""
        log(INFO, rpc_name := self.ConfigureSimulationFederation.__qualname__)

        state = self.linkstate_factory.state()

        # Get caller's account info
        account = _get_account(context)
        flwr_aid = cast(str, account.flwr_aid)
        account_name = cast(str, account.account_name)

        with rpc_error_translator(context, rpc_name):
            state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)
            federation = self._resolve_federation(account_name, request.federation_name)
            if not state.federation_manager.exists(federation):
                if request.federation_name:
                    raise FlowerError(
                        ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PERMISSION,
                        f"Federation '{federation}' not found or has been archived.",
                    )
                raise FlowerError(
                    ApiErrorCode.FEDERATION_NOT_SPECIFIED, "No federation specified."
                )
            state.federation_manager.set_simulation_config(
                flwr_aid=flwr_aid,
                federation=federation,
                config=request.config,
            )

        return ConfigureSimulationFederationResponse(federation_name=federation)

    def StreamRunEvents(
        self, request: StreamRunEventsRequest, context: grpc.ServicerContext
    ) -> Generator[StreamRunEventsResponse, Any, None]:
        """Start run event stream."""
        log(INFO, rpc_name := self.StreamRunEvents.__qualname__)

        # Init link state
        state = self.linkstate_factory.state()

        # Retrieve run ID and run
        run_id = request.run_id
        runs = state.get_run_info(run_ids=[run_id])

        # Exit if `run_id` not found
        if not runs:
            context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)
            raise grpc.RpcError()  # This line is unreachable
        run = runs[0]

        with rpc_error_translator(context, rpc_name):
            flwr_aid = _get_flwr_aid(context)
            _validate_federation_membership_in_request(
                state, flwr_aid, run.federation, context
            )

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

    def _resolve_federation(self, account_name: str, federation: str) -> str:
        """Return the requested federation or derive the default federation."""
        if not federation:
            federation_manager = self.linkstate_factory.federation_manager
            if isinstance(federation_manager, NoOpFederationManager):
                federation = NOOP_FEDERATION
            else:
                federation = f"@{account_name}/{DEFAULT_FEDERATION_SIMULATION}"
        return federation


class FederationNotSpecified(FlowerError):
    """Exception raised when a federation is not specified in a request that requires
    one."""

    def __init__(self) -> None:
        super().__init__(
            ApiErrorCode.FEDERATION_NOT_SPECIFIED, "No federation specified in request."
        )


def _validate_federation_and_node_in_request(
    state: LinkState,
    flwr_aid: str,
    federation_name: str,
    node_id: int,
    context: grpc.ServicerContext,
) -> None:
    """Validate federation membership and node ownership for federation updates."""
    _validate_federation_membership_in_request(
        state, flwr_aid, federation_name, context
    )
    nodes_info = state.get_node_info(node_ids=[node_id])
    if not nodes_info or nodes_info[0].owner_aid != flwr_aid:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"Node {node_id} not found or you are not its owner.",
        )


def _validate_federation_membership_in_request(
    state: LinkState,
    flwr_aid: str,
    federation_name: str,
    context: grpc.ServicerContext,
) -> None:
    """Validate that a federation exists and the requester is one of its members."""
    if not federation_name:
        raise FederationNotSpecified()

    # Check that the federation exists
    if not state.federation_manager.exists(federation_name):
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            FEDERATION_NOT_FOUND_MESSAGE % federation_name,
        )

    # Check that the requester is a member of the federation
    if not state.federation_manager.has_member(flwr_aid, federation_name):
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            FEDERATION_NOT_FOUND_MESSAGE % federation_name,
        )


def _with_last_run_statuses(
    state: LinkState, run_series: Sequence[RunSeries]
) -> list[RunSeries]:
    """Return RunSeries with last_run_status populated from run state."""
    last_run_ids = {entry.run_ids[-1] for entry in run_series if entry.run_ids}
    run_statuses = state.get_run_status(last_run_ids)

    result = []
    for entry in run_series:
        if entry.run_ids:
            last_run_id = entry.run_ids[-1]
            if (run_status := run_statuses.get(last_run_id)) is not None:
                entry.last_run_status.CopyFrom(run_status_to_proto(run_status))
        result.append(entry)
    return result


def _get_account(context: grpc.ServicerContext) -> AccountInfo:
    """Guard clause to check if account information exists."""
    account = get_current_account_info()
    if account.flwr_aid is None:
        context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            "️⛔️ Failed to fetch the account information.",
        )
        raise RuntimeError  # This line is unreachable
    return account


def _get_flwr_aid(context: grpc.ServicerContext) -> str:
    """Guard clause to check if `flwr_aid` exists."""
    return cast(str, _get_account(context).flwr_aid)


def _check_flwr_aid_in_run(
    flwr_aid: str | None, run: Run, context: grpc.ServicerContext
) -> None:
    """Guard clause to check if `flwr_aid` matches the run's `flwr_aid`."""
    # `run.flwr_aid` must not be an empty string. Abort if it is empty.
    run_flwr_aid = run.flwr_aid
    if not run_flwr_aid:
        context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            "⛔️ Run is not associated with a `flwr_aid`.",
        )

    # Exit if `flwr_aid` does not match the run's `flwr_aid`
    if run_flwr_aid != flwr_aid:
        context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            "⛔️ Run ID does not belong to the account",
        )


def _format_verification(verifications: list[dict[str, str]]) -> dict[str, str]:
    """Format verification information for FAB."""
    # Convert verifications to dict[str, str] type
    verification_dict = {
        item["public_key_id"]: json.dumps(
            {k: v for k, v in item.items() if k != "public_key_id"}
        )
        for item in verifications
    }
    verification_dict.update({"valid_license": "Valid"})

    return verification_dict


def _get_remote_fab(
    fleet_api_type: str | None,
    app_spec: str,
    context: grpc.ServicerContext,
) -> tuple[bytes, dict[str, str], str | None]:
    """Get remote FAB from Flower Hub."""
    if fleet_api_type == TRANSPORT_TYPE_GRPC_ADAPTER:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            "The selected SuperLink transport type is not "
            "supported for connecting to Flower Hub.",
        )

    # Parse and validate app specification
    try:
        app_id, app_version = parse_app_spec(app_spec)
    except ValueError as e:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"{e}",
        )

    # Request download link and verification information
    url = f"{PLATFORM_API_URL}/hub/fetch-fab"
    try:
        presigned_url, verifications, note = request_download_link(
            app_id, app_version, url, "fab_url"
        )
    except ValueError as e:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"{e}",
        )

    # Format verification information
    verification_dict = (
        _format_verification(verifications)
        if verifications is not None
        else {"valid_license": ""}
    )

    # Download FAB from Flower Hub
    try:
        r = requests.get(presigned_url, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"FAB download failed: {str(e)}",
        )
    fab_file = r.content
    return fab_file, verification_dict, note
