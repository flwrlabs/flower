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
"""Control API functions."""

# pylint: disable=too-many-lines

import hashlib
import json
from collections.abc import Sequence
from logging import ERROR, INFO

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
    HEARTBEAT_DEFAULT_INTERVAL,
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
    UnregisterNodeRequest,
    UnregisterNodeResponse,
)
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.constant import (
    DEFAULT_FEDERATION_SIMULATION,
    NOOP_FEDERATION_ID,
    PLATFORM_API_URL,
    ActionType,
    RunTime,
    TaskType,
)
from flwr.supercore.date import now
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.fab import Fab
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


def start_run(  # pylint: disable=too-many-locals, too-many-statements
    request: StartRunRequest,
    account: AccountInfo,
    state: LinkState,
    fleet_api_type: str | None,
) -> StartRunResponse:
    """Create run ID."""
    log(INFO, "ControlServicer.StartRun")

    verification_dict: dict[str, str] = {}
    note: str | None = None

    builtin_agent_fab = try_resolve_builtin_agent_fab(request.app_spec)
    if builtin_agent_fab is not None:
        fab_file, verification_dict = builtin_agent_fab
    elif request.app_spec:
        fab_file, verification_dict, note = _get_remote_fab(
            fleet_api_type, request.app_spec
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

    flwr_aid = account.flwr_aid
    account_name = account.account_name
    override_config = user_config_from_proto(request.override_config)

    state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)

    # Check (1) federation exists and (2) the flwr_aid is a member
    federation_id = _resolve_federation_id(state, account_name, request.federation)
    if not state.federation_manager.exists(federation_id):
        if request.federation:
            raise FlowerError(
                ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PERMISSION,
                f"Federation '{federation_id}' not found or has been archived.",
            )
        raise FlowerError(
            ApiErrorCode.FEDERATION_NOT_SPECIFIED, "No federation specified."
        )

    if not state.federation_manager.has_member(flwr_aid, federation_id):
        raise FlowerError(
            ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PERMISSION,
            f"Account with ID '{flwr_aid}' is not a member of the "
            f"federation '{federation_id}'.",
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
        sim_cfg = state.federation_manager.get_simulation_config(federation_id)
        if sim_cfg and not is_agentapp_bundle:
            primary_task_type = TaskType.SIMULATION
            runtime = RunTime.SIMULATION
            resolved_federation_config = SimulationConfig()
            resolved_federation_config.CopyFrom(sim_cfg)
            resolved_federation_config.MergeFrom(request.override_federation_config)

        state.federation_manager.can_execute(
            flwr_aid,
            ActionType.START_RUN,
            StartRunContext(federation_id=federation_id, runtime=runtime),
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
            federation_id,
            resolved_federation_config,
            flwr_aid,
            primary_task_type,
            request.series_id if request.HasField("series_id") else None,
        )

        if run_id == 0:
            raise FlowerError(
                ApiErrorCode.FAILED_TO_CREATE_RUN,
                "Failed to create or initialize run for "
                f"flwr_aid={flwr_aid}, federation_id={federation_id}, "
                f"fab_id={fab_id}, fab_version={fab_version}, "
                f"fab_hash={fab_hash}, primary_task_type={primary_task_type}.",
            )

        run = state.get_run_info(run_ids=[run_id])[0]
        series_id = run.series_id

    except ValueError as e:
        log(ERROR, "Could not start run: %s", str(e))
        raise FlowerError(
            ApiErrorCode.INVALID_RUN_CONFIG,
            "Could not start run for "
            f"flwr_aid={flwr_aid}, federation_id={federation_id}: {e}",
        ) from e

    log_msg = f"Created run {run_id} in federation {run.federation_id}"
    log(INFO, log_msg)
    return StartRunResponse(
        run_id=run_id, note=note, series_id=series_id, federation=run.federation_id
    )


def list_runs(
    request: ListRunsRequest,
    account: AccountInfo,
    state: LinkState,
) -> ListRunsResponse:
    """Handle `flwr ls` command."""
    log(INFO, "ControlServicer.ListRuns")

    flwr_aid = account.flwr_aid
    account_name = account.account_name
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
        run_id = request.run_id
        runs = state.get_run_info(run_ids=[run_id])

        # Exit if `run_id` not found
        if not runs:
            raise FlowerError(
                ApiErrorCode.RUN_ID_NOT_FOUND,
                f"Run {run_id} not found while listing runs for {flwr_aid}.",
            )

        # Check if requester is a member of the federation
        # that the run belongs to
        _validate_federation_membership_in_request(
            state, flwr_aid, runs[0].federation_id
        )

    # Clear objects of finished runs
    # Resolve only non-caller run owners; caller-owned runs use `account_name`.
    account_names = resolve_account_ids(
        {run.flwr_aid for run in runs if run.flwr_aid != flwr_aid}
    )
    account_names[flwr_aid] = account_name
    for run in runs:
        run.account_name = account_names[run.flwr_aid]
        if run.status.status == Status.FINISHED:
            state.object_store.delete_objects_in_run(run.run_id)

    # Construct and return response
    return ListRunsResponse(
        run_dict={run.run_id: run_to_proto(run) for run in runs},
        now=now().isoformat(),
    )


def list_run_series(
    request: ListRunSeriesRequest, account: AccountInfo, state: LinkState
) -> ListRunSeriesResponse:
    """List run series."""
    log(INFO, "ControlServicer.ListRunSeries")

    flwr_aid = account.flwr_aid
    updated_before = (
        request.updated_before if request.HasField("updated_before") else None
    )
    limit = request.limit if request.HasField("limit") else None
    federation_id = request.federation_id if request.HasField("federation_id") else None

    if federation_id is not None:
        _validate_federation_membership_in_request(state, flwr_aid, federation_id)
        federation_ids = [federation_id]
    else:
        federations = state.federation_manager.get_federations(flwr_aid)
        federation_ids = [federation.id for federation in federations]
    entries = state.get_run_series(
        federation_ids=federation_ids,
        updated_before=updated_before,
        limit=limit,
    )

    return ListRunSeriesResponse(entries=_with_last_run_statuses(state, entries))


def get_run_series(
    request: GetRunSeriesRequest, account: AccountInfo, state: LinkState
) -> GetRunSeriesResponse:
    """Get run series."""
    log(INFO, "ControlServicer.GetRunSeries")

    flwr_aid = account.flwr_aid
    series_id = request.series_id
    series_matches = state.get_run_series(series_ids=[series_id])

    # The caller must be a member of the federation
    if not series_matches or not state.federation_manager.has_member(
        flwr_aid, series_matches[0].federation
    ):
        raise FlowerError(
            ApiErrorCode.RUN_SERIES_ID_NOT_FOUND,
            f"Run series {series_id} not found for {flwr_aid}.",
        )

    # Get the run series context and construct the response
    # Run series context is created atomically by LinkState.create_run(...)
    # and should never be None.
    series_context = state.get_run_series_context(request.series_id)
    response = GetRunSeriesResponse(
        series=_with_last_run_statuses(state, series_matches)[0],
        context=context_to_proto(series_context) if series_context else None,
    )
    return response


def stop_run(
    request: StopRunRequest, account: AccountInfo, state: LinkState
) -> StopRunResponse:
    """Stop a given run ID."""
    log(INFO, "ControlServicer.StopRun")

    # Retrieve run ID and run
    run_id = request.run_id
    runs = state.get_run_info(run_ids=[run_id])

    # Exit if `run_id` not found
    if not runs:
        raise FlowerError(
            ApiErrorCode.RUN_ID_NOT_FOUND,
            f"Run {run_id} not found while stopping run.",
        )
    run = runs[0]

    flwr_aid = account.flwr_aid
    _validate_federation_membership_in_request(state, flwr_aid, run.federation_id)

    if run.status.status == Status.FINISHED:
        raise FlowerError(
            ApiErrorCode.RUN_ALREADY_FINISHED,
            f"Cannot stop run {run_id} for flwr_aid={flwr_aid}; "
            f"run is already finished with status={run.status}.",
        )

    return StopRunResponse(success=state.stop_run(run_id))


def get_login_details(
    request: GetLoginDetailsRequest, authn_plugin: ControlAuthnPlugin | None
) -> GetLoginDetailsResponse:
    """Start login."""
    _ = request
    log(INFO, "ControlServicer.GetLoginDetails")
    if authn_plugin is None:
        raise FlowerError(
            ApiErrorCode.NO_ACCOUNT_AUTH,
            "ControlServicer initialized without account authentication.",
        )

    # Get login details
    details = authn_plugin.get_login_details()

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


def get_auth_tokens(
    request: GetAuthTokensRequest, authn_plugin: ControlAuthnPlugin | None
) -> GetAuthTokensResponse:
    """Get auth token."""
    log(INFO, "ControlServicer.GetAuthTokens")
    if authn_plugin is None:
        raise FlowerError(
            ApiErrorCode.NO_ACCOUNT_AUTH,
            "ControlServicer initialized without account authentication.",
        )

    # Get auth tokens
    credentials = authn_plugin.get_auth_tokens(request.device_code)

    # Return empty response if credentials is None
    if credentials is None:
        return GetAuthTokensResponse()

    return GetAuthTokensResponse(
        access_token=credentials.access_token,
        refresh_token=credentials.refresh_token,
    )


def pull_artifacts(
    request: PullArtifactsRequest,
    account: AccountInfo,
    state: LinkState,
    artifact_provider: ArtifactProvider | None,
) -> PullArtifactsResponse:
    """Pull artifacts for a given run ID."""
    log(INFO, "ControlServicer.PullArtifacts")

    # Check if artifact provider is configured
    if artifact_provider is None:
        raise FlowerError(
            ApiErrorCode.NO_ARTIFACT_PROVIDER,
            "ControlServicer initialized without artifact provider.",
        )

    # Retrieve run ID and run
    run_id = request.run_id
    runs = state.get_run_info(run_ids=[run_id])

    # Exit if `run_id` not found
    if not runs:
        raise FlowerError(
            ApiErrorCode.RUN_ID_NOT_FOUND,
            f"Run {run_id} not found while pulling artifacts.",
        )
    run = runs[0]

    # Exit if the run is not finished yet
    if run.status.status != Status.FINISHED:
        raise FlowerError(
            ApiErrorCode.PULL_UNFINISHED_RUN,
            f"Cannot pull artifacts for run {run_id}; "
            f"status={run.status.status}, owner_aid={run.flwr_aid}.",
        )

    # Check if `flwr_aid` matches the run's `flwr_aid`
    flwr_aid = account.flwr_aid
    _check_flwr_aid_in_run(flwr_aid=flwr_aid, run=run)

    # Call artifact provider
    download_url = artifact_provider.get_url(run_id)
    return PullArtifactsResponse(url=download_url)


def register_node(
    request: RegisterNodeRequest, account: AccountInfo, state: LinkState
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
        raise FlowerError(
            ApiErrorCode.PUBLIC_KEY_NOT_VALID,
            f"Invalid public key in RegisterNode request: {err}",
        ) from err

    node_id = 0

    flwr_aid = account.flwr_aid
    state.federation_manager.can_execute(
        flwr_aid,
        ActionType.REGISTER_SUPERNODE,
        RegisterSupernodeContext(),
    )

    # Account name exists if `flwr_aid` exists
    account_name = account.account_name
    try:
        node_id = state.create_node(
            owner_aid=flwr_aid,
            owner_name=account_name,
            public_key=request.public_key,
            heartbeat_interval=HEARTBEAT_DEFAULT_INTERVAL,
        )

    except ValueError as err:
        # Public key already in use
        log(ERROR, "Public key already in use")
        raise FlowerError(
            ApiErrorCode.PUBLIC_KEY_ALREADY_IN_USE,
            f"Public key already in use while registering node for "
            f"flwr_aid={flwr_aid}, account_name={account_name}.",
        ) from err
    log(INFO, "[ControlServicer.RegisterNode] Created node_id=%s", node_id)

    return RegisterNodeResponse(node_id=node_id)


def unregister_node(
    request: UnregisterNodeRequest, account: AccountInfo, state: LinkState
) -> UnregisterNodeResponse:
    """Remove a SuperNode."""
    log(INFO, "ControlServicer.UnregisterNode")

    flwr_aid = account.flwr_aid
    try:
        state.delete_node(owner_aid=flwr_aid, node_id=request.node_id)
    except ValueError as err:
        log(ERROR, "Node ID not found for account")
        raise FlowerError(
            ApiErrorCode.NODE_NOT_FOUND,
            f"Node {request.node_id} not found for flwr_aid={flwr_aid}.",
        ) from err

    return UnregisterNodeResponse()


def list_nodes(
    request: ListNodesRequest, account: AccountInfo, state: LinkState
) -> ListNodesResponse:
    """List all SuperNodes."""
    _ = request
    log(INFO, "ControlServicer.ListNodes")

    nodes_info: Sequence[NodeInfo] = []
    # Retrieve all nodes for the account
    nodes_info = state.get_node_info(owner_aids=[account.flwr_aid])

    return ListNodesResponse(nodes_info=nodes_info, now=now().isoformat())


def list_federations(
    request: ListFederationsRequest, account: AccountInfo, state: LinkState
) -> ListFederationsResponse:
    """List all SuperNodes."""
    _ = request
    log(INFO, "ControlServicer.ListFederations")

    flwr_aid = account.flwr_aid

    # Get federations the account is a member of
    state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)
    federations = state.federation_manager.get_federations(flwr_aid)

    return ListFederationsResponse(
        federations=[
            Federation(
                name=fed.id,
                description=fed.description,
                archived=fed.archived,
                simulation=fed.simulation,
            )
            for fed in federations
        ]
    )


def show_federation(
    request: ShowFederationRequest, account: AccountInfo, state: LinkState
) -> ShowFederationResponse:
    """Show details of a specific Federation."""
    log(INFO, "ControlServicer.ShowFederation")

    # Ensure flwr_aid is a member of the requested federation
    federation_id = request.federation_name
    flwr_aid = account.flwr_aid
    state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)
    if not state.federation_manager.has_member(flwr_aid, federation_id):
        raise FlowerError(
            ApiErrorCode.FEDERATION_NOT_FOUND_OR_NOT_MEMBER,
            f"Federation '{federation_id}' not found or flwr_aid={flwr_aid} "
            "is not a member.",
        )

    # Fetch federation details
    details = state.federation_manager.get_details(federation_id)

    # Build Federation proto object
    federation_proto = Federation(
        name=federation_id,
        description=details.description,
        members=details.members,
        nodes=details.nodes,
        runs=[run_to_proto(run) for run in details.runs],
        archived=details.archived,
        simulation=details.simulation,
        config=details.config,
    )
    return ShowFederationResponse(federation=federation_proto, now=now().isoformat())


def create_federation(
    request: CreateFederationRequest, account: AccountInfo, state: LinkState
) -> CreateFederationResponse:
    """Create a new Federation."""
    log(INFO, "ControlServicer.CreateFederation")

    # Check that a federation is specified
    if not request.federation_name:
        raise FederationNotSpecified()

    # Ensure valid federation name is provided
    success, err_msg = validate_federation_name(request.federation_name)
    if not success:
        raise FlowerError(
            ApiErrorCode.INVALID_FEDERATION_NAME,
            f"Invalid federation name in CreateFederation request: "
            f"federation_name={request.federation_name}. {err_msg}",
            public_details=err_msg,
        )

    # Construct federation ID
    flwr_aid = account.flwr_aid
    state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)
    federation_id = f"@{account.account_name}/{request.federation_name}"

    runtime = RunTime.SIMULATION if request.simulation else RunTime.DEPLOYMENT
    state.federation_manager.can_execute(
        flwr_aid,
        ActionType.CREATE_FEDERATION,
        CreateFederationContext(
            federation_id=federation_id,
            runtime=runtime,
            visibility="private",
        ),
    )

    # Create federation
    federation = state.federation_manager.create_federation(
        federation_id=federation_id,
        description=request.description,
        flwr_aid=flwr_aid,
        simulation=request.simulation,
    )

    return CreateFederationResponse(
        federation=Federation(
            name=federation.id,
            description=federation.description,
            members=federation.members,
            simulation=federation.simulation,
        )
    )


def archive_federation(
    request: ArchiveFederationRequest, account: AccountInfo, state: LinkState
) -> ArchiveFederationResponse:
    """Archive a Federation."""
    log(INFO, "ControlServicer.ArchiveFederation")

    # Check that a federation is specified
    if not request.federation_name:
        raise FederationNotSpecified()

    # Archive federation
    state.federation_manager.archive_federation(
        flwr_aid=account.flwr_aid,
        federation_id=request.federation_name,
    )
    for run in state.get_run_info(federation_ids=[request.federation_name]):
        if run.status.status != Status.FINISHED:
            state.stop_run(run.run_id)

    return ArchiveFederationResponse()


def add_node_to_federation(
    request: AddNodeToFederationRequest, account: AccountInfo, state: LinkState
) -> AddNodeToFederationResponse:
    """Add a node to a Federation."""
    log(INFO, "ControlServicer.AddNodeToFederation")

    # Validate federation, node ID, and ownership
    flwr_aid = account.flwr_aid
    state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)
    _validate_federation_and_node_in_request(
        state, flwr_aid, request.federation_name, request.node_id
    )

    # Add node to the federation
    state.federation_manager.add_supernode(
        flwr_aid=flwr_aid,
        federation_id=request.federation_name,
        node_id=request.node_id,
    )

    return AddNodeToFederationResponse()


def remove_node_from_federation(
    request: RemoveNodeFromFederationRequest, account: AccountInfo, state: LinkState
) -> RemoveNodeFromFederationResponse:
    """Remove a node from a Federation."""
    log(INFO, "ControlServicer.RemoveNodeFromFederation")

    # Validate federation, node ID, and ownership
    flwr_aid = account.flwr_aid
    _validate_federation_and_node_in_request(
        state, flwr_aid, request.federation_name, request.node_id
    )

    # Remove node from the federation
    state.federation_manager.remove_supernode(
        flwr_aid=flwr_aid,
        federation_id=request.federation_name,
        node_id=request.node_id,
    )

    return RemoveNodeFromFederationResponse()


def remove_account_from_federation(
    request: RemoveAccountFromFederationRequest, account: AccountInfo, state: LinkState
) -> RemoveAccountFromFederationResponse:
    """Remove an account from a Federation."""
    log(INFO, "ControlServicer.RemoveAccountFromFederation")

    target_account = None if not request.account_name else request.account_name

    removed_flwr_aid = state.federation_manager.remove_account(
        flwr_aid=account.flwr_aid,
        federation_id=request.federation_name,
        target_account_name=target_account,
    )
    # Get runs from account that was removed
    # and stop them.
    for run in state.get_run_info(
        federation_ids=[request.federation_name],
        flwr_aids=[removed_flwr_aid],
        statuses=[Status.PENDING, Status.STARTING, Status.RUNNING],
    ):
        state.stop_run(run.run_id)
    return RemoveAccountFromFederationResponse()


def create_invitation(
    request: CreateInvitationRequest, account: AccountInfo, state: LinkState
) -> CreateInvitationResponse:
    """Create an invitation."""
    log(INFO, "ControlServicer.CreateInvitation")

    flwr_aid = account.flwr_aid
    state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)
    federation_id = request.federation_name
    invitee_account_name = request.invitee_account_name

    runtime = (
        RunTime.SIMULATION
        if state.federation_manager.get_simulation_config(federation_id)
        else RunTime.DEPLOYMENT
    )

    state.federation_manager.can_execute(
        flwr_aid=flwr_aid,
        action=ActionType.CREATE_INVITATION,
        context=CreateInvitationContext(
            federation_id=federation_id,
            invitee_account_name=invitee_account_name,
            runtime=runtime,
        ),
    )

    state.federation_manager.create_invitation(
        flwr_aid=flwr_aid,
        federation_id=federation_id,
        invitee_account_name=invitee_account_name,
    )
    return CreateInvitationResponse()


def list_invitations(
    request: ListInvitationsRequest, account: AccountInfo, state: LinkState
) -> ListInvitationsResponse:
    """List invitations."""
    _ = request
    log(INFO, "ControlServicer.ListInvitations")

    created_invitations, received_invitations = (
        state.federation_manager.list_invitations(account.flwr_aid)
    )
    return ListInvitationsResponse(
        created_invitations=created_invitations,
        received_invitations=received_invitations,
    )


def accept_invitation(
    request: AcceptInvitationRequest, account: AccountInfo, state: LinkState
) -> AcceptInvitationResponse:
    """Accept an invitation."""
    log(INFO, "ControlServicer.AcceptInvitation")

    flwr_aid = account.flwr_aid
    federation_id = request.federation_name

    runtime = (
        RunTime.SIMULATION
        if state.federation_manager.get_simulation_config(federation_id)
        else RunTime.DEPLOYMENT
    )

    state.federation_manager.can_execute(
        flwr_aid=flwr_aid,
        action=ActionType.ACCEPT_INVITATION,
        context=AcceptInvitationContext(
            federation_id=federation_id,
            runtime=runtime,
        ),
    )

    state.federation_manager.accept_invitation(
        flwr_aid=flwr_aid,
        federation_id=request.federation_name,
    )
    return AcceptInvitationResponse()


def reject_invitation(
    request: RejectInvitationRequest, account: AccountInfo, state: LinkState
) -> RejectInvitationResponse:
    """Reject an invitation."""
    log(INFO, "ControlServicer.RejectInvitation")

    state.federation_manager.reject_invitation(
        flwr_aid=account.flwr_aid,
        federation_id=request.federation_name,
    )
    return RejectInvitationResponse()


def revoke_invitation(
    request: RevokeInvitationRequest, account: AccountInfo, state: LinkState
) -> RevokeInvitationResponse:
    """Revoke an invitation."""
    log(INFO, "ControlServicer.RevokeInvitation")

    state.federation_manager.revoke_invitation(
        flwr_aid=account.flwr_aid,
        federation_id=request.federation_name,
        invitee_account_name=request.invitee_account_name,
    )
    return RevokeInvitationResponse()


def configure_simulation_federation(
    request: ConfigureSimulationFederationRequest,
    account: AccountInfo,
    state: LinkState,
) -> ConfigureSimulationFederationResponse:
    """Configure a federation for simulation."""
    log(INFO, "ControlServicer.ConfigureSimulationFederation")

    flwr_aid = account.flwr_aid
    account_name = account.account_name

    state.federation_manager.ensure_default_federations_exist(flwr_aid=flwr_aid)
    federation_id = _resolve_federation_id(state, account_name, request.federation_name)
    if not state.federation_manager.exists(federation_id):
        if request.federation_name:
            raise FlowerError(
                ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PERMISSION,
                f"Federation '{federation_id}' not found or has been archived.",
            )
        raise FlowerError(
            ApiErrorCode.FEDERATION_NOT_SPECIFIED, "No federation specified."
        )
    state.federation_manager.set_simulation_config(
        flwr_aid=flwr_aid,
        federation_id=federation_id,
        config=request.config,
    )

    return ConfigureSimulationFederationResponse(federation_name=federation_id)


def _resolve_federation_id(
    state: LinkState, account_name: str, federation_id: str
) -> str:
    """Return the requested federation ID or derive the default federation ID."""
    if not federation_id:
        federation_manager = state.federation_manager
        if isinstance(federation_manager, NoOpFederationManager):
            federation_id = NOOP_FEDERATION_ID
        else:
            federation_id = f"@{account_name}/{DEFAULT_FEDERATION_SIMULATION}"
    return federation_id


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
    federation_id: str,
    node_id: int,
) -> None:
    """Validate federation membership and node ownership for federation updates."""
    _validate_federation_membership_in_request(state, flwr_aid, federation_id)
    nodes_info = state.get_node_info(node_ids=[node_id])
    if not nodes_info or nodes_info[0].owner_aid != flwr_aid:
        raise FlowerError(
            ApiErrorCode.NODE_NOT_FOUND_OR_NOT_OWNER,
            f"Node {node_id} not found or {flwr_aid} is not its owner.",
        )


def _validate_federation_membership_in_request(
    state: LinkState,
    flwr_aid: str,
    federation_id: str,
) -> None:
    """Validate that a federation exists and the requester is one of its members."""
    if not federation_id:
        raise FederationNotSpecified()

    # Check that the federation exists
    if not state.federation_manager.exists(federation_id):
        raise FlowerError(
            ApiErrorCode.FEDERATION_NOT_FOUND,
            message=f"Federation `{federation_id}` not found.",
        )

    # Check that the requester is a member of the federation
    if not state.federation_manager.has_member(flwr_aid, federation_id):
        raise FlowerError(
            ApiErrorCode.FEDERATION_NOT_FOUND,
            message=f"`{flwr_aid}` is not a member of federation `{federation_id}`.",
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


def _check_flwr_aid_in_run(flwr_aid: str, run: Run) -> None:
    """Guard clause to check if `flwr_aid` matches the run's `flwr_aid`."""
    # `run.flwr_aid` must not be an empty string. Abort if it is empty.
    run_flwr_aid = run.flwr_aid
    if not run_flwr_aid:
        raise FlowerError(
            ApiErrorCode.RUN_NOT_ASSOCIATED_WITH_ACCOUNT,
            f"Run {run.run_id} is not associated with a `flwr_aid`.",
        )

    # Exit if `flwr_aid` does not match the run's `flwr_aid`
    if run_flwr_aid != flwr_aid:
        raise FlowerError(
            ApiErrorCode.RUN_ID_NOT_BELONG_TO_ACCOUNT,
            f"Run {run.run_id} does not belong to the account {flwr_aid}",
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
) -> tuple[bytes, dict[str, str], str | None]:
    """Get remote FAB from Flower Hub."""
    if fleet_api_type == TRANSPORT_TYPE_GRPC_ADAPTER:
        raise FlowerError(
            ApiErrorCode.UNSUPPORTED_FAB_HUB_TRANSPORT,
            "The selected SuperLink transport type is not "
            "supported for connecting to Flower Hub.",
        )

    # Parse and validate app specification
    try:
        app_id, app_version = parse_app_spec(app_spec)
    except ValueError as e:
        raise FlowerError(
            ApiErrorCode.INVALID_APP_SPEC,
            f"Invalid app specification: {app_spec}",
        ) from e

    # Request download link and verification information
    url = f"{PLATFORM_API_URL}/hub/fetch-fab"
    try:
        presigned_url, verifications, note = request_download_link(
            app_id, app_version, url, "fab_url"
        )
    except ValueError as e:
        raise FlowerError(
            ApiErrorCode.FAB_DOWNLOAD_LINK_FAILURE,
            f"Failed to request FAB download link. app-id:{app_id}, ",
            f"app_version: {app_version}, url: {url}",
        ) from e

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
        raise FlowerError(
            ApiErrorCode.FAB_DOWNLOAD_FAILURE,
            f"FAB download failed for app_id={app_id}, app_version={app_version}: {e}",
        ) from e
    fab_file = r.content
    return fab_file, verification_dict, note
