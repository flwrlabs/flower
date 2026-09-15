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


import hashlib
import json
import time
from collections.abc import Generator, Sequence
from logging import ERROR, INFO
from typing import Any, cast

import grpc
import requests

from flwr.cli.config_utils import get_fab_metadata
from flwr.common import Context, RecordDict, now
from flwr.common.constant import (
    FAB_MAX_SIZE,
    HEARTBEAT_DEFAULT_INTERVAL,
    LOG_STREAM_INTERVAL,
    NO_ACCOUNT_AUTH_MESSAGE,
    NO_ARTIFACT_PROVIDER_MESSAGE,
    NODE_NOT_FOUND_MESSAGE,
    PUBLIC_KEY_ALREADY_IN_USE_MESSAGE,
    PUBLIC_KEY_NOT_VALID,
    PULL_UNFINISHED_RUN_MESSAGE,
    RUN_ID_NOT_FOUND_MESSAGE,
    TRANSPORT_TYPE_GRPC_ADAPTER,
    Status,
    SubStatus,
)
from flwr.common.logger import log
from flwr.common.profiling import merge_profile_events
from flwr.common.serde import (
    config_record_from_proto,
    run_to_proto,
    user_config_from_proto,
)
from flwr.common.typing import Fab, Run, RunStatus
from flwr.proto import control_pb2_grpc  # pylint: disable=E0611
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetAuthTokensRequest,
    GetAuthTokensResponse,
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
    GetRunProfileRequest,
    GetRunProfileResponse,
    ListFederationsRequest,
    ListFederationsResponse,
    ListNodesRequest,
    ListNodesResponse,
    ListRunsRequest,
    ListRunsResponse,
    PullArtifactsRequest,
    PullArtifactsResponse,
    RegisterNodeRequest,
    RegisterNodeResponse,
    ShowFederationRequest,
    ShowFederationResponse,
    StartRunRequest,
    StartRunResponse,
    StopRunRequest,
    StopRunResponse,
    StreamLogsRequest,
    StreamLogsResponse,
    StreamRunProfileRequest,
    UnregisterNodeRequest,
    UnregisterNodeResponse,
)
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.constant import PLATFORM_API_URL
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.object_store import ObjectStore, ObjectStoreFactory
from flwr.supercore.primitives.asymmetric import bytes_to_public_key, uses_nist_ec_curve
from flwr.supercore.utils import parse_app_spec, request_download_link
from flwr.superlink.artifact_provider import ArtifactProvider
from flwr.superlink.auth_plugin import ControlAuthnPlugin

from .control_account_auth_interceptor import get_current_account_info


class ControlServicer(control_pb2_grpc.ControlServicer):
    """Control API servicer."""

    def __init__(  # pylint: disable=R0913, R0917
        self,
        linkstate_factory: LinkStateFactory,
        ffs_factory: FfsFactory,
        objectstore_factory: ObjectStoreFactory,
        is_simulation: bool,
        authn_plugin: ControlAuthnPlugin,
        artifact_provider: ArtifactProvider | None = None,
        fleet_api_type: str | None = None,
    ) -> None:
        self.linkstate_factory = linkstate_factory
        self.ffs_factory = ffs_factory
        self.objectstore_factory = objectstore_factory
        self.is_simulation = is_simulation
        self.authn_plugin = authn_plugin
        self.artifact_provider = artifact_provider
        self.fleet_api_type = fleet_api_type

    def StartRun(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        self, request: StartRunRequest, context: grpc.ServicerContext
    ) -> StartRunResponse:
        """Create run ID."""
        log(INFO, "ControlServicer.StartRun")
        state = self.linkstate_factory.state()
        ffs = self.ffs_factory.ffs()

        verification_dict: dict[str, str] = {}
        if request.app_spec:
            fab_file, verification_dict = _get_remote_fab(
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

        flwr_aid = get_current_account_info().flwr_aid
        flwr_aid = _check_flwr_aid_exists(flwr_aid, context)
        override_config = user_config_from_proto(request.override_config)
        federation_options = config_record_from_proto(request.federation_options)

        try:
            # Check that num-supernodes is set
            if self.is_simulation and "num-supernodes" not in federation_options:
                raise ValueError(
                    "Federation options doesn't contain key `num-supernodes`."
                )

            # Check (1) federation exists and (2) the flwr_aid is a member
            federation = request.federation

            if not state.federation_manager.exists(federation):
                raise ValueError(f"Federation '{federation}' does not exist.")

            if not state.federation_manager.has_member(flwr_aid, federation):
                raise ValueError(
                    f"Account with ID '{flwr_aid}' is not a member of the "
                    f"federation '{federation}'. Please log in with another account "
                    "or request access to this federation."
                )

            # Create run
            fab = Fab(
                hashlib.sha256(fab_file).hexdigest(),
                fab_file,
                verification_dict,
            )
            fab_hash = ffs.put(fab.content, fab.verifications)

            if fab_hash != fab.hash_str:
                raise RuntimeError(
                    f"FAB ({fab.hash_str}) hash from request doesn't match contents"
                )
            fab_id, fab_version = get_fab_metadata(fab.content)

            run_id = state.create_run(
                fab_id,
                fab_version,
                fab_hash,
                override_config,
                request.federation,
                federation_options,
                flwr_aid,
            )

            # Initialize node config
            node_config = {}
            if self.artifact_provider is not None:
                node_config = {
                    "output_dir": self.artifact_provider.output_dir,
                    "tmp_dir": self.artifact_provider.tmp_dir,
                }

            # Create an empty context for the Run
            context = Context(
                run_id=run_id,
                node_id=0,
                # Dict is invariant in mypy
                node_config=node_config,  # type: ignore[arg-type]
                state=RecordDict(),
                run_config={},
            )

            # Register the context at the LinkState
            state.set_serverapp_context(run_id=run_id, context=context)

        # pylint: disable-next=broad-except
        except Exception as e:
            log(ERROR, "Could not start run: %s", str(e))
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                str(e),
            )

        log(INFO, "Created run %s", str(run_id))
        return StartRunResponse(run_id=run_id)

    def StreamLogs(  # pylint: disable=C0103
        self, request: StreamLogsRequest, context: grpc.ServicerContext
    ) -> Generator[StreamLogsResponse, Any, None]:
        """Get logs."""
        log(INFO, "ControlServicer.StreamLogs")
        state = self.linkstate_factory.state()

        # Retrieve run ID and run
        run_id = request.run_id
        run = state.get_run(run_id)

        # Exit if `run_id` not found
        if not run:
            context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)

        # Check if `flwr_aid` matches the run's `flwr_aid`
        flwr_aid = get_current_account_info().flwr_aid
        _check_flwr_aid_in_run(flwr_aid=flwr_aid, run=cast(Run, run), context=context)

        after_timestamp = request.after_timestamp + 1e-6
        while context.is_active():
            log_msg, latest_timestamp = state.get_serverapp_log(run_id, after_timestamp)
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
            run_status = state.get_run_status({run_id})[run_id]
            if run_status.status == Status.FINISHED:
                log(INFO, "All logs for run ID `%s` returned", run_id)

                # Delete objects of the run from the object store
                self.objectstore_factory.store().delete_objects_in_run(run_id)
                break

            time.sleep(LOG_STREAM_INTERVAL)  # Sleep briefly to avoid busy waiting

    def ListRuns(
        self, request: ListRunsRequest, context: grpc.ServicerContext
    ) -> ListRunsResponse:
        """Handle `flwr ls` command."""
        log(INFO, "ControlServicer.List")
        state = self.linkstate_factory.state()

        # Build a set of run IDs for `flwr ls --runs`
        if not request.HasField("run_id"):
            # If no `run_id` is specified and account auth is enabled,
            # return run IDs for the authenticated account
            flwr_aid = get_current_account_info().flwr_aid
            _check_flwr_aid_exists(flwr_aid, context)
            run_ids = state.get_run_ids(flwr_aid=flwr_aid)
        # Build a set of run IDs for `flwr ls --run-id <run_id>`
        else:
            # Retrieve run ID and run
            run_id = request.run_id
            run = state.get_run(run_id)

            # Exit if `run_id` not found
            if not run:
                context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)
                raise grpc.RpcError()  # This line is unreachable

            # Check if `flwr_aid` matches the run's `flwr_aid`
            flwr_aid = get_current_account_info().flwr_aid
            _check_flwr_aid_in_run(flwr_aid=flwr_aid, run=run, context=context)

            run_ids = {run_id}

        # Init the object store
        store = self.objectstore_factory.store()
        return _create_list_runs_response(run_ids, state, store)

    def GetRunProfile(
        self, request: GetRunProfileRequest, context: grpc.ServicerContext
    ) -> GetRunProfileResponse:
        """Get run profile summary."""
        log(INFO, "ControlServicer.GetRunProfile")
        state = self.linkstate_factory.state()

        run_id = request.run_id
        run = state.get_run(run_id)
        if not run:
            context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)

        flwr_aid = get_current_account_info().flwr_aid
        _check_flwr_aid_in_run(flwr_aid=flwr_aid, run=cast(Run, run), context=context)

        serverapp_context = state.get_serverapp_context(run_id)
        if serverapp_context is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "Profile summary not found.")
        serverapp_context = cast(Context, serverapp_context)

        if "profile_summary" not in serverapp_context.state:
            context.abort(grpc.StatusCode.NOT_FOUND, "Profile summary not found.")

        record = serverapp_context.state["profile_summary"]
        if "json" not in record:
            context.abort(grpc.StatusCode.NOT_FOUND, "Profile summary not found.")

        summary = json.loads(cast(bytes, record["json"]))
        summary = merge_profile_events(summary, state.get_profile_events(run_id))
        return GetRunProfileResponse(summary_json=json.dumps(summary).encode())

    def StreamRunProfile(
        self, request: StreamRunProfileRequest, context: grpc.ServicerContext
    ) -> Generator[GetRunProfileResponse, Any, None]:
        """Stream run profile summary updates."""
        log(INFO, "ControlServicer.StreamRunProfile")
        state = self.linkstate_factory.state()

        run_id = request.run_id
        run = state.get_run(run_id)
        if not run:
            context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)

        flwr_aid = get_current_account_info().flwr_aid
        _check_flwr_aid_in_run(flwr_aid=flwr_aid, run=cast(Run, run), context=context)

        last_summary = None
        while context.is_active():
            serverapp_context = state.get_serverapp_context(run_id)
            if (
                serverapp_context is not None
                and "profile_summary" in serverapp_context.state
            ):
                record = serverapp_context.state["profile_summary"]
                if "json" in record:
                    summary = json.loads(cast(bytes, record["json"]))
                    summary_json = json.dumps(
                        merge_profile_events(summary, state.get_profile_events(run_id))
                    ).encode()
                    if summary_json != last_summary:
                        last_summary = summary_json
                        yield GetRunProfileResponse(summary_json=summary_json)

            time.sleep(LOG_STREAM_INTERVAL)

    def StopRun(
        self, request: StopRunRequest, context: grpc.ServicerContext
    ) -> StopRunResponse:
        """Stop a given run ID."""
        log(INFO, "ControlServicer.StopRun")
        state = self.linkstate_factory.state()

        # Retrieve run ID and run
        run_id = request.run_id
        run = state.get_run(run_id)

        # Exit if `run_id` not found
        if not run:
            context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)
            raise grpc.RpcError()  # This line is unreachable

        # Check if `flwr_aid` matches the run's `flwr_aid`
        flwr_aid = get_current_account_info().flwr_aid
        _check_flwr_aid_in_run(flwr_aid=flwr_aid, run=run, context=context)

        run_status = state.get_run_status({run_id})[run_id]
        if run_status.status == Status.FINISHED:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"Run ID {run_id} is already finished",
            )

        # Update run status to finished:stopped
        update_success = state.update_run_status(
            run_id=run_id,
            new_status=RunStatus(Status.FINISHED, SubStatus.STOPPED, ""),
        )

        # Delete the token associated with the run to stop further operations
        state.delete_token(run_id)

        if update_success:
            message_ids: set[str] = state.get_message_ids_from_run_id(run_id)

            # Delete Messages and their replies for the `run_id`
            state.delete_messages(message_ids)

            # Delete objects of the run from the object store
            self.objectstore_factory.store().delete_objects_in_run(run_id)

        return StopRunResponse(success=update_success)

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
        run = state.get_run(run_id)

        # Exit if `run_id` not found
        if not run:
            context.abort(grpc.StatusCode.NOT_FOUND, RUN_ID_NOT_FOUND_MESSAGE)
            raise grpc.RpcError()  # This line is unreachable

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

        flwr_aid = get_current_account_info().flwr_aid
        flwr_aid = _check_flwr_aid_exists(flwr_aid, context)
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

        flwr_aid = get_current_account_info().flwr_aid
        flwr_aid = _check_flwr_aid_exists(flwr_aid, context)
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

        if self.is_simulation:
            log(ERROR, "ListNodes is not available in simulation mode.")
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "ListNodes is not available in simulation mode.",
            )
            raise grpc.RpcError()  # This line is unreachable

        nodes_info: Sequence[NodeInfo] = []
        # Init link state
        state = self.linkstate_factory.state()

        flwr_aid = get_current_account_info().flwr_aid
        flwr_aid = _check_flwr_aid_exists(flwr_aid, context)
        # Retrieve all nodes for the account
        nodes_info = state.get_node_info(owner_aids=[flwr_aid])

        return ListNodesResponse(nodes_info=nodes_info, now=now().isoformat())

    def ListFederations(
        self, request: ListFederationsRequest, context: grpc.ServicerContext
    ) -> ListFederationsResponse:
        """List all SuperNodes."""
        log(INFO, "ControlServicer.ListFederations")

        # Init link state
        state = self.linkstate_factory.state()

        flwr_aid = get_current_account_info().flwr_aid
        flwr_aid = _check_flwr_aid_exists(flwr_aid, context)

        # Get federations the account is a member of
        federations = state.federation_manager.get_federations(flwr_aid=flwr_aid)

        return ListFederationsResponse(
            federations=[Federation(name=fed) for fed in federations]
        )

    def ShowFederation(
        self, request: ShowFederationRequest, context: grpc.ServicerContext
    ) -> ShowFederationResponse:
        """Show details of a specific Federation."""
        log(INFO, "ControlServicer.ShowFederation")

        # Init link state
        state = self.linkstate_factory.state()

        flwr_aid = get_current_account_info().flwr_aid
        flwr_aid = _check_flwr_aid_exists(flwr_aid, context)

        # Get federations the account is a member of
        federations = state.federation_manager.get_federations(flwr_aid=flwr_aid)

        # Ensure flwr_aid is a member of the requested federation
        federation = request.federation_name
        if federation not in federations:
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
            member_aids=details.member_aids,
            nodes=details.nodes,
            runs=[run_to_proto(run) for run in details.runs],
        )
        return ShowFederationResponse(
            federation=federation_proto, now=now().isoformat()
        )


def _create_list_runs_response(
    run_ids: set[int], state: LinkState, store: ObjectStore
) -> ListRunsResponse:
    """Create response for `flwr ls --runs` and `flwr ls --run-id <run_id>`."""
    run_dict = {run_id: run for run_id in run_ids if (run := state.get_run(run_id))}

    # Delete objects of finished runs from the object store
    for run_id, run in run_dict.items():
        if run.status.status == Status.FINISHED:
            store.delete_objects_in_run(run_id)

    return ListRunsResponse(
        run_dict={run_id: run_to_proto(run) for run_id, run in run_dict.items()},
        now=now().isoformat(),
    )


def _check_flwr_aid_exists(flwr_aid: str | None, context: grpc.ServicerContext) -> str:
    """Guard clause to check if `flwr_aid` exists."""
    if flwr_aid is None:
        context.abort(
            grpc.StatusCode.PERMISSION_DENIED,
            "️⛔️ Failed to fetch the account information.",
        )
        raise RuntimeError  # This line is unreachable
    return flwr_aid


def _check_flwr_aid_in_run(
    flwr_aid: str | None, run: Run, context: grpc.ServicerContext
) -> None:
    """Guard clause to check if `flwr_aid` matches the run's `flwr_aid`."""
    _check_flwr_aid_exists(flwr_aid, context)
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
) -> tuple[bytes, dict[str, str]]:
    """Get remote FAB from Flower platform API."""
    if fleet_api_type == TRANSPORT_TYPE_GRPC_ADAPTER:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            "The selected SuperLink transport type is not "
            "supported for connecting to Flower Platform.",
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
        presigned_url, verifications = request_download_link(
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

    # Download FAB from Flower platform API
    try:
        r = requests.get(presigned_url, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        context.abort(
            grpc.StatusCode.FAILED_PRECONDITION,
            f"FAB download failed: {str(e)}",
        )
    fab_file = r.content
    return fab_file, verification_dict
