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
"""ClientAppIo API servicer."""


from logging import DEBUG, ERROR
from typing import cast

import grpc

from flwr.common import Context
from flwr.common.logger import log
from flwr.common.serde import (
    context_from_proto,
    context_to_proto,
    fab_to_proto,
    message_from_proto,
    message_to_proto,
    run_to_proto,
)
from flwr.common.typing import Fab, Run

# pylint: disable=E0611
from flwr.proto import clientappio_pb2_grpc
from flwr.proto.appio_pb2 import (
    ListAppsToLaunchRequest,
    ListAppsToLaunchResponse,
    PullAppInputsRequest,
    PullAppInputsResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushAppOutputsRequest,
    PushAppOutputsResponse,
    RequestTokenRequest,
    RequestTokenResponse,
)
from flwr.proto.heartbeat_pb2 import SendAppHeartbeatRequest, SendAppHeartbeatResponse
from flwr.proto.message_pb2 import (
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse

# pylint: disable=E0601
from flwr.supercore.ffs import FfsFactory
from flwr.supercore.inflatable.inflatable_object import UnexpectedObjectContentError
from flwr.supercore.object_store import NoObjectInStoreError, ObjectStoreFactory
from flwr.supernode.nodestate import NodeStateFactory


# pylint: disable=C0103,W0613,W0201
class ClientAppIoServicer(clientappio_pb2_grpc.ClientAppIoServicer):
    """ClientAppIo API servicer."""

    def __init__(
        self,
        state_factory: NodeStateFactory,
        ffs_factory: FfsFactory,
        objectstore_factory: ObjectStoreFactory,
    ) -> None:
        self.state_factory = state_factory
        self.ffs_factory = ffs_factory
        self.objectstore_factory = objectstore_factory

    def ListAppsToLaunch(
        self,
        request: ListAppsToLaunchRequest,
        context: grpc.ServicerContext,
    ) -> ListAppsToLaunchResponse:
        """Get run IDs with apps to launch."""
        log(DEBUG, "ClientAppIo.ListAppsToLaunch")

        # Initialize state connection
        state = self.state_factory.state()

        # Get run IDs with pending messages
        run_ids = state.get_run_ids_with_pending_messages()

        # Return run IDs
        return ListAppsToLaunchResponse(run_ids=run_ids)

    def RequestToken(
        self, request: RequestTokenRequest, context: grpc.ServicerContext
    ) -> RequestTokenResponse:
        """Request token."""
        log(DEBUG, "ClientAppIo.RequestToken")

        # Initialize state connection
        state = self.state_factory.state()

        # Attempt to create a token for the provided run ID
        token = state.create_token(request.run_id)

        # Return the token
        return RequestTokenResponse(token=token or "")

    def GetRun(
        self, request: GetRunRequest, context: grpc.ServicerContext
    ) -> GetRunResponse:
        """Get run information."""
        log(DEBUG, "ClientAppIo.GetRun")

        # Initialize state connection
        state = self.state_factory.state()

        # Retrieve run information
        run = state.get_run(request.run_id)

        if run is None:
            return GetRunResponse()

        return GetRunResponse(run=run_to_proto(run))

    def PullAppInputs(
        self, request: PullAppInputsRequest, context: grpc.ServicerContext
    ) -> PullAppInputsResponse:
        """Pull Message, Context, and Run."""
        log(DEBUG, "ClientAppIo.PullAppInputs")

        # Initialize state and ffs connection
        state = self.state_factory.state()
        ffs = self.ffs_factory.ffs()

        # Validate the token
        run_id = state.get_run_id_by_token(request.token)
        if run_id is None or not state.verify_token(run_id, request.token):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token.",
            )
            raise RuntimeError("This line should never be reached.")

        # Retrieve context, run and fab for this run
        context = cast(Context, state.get_context(run_id))
        run = cast(Run, state.get_run(run_id))

        # Retrieve FAB from FFS
        if result := ffs.get(run.fab_hash):
            content, verifications = result
            log(
                DEBUG,
                "Retrieved FAB: hash=%s, content_len=%d, verifications=%s",
                run.fab_hash,
                len(content),
                verifications,
            )
            fab = Fab(run.fab_hash, content, verifications)
        else:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"FAB with hash {run.fab_hash} not found in FFS.",
            )
            raise RuntimeError("This line should never be reached.")

        return PullAppInputsResponse(
            context=context_to_proto(context),
            run=run_to_proto(run),
            fab=fab_to_proto(fab),
        )

    def PushAppOutputs(
        self, request: PushAppOutputsRequest, context: grpc.ServicerContext
    ) -> PushAppOutputsResponse:
        """Push Message and Context."""
        log(DEBUG, "ClientAppIo.PushAppOutputs")

        # Initialize state connection
        state = self.state_factory.state()

        # Validate the token
        run_id = state.get_run_id_by_token(request.token)
        if run_id is None or not state.verify_token(run_id, request.token):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token.",
            )
            raise RuntimeError("This line should never be reached.")

        # Save the context to the state
        state.store_context(context_from_proto(request.context))

        # Remove the token to make the run eligible for processing
        # A run associated with a token cannot be handled until its token is cleared
        state.delete_token(run_id)

        return PushAppOutputsResponse()

    def PullMessage(
        self, request: PullAppMessagesRequest, context: grpc.ServicerContext
    ) -> PullAppMessagesResponse:
        """Pull one Message."""
        # Initialize state and store connection
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Validate the token
        run_id = state.get_run_id_by_token(request.token)
        if run_id is None or not state.verify_token(run_id, request.token):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token.",
            )
            raise RuntimeError("This line should never be reached.")

        # Retrieve message for this run
        message = state.get_messages(run_ids=[run_id], is_reply=False)[0]

        # Record message processing start time
        state.record_message_processing_start(message_id=message.metadata.message_id)

        # Retrieve the object tree for the message
        object_tree = store.get_object_tree(message.metadata.message_id)

        return PullAppMessagesResponse(
            messages_list=[message_to_proto(message)],
            message_object_trees=[object_tree],
        )

    def PushMessage(
        self, request: PushAppMessagesRequest, context: grpc.ServicerContext
    ) -> PushAppMessagesResponse:
        """Push one Message."""
        # Initialize state and store connection
        state = self.state_factory.state()
        store = self.objectstore_factory.store()

        # Validate the token
        run_id = state.get_run_id_by_token(request.token)
        if run_id is None or not state.verify_token(run_id, request.token):
            context.abort(
                grpc.StatusCode.PERMISSION_DENIED,
                "Invalid token.",
            )
            raise RuntimeError("This line should never be reached.")

        # Record message processing end time
        state.record_message_processing_end(
            message_id=request.messages_list[0].metadata.reply_to_message_id
        )

        # Store Message object to descendants mapping and preregister objects
        objects_to_push: set[str] = set()
        for object_tree in request.message_object_trees:
            objects_to_push |= set(store.preregister(run_id, object_tree))

        # Save the message to the state
        state.store_message(message_from_proto(request.messages_list[0]))
        return PushAppMessagesResponse(objects_to_push=objects_to_push)

    def SendAppHeartbeat(
        self, request: SendAppHeartbeatRequest, context: grpc.ServicerContext
    ) -> SendAppHeartbeatResponse:
        """Handle a heartbeat from an app process."""
        log(DEBUG, "ClientAppIoServicer.SendAppHeartbeat")
        # Initialize state
        state = self.state_factory.state()

        # Acknowledge the heartbeat
        success = state.acknowledge_app_heartbeat(request.token)
        return SendAppHeartbeatResponse(success=success)

    def PushObject(
        self, request: PushObjectRequest, context: grpc.ServicerContext
    ) -> PushObjectResponse:
        """Push an object to the ObjectStore."""
        log(DEBUG, "ClientAppIoServicer.PushObject")

        # Init state and store
        store = self.objectstore_factory.store()

        # Insert in store
        stored = False
        try:
            store.put(request.object_id, request.object_content)
            stored = True
        except (NoObjectInStoreError, ValueError) as e:
            log(ERROR, str(e))
        except UnexpectedObjectContentError as e:
            # Object content is not valid
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(e))

        return PushObjectResponse(stored=stored)

    def PullObject(
        self, request: PullObjectRequest, context: grpc.ServicerContext
    ) -> PullObjectResponse:
        """Pull an object from the ObjectStore."""
        log(DEBUG, "ClientAppIoServicer.PullObject")

        # Init state and store
        store = self.objectstore_factory.store()

        # Fetch from store
        content = store.get(request.object_id)
        if content is not None:
            object_available = content != b""
            return PullObjectResponse(
                object_found=True,
                object_available=object_available,
                object_content=content,
            )
        return PullObjectResponse(object_found=False, object_available=False)

    def ConfirmMessageReceived(
        self, request: ConfirmMessageReceivedRequest, context: grpc.ServicerContext
    ) -> ConfirmMessageReceivedResponse:
        """Confirm message received."""
        log(DEBUG, "ClientAppIoServicer.ConfirmMessageReceived")

        # Init state and store
        store = self.objectstore_factory.store()

        # Delete the message object
        store.delete(request.message_object_id)

        return ConfirmMessageReceivedResponse()
