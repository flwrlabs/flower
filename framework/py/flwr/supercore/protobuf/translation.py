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
"""FastAPI translation helpers for protobuf RPC APIs."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from fastapi import Depends, Request
from fastapi.responses import Response, StreamingResponse
from google.protobuf.message import DecodeError, Message
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AcceptInvitationRequest,
    AddAppRequest,
    AddNodeToFederationRequest,
    ArchiveFederationRequest,
    ConfigureSimulationFederationRequest,
    CreateFederationRequest,
    CreateInvitationRequest,
    GetRunSeriesRequest,
    ListAppsRequest,
    ListAutomationsRequest,
    ListFederationsRequest,
    ListInvitationsRequest,
    ListNodesRequest,
    ListRunSeriesRequest,
    ListRunsRequest,
    RegisterNodeRequest,
    RejectInvitationRequest,
    RemoveAccountFromFederationRequest,
    RemoveAppRequest,
    RemoveNodeFromFederationRequest,
    RevokeInvitationRequest,
    ShowFederationRequest,
    StartAutomationRequest,
    StartRunRequest,
    StopAutomationRequest,
    StopRunRequest,
    UnregisterNodeRequest,
)
from flwr.proto.log_pb2 import PushLogsRequest  # pylint: disable=E0611
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    PullObjectRequest,
    PushObjectRequest,
)
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    CreateTaskRequest,
    GetConnectorRequest,
    GetNodesRequest,
    PullAppMessagesRequest,
    PullPendingTasksRequest,
    PullTaskInputRequest,
    PullTaskMessageRequest,
    PushAppMessagesRequest,
    PushTaskEventsRequest,
    PushTaskMessageRequest,
    PushTaskOutputRequest,
    RecordTaskUsageRequest,
    SendTaskHeartbeatRequest,
)
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.protobuf.constants import (
    PROTOBUF_MEDIA_TYPE,
    PROTOBUF_STREAM_MEDIA_TYPE,
)
from flwr.supercore.protobuf.framing import frame_message

RouteKey = tuple[str, str]

PROTOBUF_REQUEST_TYPES: dict[RouteKey, type[Message]] = {
    ("POST", "/v1/control/start-run"): StartRunRequest,
    ("POST", "/v1/control/list-runs"): ListRunsRequest,
    ("POST", "/v1/control/list-run-series"): ListRunSeriesRequest,
    ("POST", "/v1/control/get-run-series"): GetRunSeriesRequest,
    ("POST", "/v1/control/stop-run"): StopRunRequest,
    ("POST", "/v1/control/start-automation"): StartAutomationRequest,
    ("POST", "/v1/control/list-automations"): ListAutomationsRequest,
    ("POST", "/v1/control/stop-automation"): StopAutomationRequest,
    ("POST", "/v1/control/register-node"): RegisterNodeRequest,
    ("POST", "/v1/control/unregister-node"): UnregisterNodeRequest,
    ("POST", "/v1/control/list-nodes"): ListNodesRequest,
    ("POST", "/v1/control/list-apps"): ListAppsRequest,
    ("POST", "/v1/control/add-app"): AddAppRequest,
    ("POST", "/v1/control/remove-app"): RemoveAppRequest,
    ("POST", "/v1/control/list-federations"): ListFederationsRequest,
    ("POST", "/v1/control/show-federation"): ShowFederationRequest,
    ("POST", "/v1/control/create-federation"): CreateFederationRequest,
    ("POST", "/v1/control/archive-federation"): ArchiveFederationRequest,
    ("POST", "/v1/control/add-node-to-federation"): AddNodeToFederationRequest,
    (
        "POST",
        "/v1/control/remove-node-from-federation",
    ): RemoveNodeFromFederationRequest,
    (
        "POST",
        "/v1/control/remove-account-from-federation",
    ): RemoveAccountFromFederationRequest,
    ("POST", "/v1/control/create-invitation"): CreateInvitationRequest,
    ("POST", "/v1/control/list-invitations"): ListInvitationsRequest,
    ("POST", "/v1/control/accept-invitation"): AcceptInvitationRequest,
    ("POST", "/v1/control/reject-invitation"): RejectInvitationRequest,
    ("POST", "/v1/control/revoke-invitation"): RevokeInvitationRequest,
    (
        "POST",
        "/v1/control/configure-simulation-federation",
    ): ConfigureSimulationFederationRequest,
    ("POST", "/v1/runtime/pull-pending-tasks"): PullPendingTasksRequest,
    ("POST", "/v1/runtime/claim-task"): ClaimTaskRequest,
    ("POST", "/v1/runtime/send-task-heartbeat"): SendTaskHeartbeatRequest,
    ("POST", "/v1/runtime/pull-task-input"): PullTaskInputRequest,
    ("POST", "/v1/runtime/push-task-output"): PushTaskOutputRequest,
    ("POST", "/v1/runtime/push-object"): PushObjectRequest,
    ("POST", "/v1/runtime/pull-object"): PullObjectRequest,
    (
        "POST",
        "/v1/runtime/confirm-message-received",
    ): ConfirmMessageReceivedRequest,
    ("POST", "/v1/runtime/create-task"): CreateTaskRequest,
    ("POST", "/v1/runtime/start-automation"): StartAutomationRequest,
    ("POST", "/v1/runtime/push-task-message"): PushTaskMessageRequest,
    ("POST", "/v1/runtime/push-task-events"): PushTaskEventsRequest,
    ("POST", "/v1/runtime/pull-task-message"): PullTaskMessageRequest,
    ("POST", "/v1/runtime/record-task-usage"): RecordTaskUsageRequest,
    ("POST", "/v1/runtime/get-connector"): GetConnectorRequest,
    ("POST", "/v1/runtime/push-logs"): PushLogsRequest,
    ("POST", "/v1/runtime/push-messages"): PushAppMessagesRequest,
    ("POST", "/v1/runtime/pull-messages"): PullAppMessagesRequest,
    ("POST", "/v1/runtime/get-nodes"): GetNodesRequest,
}


class ProtobufTranslationMiddleware(BaseHTTPMiddleware):
    """Translate protobuf requests and handler results at the HTTP boundary."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Parse the protobuf request and serialize the protobuf handler result."""
        request_type = PROTOBUF_REQUEST_TYPES.get((request.method, request.url.path))
        if request_type is not None:
            self._check_request_media_type(request)
            request.state.protobuf_request = self._parse_request(
                await request.body(), request_type
            )
        else:
            # Continue for unrecognized requests
            return await call_next(request)
        response = await call_next(request)

        if not hasattr(request.state, "protobuf_response"):
            raise FlowerError(
                ApiErrorCode.INVALID_PROTOBUF_RESPONSE,
                "Protobuf response missing from request state after handler completed.",
            )

        result = request.state.protobuf_response
        protobuf_response = self._response_for(result)
        del request.state.protobuf_response
        # Preserve metadata set by inner middleware, but not placeholder body headers.
        protobuf_response.status_code = response.status_code
        protobuf_response.headers.raw.extend(
            header
            for header in response.headers.raw
            if header[0] not in (b"content-length", b"content-type")
        )
        return protobuf_response

    @staticmethod
    def _check_request_media_type(request: Request) -> None:
        content_type = request.headers.get("content-type", "")
        media_type = content_type.partition(";")[0].strip().lower()
        if media_type != PROTOBUF_MEDIA_TYPE:
            raise FlowerError(
                ApiErrorCode.UNSUPPORTED_CONTENT_TYPE,
                f"Unsupported Content-Type: {content_type!r}",
            )

    @staticmethod
    def _parse_request(body: bytes, request_type: type[Message]) -> Message:
        message = request_type()
        try:
            message.ParseFromString(body)
        except DecodeError as exc:
            raise FlowerError(
                ApiErrorCode.INVALID_PROTOBUF_PAYLOAD,
                f"Invalid protobuf payload: {exc!r}",
            ) from exc
        return message

    @staticmethod
    def _response_for(result: object) -> Response:
        """Return the HTTP response matching a protobuf handler result."""
        # ``Message`` is also the most specific contract and must be checked
        # first. Unary responses are not framed; framing is reserved for streams.
        if isinstance(result, Message):
            return Response(
                content=result.SerializeToString(), media_type=PROTOBUF_MEDIA_TYPE
            )

        # Synchronous generators and other iterables are streamed lazily too.
        # Starlette advances a synchronous iterator outside the event loop.
        if isinstance(result, Iterable):
            return StreamingResponse(
                (frame_message(message) for message in cast(Iterable[Message], result)),
                media_type=PROTOBUF_STREAM_MEDIA_TYPE,
            )

        raise FlowerError(
            ApiErrorCode.INVALID_HANDLER_RESPONSE,
            "Invalid response returned from Control handler: expected a protobuf "
            "Message or Iterable[Message], got "
            f"{result!r} ({type(result).__name__})",
        )


def get_protobuf_request(request: Request) -> Message:
    """Return the protobuf request parsed by ``ProtobufTranslationMiddleware``."""
    protobuf_request = getattr(request.state, "protobuf_request", None)
    if not isinstance(protobuf_request, Message):
        raise FlowerError(
            ApiErrorCode.INVALID_PROTOBUF_REQUEST,
            "Invalid protobuf request in request state: expected a protobuf "
            f"Message, got {type(protobuf_request).__name__}.",
        )
    return protobuf_request


PROTOBUF_REQUEST_DEPENDENCY = Depends(get_protobuf_request)
