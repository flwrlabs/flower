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

from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from google.protobuf.message import DecodeError, Message
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    AcceptInvitationRequest,
    AddNodeToFederationRequest,
    ArchiveFederationRequest,
    ConfigureSimulationFederationRequest,
    CreateFederationRequest,
    CreateInvitationRequest,
    GetAuthTokensRequest,
    GetLoginDetailsRequest,
    GetRunSeriesRequest,
    ListFederationsRequest,
    ListInvitationsRequest,
    ListNodesRequest,
    ListRunSeriesRequest,
    ListRunsRequest,
    RegisterNodeRequest,
    RejectInvitationRequest,
    RemoveAccountFromFederationRequest,
    RemoveNodeFromFederationRequest,
    RevokeInvitationRequest,
    ShowFederationRequest,
    StartRunRequest,
    StopRunRequest,
    UnregisterNodeRequest,
)
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.protobuf.constants import (
    PROTOBUF_MEDIA_TYPE,
    PROTOBUF_STREAM_MEDIA_TYPE,
)
from flwr.supercore.protobuf.framing import frame_message

RouteKey = tuple[str, str]

PROTOBUF_REQUEST_TYPES: dict[RouteKey, type[Message]] = {
    ("POST", "/control/start-run"): StartRunRequest,
    ("POST", "/control/list-runs"): ListRunsRequest,
    ("POST", "/control/list-run-series"): ListRunSeriesRequest,
    ("POST", "/control/get-run-series"): GetRunSeriesRequest,
    ("POST", "/control/stop-run"): StopRunRequest,
    ("POST", "/control/get-login-details"): GetLoginDetailsRequest,
    ("POST", "/control/get-auth-tokens"): GetAuthTokensRequest,
    ("POST", "/control/register-node"): RegisterNodeRequest,
    ("POST", "/control/unregister-node"): UnregisterNodeRequest,
    ("POST", "/control/list-nodes"): ListNodesRequest,
    ("POST", "/control/list-federations"): ListFederationsRequest,
    ("POST", "/control/show-federation"): ShowFederationRequest,
    ("POST", "/control/create-federation"): CreateFederationRequest,
    ("POST", "/control/archive-federation"): ArchiveFederationRequest,
    ("POST", "/control/add-node-to-federation"): AddNodeToFederationRequest,
    ("POST", "/control/remove-node-from-federation"): RemoveNodeFromFederationRequest,
    (
        "POST",
        "/control/remove-account-from-federation",
    ): RemoveAccountFromFederationRequest,
    ("POST", "/control/create-invitation"): CreateInvitationRequest,
    ("POST", "/control/list-invitations"): ListInvitationsRequest,
    ("POST", "/control/accept-invitation"): AcceptInvitationRequest,
    ("POST", "/control/reject-invitation"): RejectInvitationRequest,
    ("POST", "/control/revoke-invitation"): RevokeInvitationRequest,
    (
        "POST",
        "/control/configure-simulation-federation",
    ): ConfigureSimulationFederationRequest,
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
