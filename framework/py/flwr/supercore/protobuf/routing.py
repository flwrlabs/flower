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
"""FastAPI request translation helpers for protobuf RPC APIs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from fastapi import Request
from fastapi.responses import Response
from google.protobuf.message import DecodeError, Message
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

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
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE

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
    """Deserialize configured protobuf request bodies before handlers run."""

    def __init__(
        self,
        app: ASGIApp,
        request_types: Mapping[RouteKey, type[Message]],
    ) -> None:
        super().__init__(app)
        self._request_types = request_types

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint  # type: ignore[type-arg]
    ) -> Response:
        """Parse the protobuf request body and make it available to dependencies."""
        request_type = self._request_types.get((request.method, request.url.path))
        if request_type is not None:
            self._check_request_media_type(request)
            request.state.protobuf_request = self._parse_request(
                await request.body(), request_type
            )
        return await call_next(request)

    @staticmethod
    def _check_request_media_type(request: Request) -> None:  # type: ignore[type-arg]
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


def get_protobuf_request(request: Request) -> Message:  # type: ignore[type-arg]
    """Return the protobuf request parsed by ``ProtobufTranslationMiddleware``."""
    return cast(Message, request.state.protobuf_request)
