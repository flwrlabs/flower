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
"""Base error types for API-facing error translation."""


from __future__ import annotations

import json
from enum import IntEnum


class FlowerError(Exception):
    """Base exception for API errors exposed through client-safe responses.

    Parameters
    ----------
    code : int
        Internal numeric error code used to look up the API error contract.
    message : str
        Sensitive diagnostic message intended for server-side logs.
    public_details : str | None
        Optional client-safe details to include in the serialized error payload.
    """

    def __init__(
        self,
        code: int,
        message: str,
        public_details: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message  # Sensitive message
        self.public_details = public_details

    def to_json(self, public_message: str) -> str:
        """Serialize the client-visible error payload as JSON.

        Parameters
        ----------
        public_message : str
            Sanitized message that should be exposed to the client instead of the
            internal diagnostic message.

        Returns
        -------
        str
            A JSON string containing the error code, the client-visible message,
            and any client-safe details attached to the error.
        """
        return json.dumps(
            {
                "code": self.code,
                "public_message": public_message,
                "public_details": self.public_details,
            }
        )

    @staticmethod
    def from_json(value: str | None) -> FlowerError | None:
        """Deserialize a client-visible error payload.

        The internal diagnostic message is not transmitted over the wire. The returned
        error therefore uses the public message as its ``message`` value.
        """
        if value is None:
            return None

        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return None

        if not isinstance(payload, dict):
            return None

        code = payload.get("code")
        public_message = payload.get("public_message")
        public_details = payload.get("public_details")

        if (
            not isinstance(code, int)
            or isinstance(code, bool)
            or not isinstance(public_message, str)
        ):
            return None
        if public_details is not None and not isinstance(public_details, str):
            return None

        return FlowerError(
            code=code,
            message=public_message,
            public_details=public_details,
        )


class ApiErrorCode(IntEnum):
    """API error code."""

    # Control API errors (1-1000)
    NO_FEDERATION_MANAGEMENT_SUPPORT = 1
    FEDERATION_NOT_FOUND_OR_NO_PERMISSION = 2
    ACCOUNT_ALREADY_MEMBER = 3
    FEDERATION_ALREADY_EXISTS = 4
    INVITE_ALREADY_EXISTS = 5
    ACCOUNTS_NOT_FOUND = 6
    FEDERATION_NOT_FOUND_OR_NO_PENDING_INVITE = 7
    ACCOUNT_NOT_A_MEMBER = 8
    NO_PERMISSIONS = 9
    FORBIDDEN_ACTION = 10
    SUPERNODE_ALREADY_IN_FEDERATION = 11
    FEDERATION_NOT_SPECIFIED = 12
    ENTITLEMENT_ERROR = 13
    FAILED_TO_CREATE_RUN = 14
    INVALID_RUN_CONFIG = 15
    RUN_ID_NOT_FOUND = 16
    RUN_SERIES_ID_NOT_FOUND = 17
    RUN_ALREADY_FINISHED = 18
    NO_ACCOUNT_AUTH = 19
    NO_ARTIFACT_PROVIDER = 20
    PULL_UNFINISHED_RUN = 21
    PUBLIC_KEY_NOT_VALID = 22
    PUBLIC_KEY_ALREADY_IN_USE = 23
    NODE_NOT_FOUND = 24
    FEDERATION_NOT_FOUND = 25
    FEDERATION_NOT_FOUND_OR_NOT_MEMBER = 26
    INVALID_FEDERATION_NAME = 27
    NODE_NOT_FOUND_OR_NOT_OWNER = 28
    ACCOUNT_INFO_NOT_FOUND = 29
    RUN_NOT_ASSOCIATED_WITH_ACCOUNT = 30
    RUN_ID_NOT_BELONG_TO_ACCOUNT = 31
    UNSUPPORTED_FAB_HUB_TRANSPORT = 32
    INVALID_APP_SPEC = 33
    FAB_DOWNLOAD_LINK_FAILURE = 34
    FAB_DOWNLOAD_FAILURE = 35
    ACCOUNT_AUTHENTICATION_FAILED = 36
    ACCOUNT_AUTHENTICATION_NOT_INITIALIZED = 37
    INVALID_CONNECTOR_REQUEST = 38
    CONNECTOR_NOT_FOUND = 39
    CONNECTOR_FAILURE = 40
    LICENSE_CHECK_FAILED = 41

    # Common API errors (1001-2000)
    RUNTIME_VERSION_INCOMPATIBLE = 1001
    UNSUPPORTED_CONTENT_TYPE = 1002
    INVALID_PROTOBUF_PAYLOAD = 1003
    INVALID_HANDLER_RESPONSE = 1004
    LINKSTATE_NOT_INITIALIZED = 1005
    INVALID_PROTOBUF_REQUEST = 1006
    INVALID_PROTOBUF_RESPONSE = 1007

    # Fleet API errors (2001-3000)
    FLEET_SUPERNODE_REGISTRATION_DISABLED = 2001
    FLEET_SUPERNODE_UNREGISTRATION_DISABLED = 2002
    FLEET_INVALID_HEARTBEAT_INTERVAL = 2003
    FLEET_NODE_ACTIVATION_FAILED = 2004
    FLEET_NODE_DEACTIVATION_FAILED = 2005
    FLEET_NODE_UNREGISTRATION_FAILED = 2006
    FLEET_RUN_STATUS_NOT_ALLOWED = 2007
    FLEET_GET_RUN_FAILED = 2008
    FLEET_GET_FAB_FAILED = 2009
    # FLEET_OBJECT_CONTENT_INVALID = 2010 --- DELETED ---
