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
