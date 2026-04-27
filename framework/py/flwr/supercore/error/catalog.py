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
"""Error catalog for translating internal API error codes to public responses."""


from dataclasses import dataclass
from typing import Final

from grpc import StatusCode

from .base import ApiErrorCode


@dataclass(frozen=True)
class ApiErrorSpec:
    """Public API error contract for external transports."""

    status_code: StatusCode
    public_message: str


API_ERROR_MAP: Final[dict[int, ApiErrorSpec]] = {
    ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT: ApiErrorSpec(
        status_code=StatusCode.UNIMPLEMENTED,
        public_message="SuperLink does not support federation management.",
    ),
    ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PERMISSION: ApiErrorSpec(
        status_code=StatusCode.NOT_FOUND,
        public_message="Federation not found, archived, "
        "or you cannot perform this action.",
    ),
    ApiErrorCode.ACCOUNT_ALREADY_MEMBER: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        public_message="Account is already a member of the federation.",
    ),
    ApiErrorCode.FEDERATION_ALREADY_EXISTS: ApiErrorSpec(
        status_code=StatusCode.ALREADY_EXISTS,
        public_message="Federation already exists or it has been archived.",
    ),
    ApiErrorCode.INVITE_ALREADY_EXISTS: ApiErrorSpec(
        status_code=StatusCode.ALREADY_EXISTS,
        public_message="A pending invitation already exists for this account "
        "in the federation.",
    ),
    ApiErrorCode.ACCOUNTS_NOT_FOUND: ApiErrorSpec(
        status_code=StatusCode.NOT_FOUND,
        public_message="One or more specified accounts were not found.",
    ),
    ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PENDING_INVITE: ApiErrorSpec(
        status_code=StatusCode.NOT_FOUND,
        public_message="Federation does not exist, has been archived, "
        "or no pending invitation was found.",
    ),
    ApiErrorCode.ACCOUNT_NOT_A_MEMBER: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        public_message="Account is not a member of the federation.",
    ),
    ApiErrorCode.NO_PERMISSIONS: ApiErrorSpec(
        status_code=StatusCode.PERMISSION_DENIED,
        public_message="You do not have permission to perform this action.",
    ),
    ApiErrorCode.FORBIDDEN_ACTION: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        public_message="This action cannot be performed.",
    ),
    ApiErrorCode.SUPERNODE_ALREADY_IN_FEDERATION: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        public_message="The SuperNode is already part of the federation.",
    ),
    ApiErrorCode.FEDERATION_NOT_SPECIFIED: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        public_message="No federation specified. You must specify a federation to "
        "perform this action.",
    ),
    ApiErrorCode.ENTITLEMENT_ERROR: ApiErrorSpec(
        status_code=StatusCode.PERMISSION_DENIED,
        public_message="Entitlement error.",
    ),
}
