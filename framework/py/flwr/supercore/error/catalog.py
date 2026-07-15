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

from fastapi import status
from grpc import StatusCode

from .base import ApiErrorCode


@dataclass(frozen=True)
class ApiErrorSpec:
    """Public API error contract for external transports."""

    status_code: StatusCode
    http_status_code: int
    public_message: str


API_ERROR_MAP: Final[dict[int, ApiErrorSpec]] = {
    ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT: ApiErrorSpec(
        status_code=StatusCode.UNIMPLEMENTED,
        http_status_code=status.HTTP_501_NOT_IMPLEMENTED,
        public_message="SuperLink does not support federation management.",
    ),
    ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PERMISSION: ApiErrorSpec(
        status_code=StatusCode.NOT_FOUND,
        http_status_code=status.HTTP_404_NOT_FOUND,
        public_message="Federation not found, archived, "
        "or you cannot perform this action.",
    ),
    ApiErrorCode.ACCOUNT_ALREADY_MEMBER: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Account is already a member of the federation.",
    ),
    ApiErrorCode.FEDERATION_ALREADY_EXISTS: ApiErrorSpec(
        status_code=StatusCode.ALREADY_EXISTS,
        http_status_code=status.HTTP_409_CONFLICT,
        public_message="Federation already exists or it has been archived.",
    ),
    ApiErrorCode.INVITE_ALREADY_EXISTS: ApiErrorSpec(
        status_code=StatusCode.ALREADY_EXISTS,
        http_status_code=status.HTTP_409_CONFLICT,
        public_message="A pending invitation already exists for this account "
        "in the federation.",
    ),
    ApiErrorCode.ACCOUNTS_NOT_FOUND: ApiErrorSpec(
        status_code=StatusCode.NOT_FOUND,
        http_status_code=status.HTTP_404_NOT_FOUND,
        public_message="One or more specified accounts were not found.",
    ),
    ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PENDING_INVITE: ApiErrorSpec(
        status_code=StatusCode.NOT_FOUND,
        http_status_code=status.HTTP_404_NOT_FOUND,
        public_message="Federation does not exist, has been archived, "
        "or no pending invitation was found.",
    ),
    ApiErrorCode.ACCOUNT_NOT_A_MEMBER: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Account is not a member of the federation.",
    ),
    ApiErrorCode.NO_PERMISSIONS: ApiErrorSpec(
        status_code=StatusCode.PERMISSION_DENIED,
        http_status_code=status.HTTP_403_FORBIDDEN,
        public_message="You do not have permission to perform this action.",
    ),
    ApiErrorCode.FORBIDDEN_ACTION: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="This action cannot be performed.",
    ),
    ApiErrorCode.SUPERNODE_ALREADY_IN_FEDERATION: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="The SuperNode is already part of the federation.",
    ),
    ApiErrorCode.FEDERATION_NOT_SPECIFIED: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="No federation specified. You must specify a federation to "
        "perform this action.",
    ),
    ApiErrorCode.ENTITLEMENT_ERROR: ApiErrorSpec(
        status_code=StatusCode.PERMISSION_DENIED,
        http_status_code=status.HTTP_403_FORBIDDEN,
        public_message="Entitlement error.",
    ),
    ApiErrorCode.FAILED_TO_CREATE_RUN: ApiErrorSpec(
        status_code=StatusCode.INTERNAL,
        http_status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        public_message="Failed to create or initialize the run.",
    ),
    ApiErrorCode.INVALID_RUN_CONFIG: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Invalid run configuration.",
    ),
    ApiErrorCode.RUN_ID_NOT_FOUND: ApiErrorSpec(
        status_code=StatusCode.NOT_FOUND,
        http_status_code=status.HTTP_404_NOT_FOUND,
        public_message="Run ID not found.",
    ),
    ApiErrorCode.RUN_SERIES_ID_NOT_FOUND: ApiErrorSpec(
        status_code=StatusCode.NOT_FOUND,
        http_status_code=status.HTTP_404_NOT_FOUND,
        public_message="Run series ID not found.",
    ),
    ApiErrorCode.RUN_ALREADY_FINISHED: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Run is already finished.",
    ),
    ApiErrorCode.NO_ACCOUNT_AUTH: ApiErrorSpec(
        status_code=StatusCode.UNIMPLEMENTED,
        http_status_code=status.HTTP_501_NOT_IMPLEMENTED,
        public_message="ControlServicer initialized without account authentication.",
    ),
    ApiErrorCode.NO_ARTIFACT_PROVIDER: ApiErrorSpec(
        status_code=StatusCode.UNIMPLEMENTED,
        http_status_code=status.HTTP_501_NOT_IMPLEMENTED,
        public_message="ControlServicer initialized without artifact provider.",
    ),
    ApiErrorCode.PULL_UNFINISHED_RUN: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Cannot pull artifacts for an unfinished run.",
    ),
    ApiErrorCode.PUBLIC_KEY_NOT_VALID: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="The provided public key is not valid.",
    ),
    ApiErrorCode.PUBLIC_KEY_ALREADY_IN_USE: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Public key already in use.",
    ),
    ApiErrorCode.NODE_NOT_FOUND: ApiErrorSpec(
        status_code=StatusCode.NOT_FOUND,
        http_status_code=status.HTTP_404_NOT_FOUND,
        public_message="Node ID not found for account.",
    ),
    ApiErrorCode.FEDERATION_NOT_FOUND: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Federation does not exist.",
    ),
    ApiErrorCode.FEDERATION_NOT_FOUND_OR_NOT_MEMBER: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Federation does not exist or you are not a member of it.",
    ),
    ApiErrorCode.INVALID_FEDERATION_NAME: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Invalid federation name.",
    ),
    ApiErrorCode.NODE_NOT_FOUND_OR_NOT_OWNER: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Node not found or you are not its owner.",
    ),
    ApiErrorCode.ACCOUNT_INFO_NOT_FOUND: ApiErrorSpec(
        status_code=StatusCode.PERMISSION_DENIED,
        http_status_code=status.HTTP_403_FORBIDDEN,
        public_message="Failed to fetch the account information.",
    ),
    ApiErrorCode.RUN_NOT_ASSOCIATED_WITH_ACCOUNT: ApiErrorSpec(
        status_code=StatusCode.PERMISSION_DENIED,
        http_status_code=status.HTTP_403_FORBIDDEN,
        public_message="Run is not associated with an account.",
    ),
    ApiErrorCode.RUN_ID_NOT_BELONG_TO_ACCOUNT: ApiErrorSpec(
        status_code=StatusCode.PERMISSION_DENIED,
        http_status_code=status.HTTP_403_FORBIDDEN,
        public_message="Run ID does not belong to the account.",
    ),
    ApiErrorCode.UNSUPPORTED_FAB_HUB_TRANSPORT: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="The selected SuperLink transport type is not supported for "
        "connecting to Flower Hub.",
    ),
    ApiErrorCode.INVALID_APP_SPEC: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Invalid app specification.",
    ),
    ApiErrorCode.FAB_DOWNLOAD_LINK_FAILURE: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Failed to request FAB download link.",
    ),
    ApiErrorCode.FAB_DOWNLOAD_FAILURE: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="FAB download failed.",
    ),
    ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED: ApiErrorSpec(
        status_code=StatusCode.UNAUTHENTICATED,
        http_status_code=status.HTTP_401_UNAUTHORIZED,
        public_message="Authentication failed.",
    ),
    ApiErrorCode.ACCOUNT_AUTHENTICATION_NOT_INITIALIZED: ApiErrorSpec(
        status_code=StatusCode.UNAVAILABLE,
        http_status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        public_message="Authentication is not initialized.",
    ),
    ApiErrorCode.RUNTIME_VERSION_INCOMPATIBLE: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Runtime version compatibility check failed.",
    ),
    ApiErrorCode.UNSUPPORTED_CONTENT_TYPE: ApiErrorSpec(
        status_code=StatusCode.INVALID_ARGUMENT,
        http_status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        public_message="Unsupported Content-Type.",
    ),
    ApiErrorCode.INVALID_PROTOBUF_PAYLOAD: ApiErrorSpec(
        status_code=StatusCode.INVALID_ARGUMENT,
        http_status_code=status.HTTP_400_BAD_REQUEST,
        public_message="Invalid protobuf payload.",
    ),
    ApiErrorCode.INVALID_HANDLER_RESPONSE: ApiErrorSpec(
        status_code=StatusCode.INTERNAL,
        http_status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        public_message="Invalid response returned from handler.",
    ),
    ApiErrorCode.LINKSTATE_NOT_INITIALIZED: ApiErrorSpec(
        status_code=StatusCode.UNAVAILABLE,
        http_status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        public_message="LinkState is not initialized.",
    ),
    ApiErrorCode.FLEET_SUPERNODE_REGISTRATION_DISABLED: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="SuperNode authentication is enabled. "
        "All SuperNodes must be registered via the CLI.",
    ),
    ApiErrorCode.FLEET_SUPERNODE_UNREGISTRATION_DISABLED: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="SuperNode authentication is enabled. "
        "All SuperNodes must be unregistered via the CLI.",
    ),
    ApiErrorCode.FLEET_INVALID_HEARTBEAT_INTERVAL: ApiErrorSpec(
        status_code=StatusCode.INVALID_ARGUMENT,
        http_status_code=status.HTTP_400_BAD_REQUEST,
        public_message="Invalid heartbeat interval.",
    ),
    ApiErrorCode.FLEET_NODE_ACTIVATION_FAILED: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Failed to activate SuperNode.",
    ),
    ApiErrorCode.FLEET_NODE_DEACTIVATION_FAILED: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Failed to deactivate SuperNode.",
    ),
    ApiErrorCode.FLEET_NODE_UNREGISTRATION_FAILED: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Failed to unregister SuperNode.",
    ),
    ApiErrorCode.FLEET_RUN_STATUS_NOT_ALLOWED: ApiErrorSpec(
        status_code=StatusCode.PERMISSION_DENIED,
        http_status_code=status.HTTP_403_FORBIDDEN,
        public_message="The current run status does not allow this Fleet API "
        "operation.",
    ),
    ApiErrorCode.FLEET_GET_RUN_FAILED: ApiErrorSpec(
        status_code=StatusCode.PERMISSION_DENIED,
        http_status_code=status.HTTP_403_FORBIDDEN,
        public_message="Failed to get run.",
    ),
    ApiErrorCode.FLEET_GET_FAB_FAILED: ApiErrorSpec(
        status_code=StatusCode.PERMISSION_DENIED,
        http_status_code=status.HTTP_403_FORBIDDEN,
        public_message="Failed to get FAB.",
    ),
    ApiErrorCode.FLEET_OBJECT_CONTENT_INVALID: ApiErrorSpec(
        status_code=StatusCode.FAILED_PRECONDITION,
        http_status_code=status.HTTP_412_PRECONDITION_FAILED,
        public_message="Object content is invalid.",
    ),
}
