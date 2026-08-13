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
"""Shared client and server interceptors used across supercore services."""

from flwr.supercore.constant import TASK_TOKEN_HEADER

from .http import RuntimeVersionHttpInterceptor, SuperExecAuthHttpInterceptor
from .rpc_error_translation_interceptor import RpcErrorTranslationServerInterceptor
from .runtime_token_interceptor import (
    AUTHENTICATION_FAILED_MESSAGE,
    RuntimeTokenClientInterceptor,
    RuntimeTokenServerInterceptor,
    create_superlink_runtime_token_auth_server_interceptor,
    create_supernode_runtime_token_auth_server_interceptor,
    get_authenticated_task,
)
from .runtime_version_interceptor import (
    RuntimeVersionClientInterceptor,
    RuntimeVersionServerInterceptor,
    create_control_runtime_version_server_interceptor,
    create_fleet_runtime_version_server_interceptor,
    create_superlink_runtime_version_server_interceptor,
    create_supernode_runtime_version_server_interceptor,
)
from .superexec_auth_interceptor import (
    SuperExecAuthClientInterceptor,
    SuperExecAuthServerInterceptor,
    create_superlink_runtime_superexec_auth_server_interceptor,
    create_supernode_runtime_superexec_auth_server_interceptor,
)

__all__ = [
    "AUTHENTICATION_FAILED_MESSAGE",
    "RpcErrorTranslationServerInterceptor",
    "RuntimeTokenClientInterceptor",
    "RuntimeTokenServerInterceptor",
    "RuntimeVersionClientInterceptor",
    "RuntimeVersionHttpInterceptor",
    "RuntimeVersionServerInterceptor",
    "SuperExecAuthClientInterceptor",
    "SuperExecAuthHttpInterceptor",
    "SuperExecAuthServerInterceptor",
    "TASK_TOKEN_HEADER",
    "create_control_runtime_version_server_interceptor",
    "create_fleet_runtime_version_server_interceptor",
    "create_superlink_runtime_superexec_auth_server_interceptor",
    "create_superlink_runtime_token_auth_server_interceptor",
    "create_superlink_runtime_version_server_interceptor",
    "create_supernode_runtime_superexec_auth_server_interceptor",
    "create_supernode_runtime_token_auth_server_interceptor",
    "create_supernode_runtime_version_server_interceptor",
    "get_authenticated_task",
]
