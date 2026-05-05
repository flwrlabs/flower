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
"""Shared gRPC interceptors used across supercore services."""


from .appio_token_interceptor import (
    APP_TOKEN_HEADER,
    AUTHENTICATION_FAILED_MESSAGE,
    AppIoTokenClientInterceptor,
    AppIoTokenServerInterceptor,
    create_clientappio_token_auth_server_interceptor,
    create_serverappio_token_auth_server_interceptor,
    get_authenticated_task_id,
)
from .runtime_version_interceptor import (
    RuntimeVersionClientInterceptor,
    RuntimeVersionServerInterceptor,
    create_serverappio_runtime_version_server_interceptor,
)
from .superexec_auth_interceptor import (
    SuperExecAuthClientInterceptor,
    SuperExecAuthServerInterceptor,
    create_clientappio_superexec_auth_server_interceptor,
    create_serverappio_superexec_auth_server_interceptor,
)

__all__ = [
    "APP_TOKEN_HEADER",
    "AUTHENTICATION_FAILED_MESSAGE",
    "AppIoTokenClientInterceptor",
    "AppIoTokenServerInterceptor",
    "RuntimeVersionClientInterceptor",
    "RuntimeVersionServerInterceptor",
    "SuperExecAuthClientInterceptor",
    "SuperExecAuthServerInterceptor",
    "create_clientappio_superexec_auth_server_interceptor",
    "create_clientappio_token_auth_server_interceptor",
    "create_serverappio_runtime_version_server_interceptor",
    "create_serverappio_superexec_auth_server_interceptor",
    "create_serverappio_token_auth_server_interceptor",
    "get_authenticated_task_id",
]
