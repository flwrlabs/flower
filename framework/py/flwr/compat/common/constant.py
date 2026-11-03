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
"""Flower constants."""

from flwr.common.constant import TRANSPORT_TYPE_VCE

GRPC_MAX_MESSAGE_LENGTH: int = 2_147_483_647  # 2GB - 1
FLEET_API_GRPC_BIDI_DEFAULT_ADDRESS = "[::]:8080"
TRANSPORT_TYPE_GRPC_BIDI = "grpc-bidi"
TRANSPORT_TYPE_GRPC_RERE = "grpc-rere"
TRANSPORT_TYPES_COMPAT = [
    TRANSPORT_TYPE_GRPC_BIDI,
    TRANSPORT_TYPE_GRPC_RERE,
    TRANSPORT_TYPE_VCE,
]
