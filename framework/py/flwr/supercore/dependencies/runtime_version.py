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
"""FastAPI dependency for Flower Runtime version compatibility."""

from fastapi import Request

from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.runtime_version_compatibility import RuntimeVersionMetadata


class RuntimeVersionDependency:
    """Validate peer Runtime version metadata on HTTP requests."""

    def __init__(self, *, component_name: str, connection_name: str) -> None:
        self._connection_name = connection_name
        self._local_metadata = RuntimeVersionMetadata.from_local_component(
            component_name
        )

    def __call__(self, request: Request) -> None:
        """Reject malformed or incompatible peer Runtime metadata."""
        peer_metadata, incompat_details = RuntimeVersionMetadata.from_metadata(
            tuple(request.headers.items())
        )
        if incompat_details is None:
            incompat_details = self._local_metadata.check_compatibility(peer_metadata)
        if incompat_details is None:
            return

        raise FlowerError(
            ApiErrorCode.RUNTIME_VERSION_INCOMPATIBLE,
            (
                "Runtime version compatibility check failed for "
                f"{self._connection_name}. {incompat_details}"
            ),
            public_details=incompat_details,
        )
