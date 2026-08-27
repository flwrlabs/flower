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
"""FastAPI dependency for the configured Fleet API transport type."""

from typing import Annotated, cast

from fastapi import Depends, Request

from flwr.supercore.error import ApiErrorCode, FlowerError


def get_fleet_api_type(request: Request) -> str:
    """Return the configured Fleet API transport type."""
    fleet_api_type = cast(
        str | None, getattr(request.app.state, "fleet_api_type", None)
    )
    if not fleet_api_type:
        raise FlowerError(
            ApiErrorCode.FLEET_API_TYPE_NOT_INITIALIZED,
            "SuperLink Fleet API type is not initialized.",
        )
    return fleet_api_type


FleetApiTypeDependency = Annotated[str, Depends(get_fleet_api_type)]
