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
"""Health API router implementation."""


from fastapi import APIRouter, Request, Response, status
from starlette.datastructures import State

router = APIRouter(tags=["health"])


@router.api_route("/health", methods=["GET", "HEAD"])
async def health(_: Request[State]) -> Response:
    """Report whether the API server is healthy."""
    return Response(status_code=status.HTTP_200_OK)


@router.api_route("/ready", methods=["GET", "HEAD"])
async def ready(_: Request[State]) -> Response:
    """Report whether the API server is ready."""
    return Response(status_code=status.HTTP_200_OK)
