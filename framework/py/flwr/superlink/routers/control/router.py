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
"""Control API router."""

from typing import Annotated

from fastapi import APIRouter, Depends

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    ListRunsRequest,
    ListRunsResponse,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.protobuf.routing import ProtobufRouter
from flwr.superlink.dependencies.account import get_account
from flwr.superlink.dependencies.linkstate import get_linkstate
from flwr.superlink.servicer.control import control_handlers

router = APIRouter(prefix="/control", tags=["control"])
protobuf_router = ProtobufRouter(router)


@protobuf_router.unary_unary("/rpc/ListRuns")
def list_runs(
    request: ListRunsRequest,
    linkstate: Annotated[LinkState, Depends(get_linkstate)],
    account: Annotated[AccountInfo, Depends(get_account)],
) -> ListRunsResponse:
    """List runs.

    Parameters
    ----------
    request : ListRunsRequest
        Filters for the requested runs.
    linkstate : LinkState
        State used to retrieve runs.
    account : AccountInfo
        Authenticated account making the request.

    Returns
    -------
    ListRunsResponse
        Runs that match the requested filters.
    """
    return control_handlers.list_runs(request, account, linkstate)
