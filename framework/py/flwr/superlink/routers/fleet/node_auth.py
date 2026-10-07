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
"""Authenticate SuperNodes on Fleet HTTP routes."""

import binascii
import datetime
from base64 import b64decode
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from flwr.common.constant import (
    PUBLIC_KEY_HEADER,
    SIGNATURE_HEADER,
    SYSTEM_TIME_TOLERANCE,
    TIMESTAMP_HEADER,
    TIMESTAMP_TOLERANCE,
)
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    ActivateNodeRequest,
    RegisterNodeFleetRequest,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.date import now
from flwr.supercore.primitives.asymmetric import bytes_to_public_key, verify_signature
from flwr.superlink.dependencies.linkstate import get_linkstate


def authenticate_node(
    request: Request,
    state: Annotated[LinkState, Depends(get_linkstate)],
) -> None:
    """Validate the signed timestamp and claimed node identity."""
    try:
        # gRPC binary metadata is carried as base64 in HTTP header values.
        public_key = b64decode(request.headers[PUBLIC_KEY_HEADER], validate=True)
        signature = b64decode(request.headers[SIGNATURE_HEADER], validate=True)
        timestamp = request.headers[TIMESTAMP_HEADER]
        signed_at = datetime.datetime.fromisoformat(timestamp)
        age = (now() - signed_at).total_seconds()
        if (
            not -SYSTEM_TIME_TOLERANCE
            < age
            < TIMESTAMP_TOLERANCE + SYSTEM_TIME_TOLERANCE
        ):
            raise ValueError("Timestamp outside the allowed window")
        if not verify_signature(
            bytes_to_public_key(public_key), timestamp.encode("ascii"), signature
        ):
            raise ValueError("Invalid signature")

        protobuf_request = request.state.protobuf_request
        if isinstance(
            protobuf_request, (RegisterNodeFleetRequest, ActivateNodeRequest)
        ):
            if protobuf_request.public_key != public_key:
                raise ValueError("Public key does not match request")
        else:
            node_id = (
                protobuf_request.node.node_id
                if hasattr(protobuf_request, "node")
                else protobuf_request.node_id
            )
            if state.get_node_id_by_public_key(public_key) != node_id:
                raise ValueError("Invalid node ID")
    except (KeyError, ValueError, TypeError, UnicodeError, binascii.Error) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid SuperNode authentication",
        ) from exc
