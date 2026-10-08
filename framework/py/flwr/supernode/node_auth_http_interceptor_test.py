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
"""Tests for the Fleet node authentication HTTP interceptor."""

from base64 import b64decode
from unittest.mock import Mock

import httpx
import pytest

from flwr.common.constant import TIMESTAMP_HEADER
from flwr.proto.fleet_pb2 import ActivateNodeRequest  # pylint: disable=E0611
from flwr.supercore.constant import (
    FLEET_HTTP_PUBLIC_KEY_HEADER,
    FLEET_HTTP_SIGNATURE_HEADER,
)
from flwr.supercore.primitives.asymmetric import (
    generate_key_pairs,
    public_key_to_bytes,
    verify_signature,
)
from flwr.supercore.protobuf.client import ProtobufRequestContext

from .node_auth_http_interceptor import NodeAuthHttpInterceptor


def test_signs_fleet_request() -> None:
    """Send a verifiable signature and public key on Fleet HTTP requests."""
    private_key, public_key = generate_key_pairs()
    context = ProtobufRequestContext(
        rpc_method="/flwr.proto.Fleet/ActivateNode",
        message=ActivateNodeRequest(public_key=public_key_to_bytes(public_key)),
        request=httpx.Request("POST", "http://fleet.example/v1/fleet/activate-node"),
    )
    response = httpx.Response(200)
    call_next = Mock(return_value=response)

    assert (
        NodeAuthHttpInterceptor(private_key, public_key).intercept(context, call_next)
        is response
    )

    headers = context.request.headers
    assert b64decode(headers[FLEET_HTTP_PUBLIC_KEY_HEADER]) == public_key_to_bytes(
        public_key
    )
    assert verify_signature(
        public_key,
        headers[TIMESTAMP_HEADER].encode("ascii"),
        b64decode(headers[FLEET_HTTP_SIGNATURE_HEADER]),
    )
    call_next.assert_called_once_with(context)


def test_rejects_duplicate_auth_headers() -> None:
    """Reject ambiguous authentication headers."""
    private_key, public_key = generate_key_pairs()
    context = ProtobufRequestContext(
        rpc_method="/flwr.proto.Fleet/ActivateNode",
        message=ActivateNodeRequest(),
        request=httpx.Request(
            "POST",
            "http://fleet.example/v1/fleet/activate-node",
            headers={FLEET_HTTP_SIGNATURE_HEADER: "existing"},
        ),
    )

    with pytest.raises(RuntimeError, match=FLEET_HTTP_SIGNATURE_HEADER):
        NodeAuthHttpInterceptor(private_key, public_key).intercept(
            context, Mock(return_value=httpx.Response(200))
        )
