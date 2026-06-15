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
"""Flower client interceptor."""


from collections.abc import Callable
from typing import Any

import grpc
from cryptography.hazmat.primitives.asymmetric import ec
from google.protobuf.message import Message as GrpcMessage

from flwr.common.constant import PUBLIC_KEY_HEADER, SIGNATURE_HEADER, TIMESTAMP_HEADER
from flwr.supercore.date import now
from flwr.supercore.primitives.asymmetric import public_key_to_bytes, sign_message


class NodeAuthClientInterceptor(grpc.UnaryUnaryClientInterceptor):  # type: ignore
    """Client interceptor for node authentication."""

    def __init__(
        self,
        private_key: ec.EllipticCurvePrivateKey,
        public_key: ec.EllipticCurvePublicKey,
    ):
        self.private_key = private_key
        self.public_key_bytes = public_key_to_bytes(public_key)

    def intercept_unary_unary(
        self,
        continuation: Callable[[Any, Any], Any],
        client_call_details: grpc.ClientCallDetails,
        request: GrpcMessage,
    ) -> grpc.Call:
        """Flower client interceptor.

        Intercept unary call from client and add necessary authentication header in the
        RPC metadata.
        """
        metadata = list(client_call_details.metadata or [])

        # Add the public key
        metadata.append((PUBLIC_KEY_HEADER, self.public_key_bytes))

        # Add timestamp
        timestamp = now().isoformat()
        metadata.append((TIMESTAMP_HEADER, timestamp))

        # Sign and add the signature
        signature = sign_message(self.private_key, timestamp.encode("ascii"))
        metadata.append((SIGNATURE_HEADER, signature))

        # Overwrite the metadata
        details = client_call_details._replace(metadata=metadata)

        return continuation(details, request)
