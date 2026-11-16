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
"""SuperNode authentication interceptor for protobuf-over-HTTP clients."""

from base64 import b64encode

import httpx
from cryptography.hazmat.primitives.asymmetric import ec

from flwr.common.constant import TIMESTAMP_HEADER
from flwr.supercore.constant import (
    FLEET_HTTP_PUBLIC_KEY_HEADER,
    FLEET_HTTP_SIGNATURE_HEADER,
)
from flwr.supercore.date import now
from flwr.supercore.interceptors.http.utils import add_headers
from flwr.supercore.primitives.asymmetric import public_key_to_bytes, sign_message
from flwr.supercore.protobuf.client import ProtobufCall, ProtobufRequestContext


class NodeAuthHttpInterceptor:
    """Attach signed SuperNode identity headers to Fleet HTTP requests."""

    def __init__(
        self,
        private_key: ec.EllipticCurvePrivateKey,
        public_key: ec.EllipticCurvePublicKey,
    ) -> None:
        self._private_key = private_key
        self._public_key_bytes = public_key_to_bytes(public_key)

    def intercept(
        self,
        context: ProtobufRequestContext,
        call_next: ProtobufCall,
    ) -> httpx.Response:
        """Sign the current timestamp and send the node's public key."""
        timestamp = now().isoformat()
        signature = sign_message(self._private_key, timestamp.encode("ascii"))
        add_headers(
            context.request,
            {
                FLEET_HTTP_PUBLIC_KEY_HEADER: b64encode(self._public_key_bytes).decode(
                    "ascii"
                ),
                TIMESTAMP_HEADER: timestamp,
                FLEET_HTTP_SIGNATURE_HEADER: b64encode(signature).decode("ascii"),
            },
        )
        return call_next(context)
