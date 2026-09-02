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
"""Tests for CLI account authentication interceptors."""

from unittest.mock import Mock

import httpx

from flwr.common.constant import ACCESS_TOKEN_KEY
from flwr.proto.control_pb2 import ListFederationsRequest  # pylint: disable=E0611
from flwr.supercore.protobuf.client import ProtobufRequestContext

from .auth_plugin import CliAuthPlugin
from .cli_account_auth_interceptor import CliAccountAuthHttpInterceptor


def test_http_interceptor_adds_bearer_token() -> None:
    """Send the stored access token in the Authorization header."""
    auth_plugin = Mock(spec=CliAuthPlugin)
    auth_plugin.write_tokens_to_metadata.return_value = [
        (ACCESS_TOKEN_KEY, "access-token")
    ]
    context = ProtobufRequestContext(
        rpc_method="/flwr.proto.Control/ListFederations",
        message=ListFederationsRequest(),
        request=httpx.Request("POST", "https://control.example"),
    )
    response = httpx.Response(200)
    call_next = Mock(return_value=response)

    result = CliAccountAuthHttpInterceptor(auth_plugin).intercept(context, call_next)

    assert result is response
    assert context.request.headers["Authorization"] == "Bearer access-token"
    call_next.assert_called_once_with(context)
