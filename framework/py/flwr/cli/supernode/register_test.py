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
"""Tests for the Flower CLI SuperNode register command."""

from unittest.mock import Mock

from flwr.proto.control_pb2 import RegisterNodeResponse  # pylint: disable=E0611

from .register import _register_node


def test_register_node_sends_location() -> None:
    """Send the optional location in the registration request."""
    client = Mock()
    client.RegisterNode.return_value = RegisterNodeResponse(node_id=1)

    _register_node(client, b"public-key", False, "London, UK")

    request = client.RegisterNode.call_args.kwargs["request"]
    assert request.public_key == b"public-key"
    assert request.location == "London, UK"
