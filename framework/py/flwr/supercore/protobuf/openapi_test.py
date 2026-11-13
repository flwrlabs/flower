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
"""Tests for descriptor-derived protobuf documentation."""

import pytest

from flwr.proto.control_pb2 import CreateFederationRequest  # pylint: disable=E0611
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611
from flwr.supercore.protobuf.openapi import protobuf_message_description


def test_reject_unknown_field_notes() -> None:
    """Fail at registration when field notes refer to a removed or mistyped field."""
    with pytest.raises(ValueError, match="Unknown fields.*typo"):
        protobuf_message_description(CreateFederationRequest(), {"typo": "Note"})


def test_descriptor_fields_and_nested_type_links() -> None:
    """Include fields missing from the example and link nested protobuf messages."""
    description = protobuf_message_description(Federation(), {})
    assert "**`nodes`** ([flwr.proto.NodeInfo]" in description
    assert "field 2, repeated presence" in description
    assert "framework/proto/flwr/proto/node.proto" in description
    assert "**`member_count`** (uint32, field 11, explicit presence)" in description
    assert "**`archived`** (bool, field 6, implicit presence)" in description
