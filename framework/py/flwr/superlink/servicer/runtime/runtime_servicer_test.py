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
"""SuperLinkRuntimeServicer tests."""

import unittest
from unittest.mock import Mock, patch

from flwr.proto.runtime_pb2 import GetConnectorRequest  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkState, LinkStateFactory
from flwr.supercore.object_store import ObjectStoreFactory

from .runtime_servicer import SuperLinkRuntimeServicer


class TestSuperLinkRuntimeServicer(unittest.TestCase):
    """Tests for gRPC-specific Runtime servicer behavior."""

    def test_get_connector_authenticates_before_accessing_state(self) -> None:
        """GetConnector should authenticate before accessing state."""
        state_factory = Mock(spec=LinkStateFactory)
        state_factory.state.return_value = Mock(spec=LinkState)
        servicer = SuperLinkRuntimeServicer(
            state_factory,
            Mock(spec=ObjectStoreFactory),
        )

        with (
            patch(
                "flwr.superlink.servicer.runtime.runtime_servicer."
                "get_authenticated_task",
                side_effect=RuntimeError("No authenticated task"),
            ),
            self.assertRaisesRegex(RuntimeError, "No authenticated task"),
        ):
            servicer.GetConnector(GetConnectorRequest(), Mock())

        state_factory.state.assert_not_called()
