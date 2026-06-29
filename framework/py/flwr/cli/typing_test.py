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
"""Flower command line interface typing tests."""

import unittest

from parameterized import parameterized

from flwr.cli.typing import SuperLinkConnection


class SuperLinkConnectionTest(unittest.TestCase):
    """Test SuperLinkConnection validation."""

    def test_post_init_valid_federation(self) -> None:
        """Test valid federation input format."""
        federation = "@my-account/my-federation"
        connection = SuperLinkConnection(
            name="test",
            address="localhost:9092",
            federation=federation,
        )
        self.assertEqual(connection.federation, federation)

    @parameterized.expand(  # type: ignore
        [
            ("my-account/my-federation",),  # Missing @
            ("@my-account-my-federation",),  # Missing /
            ("invalid-format",),  # Completely invalid
            ("@/",),  # Empty account and federation
            ("@/my-federation",),  # Empty account
            ("@my-account/",),  # Empty federation
        ]
    )
    def test_post_init_invalid_federation(self, invalid_federation: str) -> None:
        """Test invalid federation input format."""
        with self.assertRaises(ValueError):
            SuperLinkConnection(
                name="test",
                address="localhost:9092",
                federation=invalid_federation,
            )
