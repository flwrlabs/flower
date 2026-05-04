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
"""CoreState utils tests."""


import unittest

from .utils import generate_rand_int_from_bytes


class UtilsTest(unittest.TestCase):
    """Test CoreState utils code."""

    def test_generate_rand_int_from_bytes_unsigned_int(self) -> None:
        """Test that the generated integer is unsigned (non-negative)."""
        for num_bytes in range(1, 9):
            with self.subTest(num_bytes=num_bytes):
                rand_int = generate_rand_int_from_bytes(num_bytes)
                self.assertGreaterEqual(rand_int, 0)
