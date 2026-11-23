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
"""Tests for model quantization."""


import unittest

import numpy as np

from .quantization import dequantize, quantize

CLIPPING_RANGE = 8.0
TARGET_RANGE = 1 << 22


class TestQuantize(unittest.TestCase):
    """Test stochastic quantization."""

    def setUp(self) -> None:
        """Create the input array."""
        self.arr = np.random.default_rng(0).uniform(-1, 1, 1000)

    def test_default_uses_global_random_state(self) -> None:
        """Test that the default path is reproducible with `np.random.seed`."""
        # Prepare
        np.random.seed(0)
        expected = quantize([self.arr], CLIPPING_RANGE, TARGET_RANGE)[0]

        # Execute
        np.random.seed(0)
        actual = quantize([self.arr], CLIPPING_RANGE, TARGET_RANGE)[0]

        # Assert
        np.testing.assert_array_equal(actual, expected)

    def test_rng_makes_result_independent_of_global_state(self) -> None:
        """Test that an explicit generator ignores and preserves the global state."""
        # Prepare
        np.random.seed(0)
        expected = quantize(
            [self.arr], CLIPPING_RANGE, TARGET_RANGE, rng=np.random.default_rng(42)
        )[0]
        np.random.seed(1)
        state_before = np.random.get_state()

        # Execute
        actual = quantize(
            [self.arr], CLIPPING_RANGE, TARGET_RANGE, rng=np.random.default_rng(42)
        )[0]
        state_after = np.random.get_state()

        # Assert
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(state_before[0], state_after[0])
        np.testing.assert_array_equal(state_before[1], state_after[1])
        self.assertEqual(state_before[2:], state_after[2:])

    def test_rng_roundtrip_within_one_step(self) -> None:
        """Test that dequantized values stay within one quantization step."""
        # Prepare
        step = 2 * CLIPPING_RANGE / TARGET_RANGE

        # Execute
        quantized = quantize(
            [self.arr], CLIPPING_RANGE, TARGET_RANGE, rng=np.random.default_rng(0)
        )
        restored = dequantize(quantized, CLIPPING_RANGE, TARGET_RANGE)[0]

        # Assert
        self.assertLess(float(np.max(np.abs(restored - self.arr))), step)
