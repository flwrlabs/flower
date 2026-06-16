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
"""Unit tests for conversion utility functions."""


import unittest
from io import BytesIO

import numpy as np

from flwr.common import array_from_numpy as common_array_from_numpy
from flwr.common.constant import SType
from flwr.compat.common.record import array_from_numpy as compat_record_array_from_numpy

from .conversion_utils import array_from_numpy


class TestArrayFromNumpy(unittest.TestCase):
    """Unit tests for array_from_numpy."""

    def test_array_from_numpy(self) -> None:
        """Test the array_from_numpy function."""
        # Prepare
        original_array = np.array([1, 2, 3], dtype=np.float32)

        # Execute
        array_instance = array_from_numpy(original_array)
        buffer = BytesIO(array_instance.data)
        deserialized_array = np.load(buffer, allow_pickle=False)

        # Assert
        self.assertEqual(array_instance.dtype, str(original_array.dtype))
        self.assertEqual(array_instance.shape, tuple(original_array.shape))
        self.assertEqual(array_instance.stype, SType.NUMPY)
        np.testing.assert_array_equal(deserialized_array, original_array)

    def test_reexports(self) -> None:
        """Test compatibility re-exports."""
        self.assertIs(common_array_from_numpy, array_from_numpy)
        self.assertIs(compat_record_array_from_numpy, array_from_numpy)
