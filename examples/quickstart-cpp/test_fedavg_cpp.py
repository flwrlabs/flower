"""Tests for C++ quickstart strategy helpers."""

import struct
import unittest

import numpy as np

from fedavg_cpp import bytes_to_ndarray, ndarray_to_bytes


class FedAvgCppSerializationTest(unittest.TestCase):
    def test_ndarray_serialization_uses_little_endian_float64(self) -> None:
        array = np.asarray([1.25, -2.5, 3.75], dtype=np.float64)

        tensor_bytes = ndarray_to_bytes(array)

        self.assertEqual(tensor_bytes, struct.pack("<3d", 1.25, -2.5, 3.75))
        np.testing.assert_array_equal(bytes_to_ndarray(tensor_bytes), array)

    def test_bytes_to_ndarray_rejects_misaligned_tensor(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by 8"):
            bytes_to_ndarray(b"123")


if __name__ == "__main__":
    unittest.main()
