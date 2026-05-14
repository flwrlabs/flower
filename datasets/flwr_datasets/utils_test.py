# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Utils tests."""

import unittest

from parameterized import parameterized_class

from datasets import Dataset, DatasetDict
from flwr_datasets.partitioner import IidPartitioner
from flwr_datasets.utils import concatenate_divisions, divide_dataset


@parameterized_class(
    (
        "partition_division",
        "sizes",
        "division_id",
        "expected_concatenation_size",
    ),
    [
        # Create 1 division
        ((1.0,), [40], 0, 40),
        ({"train": 1.0}, [40], "train", 40),
        # Create 2 divisions
        ((0.8, 0.2), [32, 8], 1, 8),
        ({"train": 0.8, "test": 0.2}, [32, 8], "test", 8),
        # Create 3 divisions
        ([0.6, 0.2, 0.2], [24, 8, 8], 1, 8),
        ({"train": 0.6, "valid": 0.2, "test": 0.2}, [24, 8, 8], "test", 8),
        # Create 4 divisions
        ([0.4, 0.2, 0.2, 0.2], [16, 8, 8, 8], 1, 8),
        ({"0": 0.4, "1": 0.2, "2": 0.2, "3": 0.2}, [16, 8, 8, 8], "1", 8),
        # Not full dataset
        # Create 1 division
        ([0.8], [32], 0, 32),
        ({"train": 0.8}, [32], "train", 32),
        # Create 2 divisions
        ([0.2, 0.1], [8, 4], 1, 4),
        ((0.2, 0.1), [8, 4], 0, 8),
        ({"train": 0.2, "test": 0.1}, [8, 4], "test", 4),
        # Create 3 divisions
        ([0.6, 0.2, 0.1], [24, 8, 4], 2, 4),
        ({"train": 0.6, "valid": 0.2, "test": 0.1}, [24, 8, 4], "test", 4),
        # Create 4 divisions
        ([0.4, 0.2, 0.1, 0.2], [16, 8, 4, 8], 2, 4),
        ({"0": 0.4, "1": 0.2, "2": 0.1, "3": 0.2}, [16, 8, 4, 8], "2", 4),
    ],
)
class UtilsTests(unittest.TestCase):
    """Utils for tests."""

    partition_division: list[float] | tuple[float, ...] | dict[str, float]
    sizes: tuple[int]
    division_id: int | str
    expected_concatenation_size: int

    def setUp(self) -> None:
        """Set up a dataset."""
        self.dataset = Dataset.from_dict({"data": range(40)})

    def test_correct_sizes(self) -> None:
        """Test correct size of the division."""
        divided_dataset = divide_dataset(self.dataset, self.partition_division)
        if isinstance(divided_dataset, (list | tuple)):
            lengths = [len(split) for split in divided_dataset]
        else:
            lengths = [len(split) for split in divided_dataset.values()]

        self.assertEqual(self.sizes, lengths)

    def test_correct_return_types(self) -> None:
        """Test correct types of the divided dataset based on the config."""
        divided_dataset = divide_dataset(self.dataset, self.partition_division)
        if isinstance(self.partition_division, (list | tuple)):
            self.assertIsInstance(divided_dataset, list)
        else:
            self.assertIsInstance(divided_dataset, DatasetDict)

    def test_concatenate_divisions(
        self,
    ) -> None:
        """Test if the length of the divisions match the concatenated dataset."""
        num_partitions = 4
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = self.dataset
        centralized_from_federated_test = concatenate_divisions(
            partitioner,
            self.partition_division,
            self.division_id,
        )

        self.assertEqual(
            len(centralized_from_federated_test), self.expected_concatenation_size
        )


class TestIncorrectUtilsUsage(unittest.TestCase):
    """Test incorrect utils usage."""

    def test_all_divisions_to_concat_size_zero(self) -> None:
        """Test raises when all divisions for concatenations are zero."""
        num_partitions = 4
        partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner.dataset = Dataset.from_dict({"data": range(40)})
        division_id = 1
        partition_division = [0.8, 0.0]

        with self.assertRaises(ValueError):
            _ = concatenate_divisions(partitioner, partition_division, division_id)


if __name__ == "__main__":
    unittest.main()
