# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Partitioner tests."""


import unittest
from collections import Counter

from parameterized import parameterized

from datasets import Dataset
from flwr_datasets.partitioner.iid_partitioner import IidPartitioner


def _dummy_setup(num_partitions: int, num_rows: int) -> tuple[Dataset, IidPartitioner]:
    """Create a dummy dataset and partitioner based on given arguments.

    The partitioner has automatically the dataset assigned to it.
    """
    data = {
        "features": list(range(num_rows)),
        "labels": [i % 2 for i in range(num_rows)],
    }
    dataset = Dataset.from_dict(data)
    partitioner = IidPartitioner(num_partitions=num_partitions)
    partitioner.dataset = dataset
    return dataset, partitioner


class TestIidPartitioner(unittest.TestCase):
    """Test IidPartitioner."""

    @parameterized.expand(  # type: ignore
        [
            # num_partitions, num_rows
            (1, 100),
            (10, 100),
            (100, 100),
        ]
    )
    def test_load_partition_size(self, num_partitions: int, num_rows: int) -> None:
        """Test if the partition size matches the manually computed size.

        Only the correct data is tested in this method.

        In case the dataset is dividable among `num_partitions` the size of each
        partition should be the same. This checks if the randomly chosen partition has
        size as expected.
        """
        _, partitioner = _dummy_setup(num_partitions, num_rows)
        partition_size = num_rows // num_partitions
        partition_index = 0
        partition = partitioner.load_partition(partition_index)
        self.assertEqual(len(partition), partition_size)

    @parameterized.expand(  # type: ignore
        [
            # num_partitions, num_rows
            (2, 3),
            (2, 7),
        ]
    )
    def test_load_partition_size_not_dividable(
        self, num_partitions: int, num_rows: int
    ) -> None:
        """Test if the partition size matches the manually computed size.

        Only the correct data is tested in this method.

        In case of the number of rows not being dividable the first partitions should be
        greater.
        """
        _, partitioner = _dummy_setup(num_partitions, num_rows)
        min_partition_size = num_rows // num_partitions
        first_partitions_size = min_partition_size + 1
        partition_index = 0
        partition = partitioner.load_partition(partition_index)
        self.assertEqual(len(partition), first_partitions_size)

    @parameterized.expand(  # type: ignore
        [
            (10, 100),
            (5, 50),
            (20, 200),
        ]
    )
    def test_load_partition_correct_data(
        self, num_partitions: int, num_rows: int
    ) -> None:
        """Test if the data in partition is equal to the expected."""
        dataset, partitioner = _dummy_setup(num_partitions, num_rows)
        partition_size = num_rows // num_partitions
        partition_index = 2
        partition = partitioner.load_partition(partition_index)
        row_id = 0
        self.assertEqual(
            partition[row_id]["features"],
            # Note it's contiguous so partition_size * partition_index gets the first
            # element of the partition of partition_index
            dataset[partition_size * partition_index + row_id]["features"],
        )

    def test_default_partitioning_preserves_contiguous_order(self) -> None:
        """Test that default partitioning preserves historical contiguous sharding."""
        data = {"features": list(range(10)), "labels": [0] * 5 + [1] * 5}
        dataset = Dataset.from_dict(data)
        partitioner = IidPartitioner(num_partitions=2)
        partitioner.dataset = dataset

        partition = partitioner.load_partition(0)

        self.assertEqual(partition["features"], list(range(5)))
        self.assertEqual(Counter(partition["labels"]), Counter({0: 5}))

    def test_shuffle_mixes_sorted_dataset(self) -> None:
        """Test that shuffling prevents sorted labels from becoming single-label shards."""
        data = {"features": list(range(200)), "labels": [0] * 100 + [1] * 100}
        dataset = Dataset.from_dict(data)
        partitioner = IidPartitioner(num_partitions=2, shuffle=True, seed=42)
        partitioner.dataset = dataset

        first_partition = partitioner.load_partition(0)
        second_partition = partitioner.load_partition(1)

        self.assertGreater(Counter(first_partition["labels"])[0], 0)
        self.assertGreater(Counter(first_partition["labels"])[1], 0)
        self.assertGreater(Counter(second_partition["labels"])[0], 0)
        self.assertGreater(Counter(second_partition["labels"])[1], 0)

    def test_shuffle_with_same_seed_is_deterministic(self) -> None:
        """Test that the same seed produces the same shuffled partition."""
        dataset = Dataset.from_dict(
            {"features": list(range(100)), "labels": [idx % 2 for idx in range(100)]}
        )
        first_partitioner = IidPartitioner(num_partitions=5, shuffle=True, seed=42)
        second_partitioner = IidPartitioner(num_partitions=5, shuffle=True, seed=42)
        first_partitioner.dataset = dataset
        second_partitioner.dataset = dataset

        self.assertEqual(
            first_partitioner.load_partition(2)["features"],
            second_partitioner.load_partition(2)["features"],
        )

    def test_shuffle_with_different_seed_changes_order(self) -> None:
        """Test that different seeds produce different shuffled partitions."""
        dataset = Dataset.from_dict(
            {"features": list(range(100)), "labels": [idx % 2 for idx in range(100)]}
        )
        first_partitioner = IidPartitioner(num_partitions=5, shuffle=True, seed=42)
        second_partitioner = IidPartitioner(num_partitions=5, shuffle=True, seed=43)
        first_partitioner.dataset = dataset
        second_partitioner.dataset = dataset

        self.assertNotEqual(
            first_partitioner.load_partition(2)["features"],
            second_partitioner.load_partition(2)["features"],
        )

    def test_shuffle_with_no_seed_is_stable_after_first_load(self) -> None:
        """Test that seedless shuffling is cached within one partitioner instance."""
        dataset = Dataset.from_dict(
            {"features": list(range(100)), "labels": [idx % 2 for idx in range(100)]}
        )
        partitioner = IidPartitioner(num_partitions=5, shuffle=True, seed=None)
        partitioner.dataset = dataset

        first_load = partitioner.load_partition(2)["features"]
        second_load = partitioner.load_partition(2)["features"]

        self.assertEqual(first_load, second_load)

    @parameterized.expand(  # type: ignore
        [
            # num_partitions, num_rows
            (0, 100),
            (0, 200),
        ]
    )
    def test_partitioner_with_zero_partitions(
        self, num_partitions: int, num_rows: int
    ) -> None:
        """Test IidPartitioner with zero partitions."""
        with self.assertRaises(ValueError):
            _dummy_setup(num_partitions, num_rows)

    @parameterized.expand(  # type: ignore
        [
            # num_partitions, num_rows, partition_index
            (10, 10, 10),
            (10, 10, -1),
            (10, 10, 11),
            (10, 100, 1000),
            (5, 50, 60),
            (20, 200, 210),
        ]
    )
    def test_load_invalid_partition_index(
        self, num_partitions: int, num_rows: int, partition_index: int
    ) -> None:
        """Test loading a partition with an index out of range."""
        _, partitioner = _dummy_setup(num_partitions, num_rows)
        with self.assertRaises(ValueError):
            partitioner.load_partition(partition_index)

    def test_is_dataset_assigned_false(self) -> None:
        """Test if the is_dataset_assigned method works correctly if not assigned."""
        partitioner = IidPartitioner(num_partitions=10)

        # Initially, the dataset should not be assigned
        self.assertFalse(partitioner.is_dataset_assigned())

    def test_is_dataset_assigned_true(self) -> None:
        """Test if the is_dataset_assigned method works correctly if assigned."""
        num_partitions = 10
        num_rows = 100
        _, partitioner = _dummy_setup(num_partitions, num_rows)
        self.assertTrue(partitioner.is_dataset_assigned())

    def test_dataset_setter(self) -> None:
        """Test the dataset.setter method."""
        num_partitions = 10
        num_rows = 100
        dataset, partitioner = _dummy_setup(num_partitions, num_rows)

        # It should not allow setting the dataset a second time
        with self.assertRaises(Exception) as context:
            partitioner.dataset = dataset
        self.assertIn(
            "The dataset should be assigned only once", str(context.exception)
        )

    def test_dataset_getter_raises(self) -> None:
        """Test the dataset getter method."""
        num_partitions = 10
        partitioner = IidPartitioner(num_partitions=num_partitions)
        with self.assertRaises(AttributeError) as context:
            _ = partitioner.dataset
        self.assertIn(
            "The dataset field should be set before using it", str(context.exception)
        )

    def test_dataset_getter_used_correctly(self) -> None:
        """Test the dataset getter method."""
        num_partitions = 10
        num_rows = 100
        dataset, partitioner = _dummy_setup(num_partitions, num_rows)
        # After setting, it should return the dataset
        self.assertEqual(partitioner.dataset, dataset)


if __name__ == "__main__":
    unittest.main()
