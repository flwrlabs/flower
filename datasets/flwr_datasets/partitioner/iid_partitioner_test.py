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


def _dummy_setup(
    num_partitions: int,
    num_rows: int,
    shuffle: bool = False,
    seed: int | None = 42,
) -> tuple[Dataset, IidPartitioner]:
    """Create a dummy dataset and partitioner based on given arguments.

    The partitioner has automatically the dataset assigned to it.
    """
    data = {
        "features": list(range(num_rows)),
        "labels": [i % 2 for i in range(num_rows)],
    }
    dataset = Dataset.from_dict(data)
    partitioner = IidPartitioner(
        num_partitions=num_partitions, shuffle=shuffle, seed=seed
    )
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

    def test_shuffle_false_preserves_contiguous_shards(self) -> None:
        """Test that the default behavior keeps contiguous shards."""
        dataset, partitioner = _dummy_setup(num_partitions=2, num_rows=10)

        partition = partitioner.load_partition(1)

        self.assertEqual(partition["features"], dataset["features"][5:10])

    def test_shuffle_true_breaks_sorted_label_blocks(self) -> None:
        """Test that shuffling avoids homogeneous shards for sorted labels."""
        dataset = Dataset.from_dict(
            {
                "features": list(range(200)),
                "labels": [0] * 100 + [1] * 100,
            }
        )
        partitioner = IidPartitioner(num_partitions=2, shuffle=True, seed=42)
        partitioner.dataset = dataset

        for partition_id in range(2):
            counts = Counter(partitioner.load_partition(partition_id)["labels"])
            self.assertGreater(counts[0], 0)
            self.assertGreater(counts[1], 0)

    def test_shuffle_true_is_deterministic_with_fixed_seed(self) -> None:
        """Test that shuffling is deterministic given a fixed seed."""
        _, partitioner_1 = _dummy_setup(
            num_partitions=4, num_rows=40, shuffle=True, seed=123
        )
        _, partitioner_2 = _dummy_setup(
            num_partitions=4, num_rows=40, shuffle=True, seed=123
        )

        partition_1 = partitioner_1.load_partition(0)["features"]
        partition_2 = partitioner_2.load_partition(0)["features"]

        self.assertEqual(partition_1, partition_2)

    def test_shuffle_true_with_no_seed_uses_consistent_partitioning(self) -> None:
        """Test that seed=None shuffles once and then reuses the same ordering."""
        num_partitions = 4
        num_rows = 40
        _, partitioner = _dummy_setup(
            num_partitions=num_partitions,
            num_rows=num_rows,
            shuffle=True,
            seed=None,
        )

        first_partition = partitioner.load_partition(0)["features"]
        repeated_first_partition = partitioner.load_partition(0)["features"]
        all_features = []
        for partition_id in range(num_partitions):
            all_features.extend(partitioner.load_partition(partition_id)["features"])

        self.assertEqual(first_partition, repeated_first_partition)
        self.assertEqual(len(all_features), len(set(all_features)))
        self.assertEqual(sorted(all_features), list(range(num_rows)))


if __name__ == "__main__":
    unittest.main()
