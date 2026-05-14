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
"""NaturalIdPartitioner partitioner tests."""

import itertools
import math
import unittest

from parameterized import parameterized

from datasets import Dataset
from flwr_datasets.partitioner.natural_id_partitioner import NaturalIdPartitioner


def _dummy_setup(
    num_rows: int, n_unique_natural_ids: int
) -> tuple[Dataset, NaturalIdPartitioner]:
    """Create a dummy dataset and partitioner based on given arguments.

    The partitioner has automatically the dataset assigned to it.
    """
    dataset = _create_dataset(num_rows, n_unique_natural_ids)
    partitioner = NaturalIdPartitioner(partition_by="natural_id")
    partitioner.dataset = dataset
    return dataset, partitioner


def _create_dataset(num_rows: int, n_unique_natural_ids: int) -> Dataset:
    """Create dataset based on the number of rows and unique natural ids."""
    data = {
        "features": list(range(num_rows)),
        "natural_id": [f"{i % n_unique_natural_ids}" for i in range(num_rows)],
        "labels": [i % 2 for i in range(num_rows)],
    }
    dataset = Dataset.from_dict(data)
    return dataset


class TestNaturalIdPartitioner(unittest.TestCase):
    """Test IidPartitioner."""

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_natural_ids
        list(itertools.product([10, 30, 100, 1000], [2, 3, 4, 5]))
    )
    def test_load_partition_num_partitions(
        self, num_rows: int, num_unique_natural_id: int
    ) -> None:
        """Test if the number of partitions match the number of unique natural ids.

        Only the correct data is tested in this method.
        """
        _, partitioner = _dummy_setup(num_rows, num_unique_natural_id)
        # Simulate usage to start lazy partition_id_to_natural_id creation
        _ = partitioner.load_partition(0)
        self.assertEqual(
            len(partitioner.partition_id_to_natural_id), num_unique_natural_id
        )

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_natural_ids
        list(itertools.product([10, 30, 100, 1000], [2, 3, 4, 5]))
    )
    def test_load_partition_max_partition_size(
        self, num_rows: int, num_unique_natural_ids: int
    ) -> None:
        """Test if the number of partitions match the number of unique natural ids.

        Only the correct data is tested in this method.
        """
        print(num_rows)
        print(num_unique_natural_ids)
        _, partitioner = _dummy_setup(num_rows, num_unique_natural_ids)
        max_size = max(
            len(partitioner.load_partition(i)) for i in range(num_unique_natural_ids)
        )
        self.assertEqual(max_size, math.ceil(num_rows / num_unique_natural_ids))

    def test_partitioner_with_non_existing_column_partition_by(self) -> None:
        """Test error when the partition_by columns does not exist."""
        dataset = _create_dataset(10, 2)
        partitioner = NaturalIdPartitioner(partition_by="not-existing")
        partitioner.dataset = dataset
        with self.assertRaises(ValueError):
            partitioner.load_partition(0)

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_natural_ids
        list(itertools.product([10, 30, 100, 1000], [2, 3, 4, 5]))
    )
    def test_correct_number_of_partitions(
        self, num_rows: int, num_unique_natural_ids: int
    ) -> None:
        """Test if the # of available partitions is equal to # of unique clients."""
        _, partitioner = _dummy_setup(num_rows, num_unique_natural_ids)
        _ = partitioner.load_partition(partition_id=0)
        self.assertEqual(
            len(partitioner.partition_id_to_natural_id), num_unique_natural_ids
        )

    def test_cannot_set_partition_id_to_natural_id(self) -> None:
        """Test the lack of ability to set partition_id_to_natural_id."""
        _, partitioner = _dummy_setup(num_rows=10, n_unique_natural_ids=2)
        with self.assertRaises(AttributeError):
            partitioner.partition_id_to_natural_id = {0: "0"}

    def test_consistent_partition_ids(self) -> None:
        """Test that the partition IDs assigned to the natural IDs are consistent."""
        train_data = {
            "features": [1, 2, 3],
            "labels": [0, 0, 1],
            "clients": ["a", "b", "a"],
        }
        test_data = {
            "features": [4, 5, 6],
            "labels": [1, 0, 0],
            "clients": ["b", "a", "a"],
        }
        train_dataset = Dataset.from_dict(train_data)
        test_dataset = Dataset.from_dict(test_data)

        # Create partitioners
        train_partitioner = NaturalIdPartitioner(partition_by="clients")
        test_partitioner = NaturalIdPartitioner(partition_by="clients")
        train_partitioner.dataset = train_dataset
        test_partitioner.dataset = test_dataset
        train_partitioner.load_partition(0)
        test_partitioner.load_partition(0)

        self.assertEqual(
            train_partitioner.partition_id_to_natural_id,
            test_partitioner.partition_id_to_natural_id,
        )


if __name__ == "__main__":
    unittest.main()
