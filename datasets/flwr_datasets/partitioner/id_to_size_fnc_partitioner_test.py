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
"""IdToSizeFuncitonPartitioner tests."""

import unittest

from parameterized import parameterized

from datasets import Dataset
from flwr_datasets.partitioner.linear_partitioner import LinearPartitioner


def _dummy_dataset(num_rows: int) -> Dataset:
    data = {
        "features": list(range(num_rows)),
        "labels": [i % 2 for i in range(num_rows)],
    }
    dataset = Dataset.from_dict(data)
    return dataset


class TestLinearPartitioner(unittest.TestCase):
    """Test LinearPartitioner."""

    @parameterized.expand(  # type: ignore
        [
            (1, 100),
            (10, 100),
            (5, 55),  # This will leave some undivided samples
        ]
    )
    def test_linear_distribution(self, num_partitions: int, num_rows: int) -> None:
        """Test the linear distribution of samples."""
        dataset = _dummy_dataset(num_rows)
        partitioner = LinearPartitioner(num_partitions=num_partitions)
        partitioner.dataset = dataset
        # Run a single partition loading to trigger the division
        _ = partitioner.load_partition(0)
        total_samples = sum(partitioner.partition_id_to_size.values())
        self.assertEqual(total_samples, num_rows)

        # Testing if each partition is getting more than the previous one
        last_count = 0
        for i in range(num_partitions):
            current_count = partitioner.partition_id_to_size[i]
            self.assertGreaterEqual(current_count, last_count)
            last_count = current_count

    @parameterized.expand(  # type: ignore
        [
            (10, 100),
            (5, 55),  # This will leave some undivided samples
            (7, 77),  # This will leave some undivided samples
        ]
    )
    def test_undivided_samples(self, num_partitions: int, num_rows: int) -> None:
        """Test the logic for distributing undivided samples."""
        dataset = _dummy_dataset(num_rows)
        partitioner = LinearPartitioner(num_partitions=num_partitions)
        partitioner.dataset = dataset
        # If there are any undivided samples, they should be added to the largest
        # partition
        last_partition_id = num_partitions - 1
        actual_samples_in_last_partition = len(
            partitioner.load_partition(last_partition_id)
        )
        expected_samples_in_last_partition = partitioner.partition_id_to_size[
            last_partition_id
        ]
        self.assertEqual(
            expected_samples_in_last_partition, actual_samples_in_last_partition
        )

    def test_meaningless_params(self) -> None:
        """Test if the params leading to partition size not greater than zero raises."""
        num_rows = 10
        num_partitions = 100
        dataset = _dummy_dataset(num_rows)
        partitioner = LinearPartitioner(num_partitions=num_partitions)
        partitioner.dataset = dataset
        with self.assertRaises(ValueError) as context:
            partitioner.load_partition(1)
        self.assertIn(
            "The given specification of the parameter num_partitions=100 for the given "
            "dataset results in the partitions sizes that are not greater than 0.",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
