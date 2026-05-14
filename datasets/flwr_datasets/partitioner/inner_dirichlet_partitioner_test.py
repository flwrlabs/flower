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
"""Test DirichletPartitioner."""

# pylint: disable=W0212
import unittest

from datasets import Dataset
from flwr_datasets.common.typing import NDArrayFloat, NDArrayInt
from flwr_datasets.partitioner.inner_dirichlet_partitioner import (
    InnerDirichletPartitioner,
)


def _dummy_setup(
    num_rows: int,
    partition_by: str,
    partition_sizes: list[int] | NDArrayInt,
    alpha: float | list[float] | NDArrayFloat,
) -> tuple[Dataset, InnerDirichletPartitioner]:
    """Create a dummy dataset and partitioner for testing."""
    data = {
        partition_by: [i % 3 for i in range(num_rows)],
        "features": list(range(num_rows)),
    }
    dataset = Dataset.from_dict(data)
    partitioner = InnerDirichletPartitioner(
        partition_sizes=partition_sizes,
        alpha=alpha,
        partition_by=partition_by,
    )
    partitioner.dataset = dataset
    return dataset, partitioner


class TestInnerDirichletPartitionerSuccess(unittest.TestCase):
    """Test InnerDirichletPartitioner used with no exceptions."""

    def test_correct_num_of_partitions(self) -> None:
        """Test correct number of partitions."""
        num_rows = 113
        partition_by = "labels"
        alpha = 1.0
        partition_sizes = [20, 20, 30, 43]

        _, partitioner = _dummy_setup(num_rows, partition_by, partition_sizes, alpha)
        _ = partitioner.load_partition(0)
        self.assertEqual(
            len(partitioner._partition_id_to_indices.keys()), len(partition_sizes)
        )

    def test_correct_partition_sizes(self) -> None:
        """Test correct partition sizes."""
        num_rows = 113
        partition_by = "labels"
        alpha = 1.0
        partition_sizes = [20, 20, 30, 43]

        _, partitioner = _dummy_setup(num_rows, partition_by, partition_sizes, alpha)
        _ = partitioner.load_partition(0)
        sizes_created = [
            len(indices) for indices in partitioner._partition_id_to_indices.values()
        ]
        self.assertEqual(sorted(sizes_created), partition_sizes)


class TestInnerDirichletPartitionerFailure(unittest.TestCase):
    """Test InnerDirichletPartitioner failures (exceptions) by incorrect usage."""

    def test_incorrect_shape_of_alpha(self) -> None:
        """Test the alpha shape not equal to the number of unique classes."""
        num_rows = 113
        partition_by = "labels"
        alpha = [1.0, 1.0]
        partition_sizes = [20, 20, 30, 43]

        _, partitioner = _dummy_setup(num_rows, partition_by, partition_sizes, alpha)
        with self.assertRaises(ValueError):
            _ = partitioner.load_partition(0)

    def test_too_big_sum_of_partition_sizes(self) -> None:
        """Test sum of partition_sizes greater than the size of the dataset."""
        num_rows = 113
        partition_by = "labels"
        alpha = 1.0
        partition_sizes = [60, 60, 30, 43]

        _, partitioner = _dummy_setup(num_rows, partition_by, partition_sizes, alpha)
        with self.assertRaises(ValueError):
            _ = partitioner.load_partition(0)


if __name__ == "__main__":
    unittest.main()
