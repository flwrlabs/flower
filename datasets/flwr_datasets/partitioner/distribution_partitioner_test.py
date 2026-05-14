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
"""Test cases for DistributionPartitioner."""

import unittest
from collections import Counter
from typing import Any

import numpy as np
from parameterized import parameterized_class

from datasets import Dataset
from flwr_datasets.common.typing import NDArrayFloat, NDArrayInt
from flwr_datasets.partitioner.distribution_partitioner import DistributionPartitioner


def _dummy_dataset_setup(
    num_samples: int, partition_by: str, num_unique_classes: int
) -> Dataset:
    """Create a dummy dataset for testing."""
    data = {
        partition_by: np.tile(
            np.arange(num_unique_classes), num_samples // num_unique_classes + 1
        )[:num_samples],
        "features": np.random.randn(num_samples),
    }
    return Dataset.from_dict(data)


def _dummy_distribution_setup(
    num_partitions: int,
    num_unique_labels_per_partition: int,
    num_unique_labels: int,
    random_mode: bool = False,
) -> NDArrayFloat | NDArrayInt:
    """Create a dummy distribution for testing."""
    num_columns = num_unique_labels_per_partition * num_partitions / num_unique_labels
    if random_mode:
        rng = np.random.default_rng(2024)
        return rng.integers(1, 10, size=(num_unique_labels, int(num_columns)))
    return np.tile(np.arange(num_columns) + 1.0, (num_unique_labels, 1))


# pylint: disable=R0913
def _get_partitioner(  # pylint: disable=R0917
    num_partitions: int,
    num_unique_labels_per_partition: int,
    num_samples: int,
    num_unique_labels: int,
    preassigned_num_samples_per_label: int,
    rescale_mode: bool = True,
) -> tuple[DistributionPartitioner, dict[int, Dataset]]:
    """Create DistributionPartitioner instance."""
    dataset = _dummy_dataset_setup(
        num_samples,
        "labels",
        num_unique_labels,
    )
    distribution = _dummy_distribution_setup(
        num_partitions,
        num_unique_labels_per_partition,
        num_unique_labels,
    )
    partitioner = DistributionPartitioner(
        distribution_array=distribution,
        num_partitions=num_partitions,
        num_unique_labels_per_partition=num_unique_labels_per_partition,
        partition_by="labels",
        preassigned_num_samples_per_label=preassigned_num_samples_per_label,
        rescale=rescale_mode,
    )
    partitioner.dataset = dataset
    partitions: dict[int, Dataset] = {
        pid: partitioner.load_partition(pid) for pid in range(num_partitions)
    }

    return partitioner, partitions


# mypy: disable-error-code="attr-defined"
@parameterized_class(
    (
        "num_partitions",
        "num_unique_labels_per_partition",
        "num_samples",
        "num_unique_labels",
        "preassigned_num_samples_per_label",
    ),
    [
        (10, 2, 200, 10, 5),
        (10, 2, 200, 10, 0),
        (20, 1, 200, 10, 5),
    ],
)
# pylint: disable=E1101
class TestDistributionPartitioner(unittest.TestCase):
    """Unit tests for DistributionPartitioner."""

    def test_correct_num_classes_when_partitioned(self) -> None:
        """Test correct number of unique classes."""
        _, partitions = _get_partitioner(
            num_partitions=self.num_partitions,
            num_unique_labels_per_partition=self.num_unique_labels_per_partition,
            num_samples=self.num_samples,
            num_unique_labels=self.num_unique_labels,
            preassigned_num_samples_per_label=self.preassigned_num_samples_per_label,
        )
        unique_classes_per_partition = {
            pid: np.unique(partition["labels"]) for pid, partition in partitions.items()
        }

        for unique_classes in unique_classes_per_partition.values():
            self.assertEqual(self.num_unique_labels_per_partition, len(unique_classes))

    def test_correct_num_times_classes_sampled_across_partitions(self) -> None:
        """Test correct number of times each unique class is drawn from distribution."""
        partitioner, partitions = _get_partitioner(
            num_partitions=self.num_partitions,
            num_unique_labels_per_partition=self.num_unique_labels_per_partition,
            num_samples=self.num_samples,
            num_unique_labels=self.num_unique_labels,
            preassigned_num_samples_per_label=self.preassigned_num_samples_per_label,
        )

        partitioned_distribution: dict[Any, list[Any]] = {
            label: [] for label in partitioner.dataset.unique("labels")
        }

        num_columns = (
            self.num_unique_labels_per_partition
            * self.num_partitions
            / self.num_unique_labels
        )
        for _, partition in partitions.items():
            for label in partition.unique("labels"):
                value_counts = Counter(partition["labels"])
                partitioned_distribution[label].append(value_counts[label])

        for label in partitioner.dataset.unique("labels"):
            self.assertEqual(num_columns, len(partitioned_distribution[label]))

    def test_exact_distribution_assignment(self) -> None:
        """Test that exact distribution is allocated to each class."""
        partitioner, partitions = _get_partitioner(
            num_partitions=self.num_partitions,
            num_unique_labels_per_partition=self.num_unique_labels_per_partition,
            num_samples=self.num_samples,
            num_unique_labels=self.num_unique_labels,
            preassigned_num_samples_per_label=self.preassigned_num_samples_per_label,
            rescale_mode=False,
        )
        partitioned_distribution: dict[Any, list[Any]] = {
            label: [] for label in partitioner.dataset.unique("labels")
        }

        for _, partition in partitions.items():
            for label in partition.unique("labels"):
                value_counts = Counter(partition["labels"])
                partitioned_distribution[label].append(value_counts[label])

        for idx, label in enumerate(sorted(partitioner.dataset.unique("labels"))):
            np.testing.assert_array_equal(
                partitioner._distribution_array[idx],  # pylint: disable=W0212
                partitioned_distribution[label],
            )


if __name__ == "__main__":
    unittest.main()
