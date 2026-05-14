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
"""Test cases for PathologicalPartitioner."""

import unittest

import numpy as np
from parameterized import parameterized, parameterized_class

import datasets
from datasets import Dataset
from flwr_datasets.partitioner.pathological_partitioner import PathologicalPartitioner


def _dummy_dataset_setup(
    num_samples: int,
    partition_by: str,
    num_unique_classes: int,
    string_partition_by: bool = False,
) -> Dataset:
    """Create a dummy dataset for testing."""
    data = {
        partition_by: np.tile(
            np.arange(num_unique_classes), num_samples // num_unique_classes + 1
        )[:num_samples],
        "features": np.random.randn(num_samples),
    }
    if string_partition_by:
        data[partition_by] = data[partition_by].astype(str)
    return Dataset.from_dict(data)


def _dummy_heterogeneous_dataset_setup(
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


@parameterized_class(("string_partition_by",), [(False,), (True,)])
class TestClassConstrainedPartitioner(unittest.TestCase):
    """Unit tests for PathologicalPartitioner."""

    string_partition_by: bool

    @parameterized.expand(  # type: ignore
        [
            # num_partition, num_classes_per_partition, num_samples, total_classes
            (3, 1, 60, 3),  # Single class per partition scenario
            (5, 2, 100, 5),
            (5, 2, 100, 10),
            (4, 3, 120, 6),
        ]
    )
    def test_correct_num_classes_when_partitioned(
        self,
        num_partitions: int,
        num_classes_per_partition: int,
        num_samples: int,
        num_unique_classes: int,
    ) -> None:
        """Test correct number of unique classes."""
        dataset = _dummy_dataset_setup(num_samples, "labels", num_unique_classes)
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by="labels",
            num_classes_per_partition=num_classes_per_partition,
        )
        partitioner.dataset = dataset
        partitions: dict[int, Dataset] = {
            pid: partitioner.load_partition(pid) for pid in range(num_partitions)
        }
        unique_classes_per_partition = {
            pid: np.unique(partition["labels"]) for pid, partition in partitions.items()
        }

        for unique_classes in unique_classes_per_partition.values():
            self.assertEqual(num_classes_per_partition, len(unique_classes))

    def test_first_class_deterministic_assignment(self) -> None:
        """Test deterministic assignment of first classes to partitions.

        Test if all the classes are used (which has to be the case, given num_partitions
        >= than the number of unique classes).
        """
        partition_by = "labels"
        dataset = _dummy_dataset_setup(100, partition_by, 10)
        partitioner = PathologicalPartitioner(
            num_partitions=10,
            partition_by="labels",
            num_classes_per_partition=2,
            class_assignment_mode="first-deterministic",
        )
        partitioner.dataset = dataset
        partitioner.load_partition(0)
        expected_classes = set(
            range(10)
            # pylint: disable=unsubscriptable-object
            if isinstance(dataset[partition_by][0], int)
            else [str(i) for i in range(10)]
        )
        actual_classes: set[int] = set()
        for pid in range(10):
            partition = partitioner.load_partition(pid)
            actual_classes.update(np.unique(partition["labels"]))
        self.assertEqual(expected_classes, actual_classes)

    @parameterized.expand(
        [  # type: ignore
            # num_partitions, num_classes_per_partition, num_samples, num_unique_classes
            (4, 2, 80, 8),
            (10, 2, 100, 10),
        ]
    )
    def test_deterministic_class_assignment(
        self, num_partitions, num_classes_per_partition, num_samples, num_unique_classes
    ) -> None:
        """Test deterministic assignment of classes to partitions."""
        dataset = _dummy_dataset_setup(num_samples, "labels", num_unique_classes)
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by="labels",
            num_classes_per_partition=num_classes_per_partition,
            class_assignment_mode="deterministic",
        )
        partitioner.dataset = dataset
        partitions = {
            pid: partitioner.load_partition(pid) for pid in range(num_partitions)
        }

        # Verify each partition has the expected classes, order does not matter
        for pid, partition in partitions.items():
            expected_labels = sorted(
                [
                    (pid + i) % num_unique_classes
                    for i in range(num_classes_per_partition)
                ]
            )
            # pylint: disable=unsubscriptable-object
            if isinstance(dataset["labels"][0], str):
                expected_labels = [str(label) for label in expected_labels]
            actual_labels = sorted(np.unique(partition["labels"]))
            self.assertTrue(
                np.array_equal(expected_labels, actual_labels),
                f"Partition {pid} does not have the expected labels: "
                f"{expected_labels} but instead {actual_labels}.",
            )

    @parameterized.expand(
        [  # type: ignore
            # num_partitions, num_classes_per_partition, num_samples, num_unique_classes
            (10, 3, 20, 3),
        ]
    )
    def test_too_many_partitions_for_a_class(
        self, num_partitions, num_classes_per_partition, num_samples, num_unique_classes
    ) -> None:
        """Test  too many partitions for the number of samples in a class."""
        dataset_1 = _dummy_dataset_setup(
            num_samples // 2,
            "labels",
            num_unique_classes - 1,
            string_partition_by=self.string_partition_by,
        )
        # Create a skewed part of the dataset for the last label
        data = {
            "labels": np.array([num_unique_classes - 1] * (num_samples // 2)),
            "features": np.random.randn(num_samples // 2),
        }
        # pylint: disable=unsubscriptable-object
        if isinstance(dataset_1["labels"][0], str):
            data["labels"] = data["labels"].astype(str)
        dataset_2 = Dataset.from_dict(data)
        dataset = datasets.concatenate_datasets([dataset_1, dataset_2])

        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by="labels",
            num_classes_per_partition=num_classes_per_partition,
            class_assignment_mode="random",
        )
        partitioner.dataset = dataset

        with self.assertRaises(ValueError) as context:
            _ = partitioner.load_partition(0)
        self.assertEqual(
            str(context.exception),
            "Label: 0 is needed to be assigned to more partitions (10) than there are "
            "samples (corresponding to this label) in the dataset (5). "
            "Please decrease the `num_partitions`, `num_classes_per_partition` to "
            "avoid this situation, or try `class_assignment_mode='deterministic'` to "
            "create a more even distribution of classes along the partitions. "
            "Alternatively use a different dataset if you can not adjust the any of "
            "these parameters.",
        )

    @parameterized.expand(  # type: ignore
        [
            # num_partitions, num_classes_per_partition, num_samples, num_unique_classes
            (10, 11, 100, 10),  # 11 > 10
            (5, 11, 100, 10),  # 11 > 10
            (10, 20, 100, 5),  # 20 > 5
        ]
    )
    def test_more_classes_per_partition_than_num_unique_classes_in_dataset_raises(
        self,
        num_partitions: int,
        num_classes_per_partition: int,
        num_samples: int,
        num_unique_classes: int,
    ) -> None:
        """Test more num_classes_per_partition > num_unique_classes in the dataset."""
        dataset = _dummy_dataset_setup(
            num_samples,
            "labels",
            num_unique_classes,
            string_partition_by=self.string_partition_by,
        )
        with self.assertRaises(ValueError) as context:
            partitioner = PathologicalPartitioner(
                num_partitions=num_partitions,
                partition_by="labels",
                num_classes_per_partition=num_classes_per_partition,
            )
            partitioner.dataset = dataset
            partitioner.load_partition(0)
        self.assertEqual(
            str(context.exception),
            "The specified "
            f"`num_classes_per_partition`={num_classes_per_partition} is "
            f"greater than the number of unique classes in the given "
            f"dataset={len(dataset.unique('labels'))}. Reduce the "
            f"`num_classes_per_partition` or make use different dataset "
            f"to apply this partitioning.",
        )

    @parameterized.expand(  # type: ignore
        [
            # num_classes_per_partition should be irrelevant since the exception should
            # be raised at the very beginning
            # num_partitions, num_classes_per_partition, num_samples
            (10, 2, 5),
            (10, 10, 5),
            (100, 10, 99),
        ]
    )
    def test_more_partitions_than_samples_raises(
        self, num_partitions: int, num_classes_per_partition: int, num_samples: int
    ) -> None:
        """Test if generation of more partitions that there are samples raises."""
        # The number of unique classes in the dataset should be irrelevant since the
        # exception should be raised at the very beginning
        dataset = _dummy_dataset_setup(num_samples, "labels", num_unique_classes=5)
        with self.assertRaises(ValueError) as context:
            partitioner = PathologicalPartitioner(
                num_partitions=num_partitions,
                partition_by="labels",
                num_classes_per_partition=num_classes_per_partition,
            )
            partitioner.dataset = dataset
            partitioner.load_partition(0)
        self.assertEqual(
            str(context.exception),
            "The number of partitions needs to be smaller than the number of "
            "samples in the dataset.",
        )


if __name__ == "__main__":
    unittest.main()
