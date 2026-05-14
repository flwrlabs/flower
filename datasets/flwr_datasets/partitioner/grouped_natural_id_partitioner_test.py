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
"""Test GroupedNaturalIdPartitioner."""

import unittest
from typing import Literal

from parameterized import parameterized, parameterized_class

from datasets import Dataset
from flwr_datasets.partitioner.grouped_natural_id_partitioner import (
    GroupedNaturalIdPartitioner,
)


def _create_dataset(num_rows: int, n_unique_natural_ids: int) -> Dataset:
    """Create dataset based on the number of rows and unique natural ids."""
    data = {
        "features": list(range(num_rows)),
        "natural_id": [f"{i % n_unique_natural_ids}" for i in range(num_rows)],
        "labels": [i % 2 for i in range(num_rows)],
    }
    dataset = Dataset.from_dict(data)
    return dataset


# mypy: disable-error-code="attr-defined"
@parameterized_class(
    ("sort_unique_ids",),
    [
        (False,),
        (True,),
    ],
)
# pylint: disable=no-member
class TestGroupedNaturalIdPartitioner(unittest.TestCase):
    """Test GroupedNaturalIdPartitioner."""

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_natural_ids, group_size, expected_num_partitions
        [
            [10, 10, 2, 5],
            [11, 10, 2, 5],
            [100, 10, 2, 5],
            [12, 6, 3, 2],
        ]
    )
    def test_strict_mode_num_partitions_and_partition_sizes(
        self,
        num_rows: int,
        num_unique_natural_id: int,
        group_size: int,
        expected_num_partitions: int,
    ) -> None:
        """Test strict mode with valid group size."""
        dataset = _create_dataset(num_rows, num_unique_natural_id)
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="natural_id",
            group_size=group_size,
            mode="strict",
            sort_unique_ids=self.sort_unique_ids,
        )
        partitioner.dataset = dataset
        # Trigger partitioning
        _ = partitioner.load_partition(0)
        self.assertEqual(partitioner.num_partitions, expected_num_partitions)

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_natural_ids, group_size, expected_num_partitions,
        # expected_num_unique_natural_ids
        [
            [10, 10, 2, [2, 2, 2, 2, 2]],
            [100, 10, 2, [2, 2, 2, 2, 2]],
            [12, 6, 3, [3, 3]],
            # The cases in which the partitions should be smaller
            [10, 7, 2, [2, 2, 2, 1]],
            [10, 3, 2, [2, 1]],
        ]
    )
    def test_allow_smaller_mode_num_partitions_and_partition_sizes(
        self,
        num_rows: int,
        num_unique_natural_id: int,
        group_size: int,
        expected_num_unique_natural_ids: list[int],
    ) -> None:
        """Test allow-smaller mode handles the remainder correctly."""
        dataset = _create_dataset(num_rows, num_unique_natural_id)
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="natural_id",
            group_size=group_size,
            mode="allow-smaller",
            sort_unique_ids=self.sort_unique_ids,
        )
        partitioner.dataset = dataset
        # Trigger partitioning
        partitions = [
            partitioner.load_partition(i) for i in range(partitioner.num_partitions)
        ]
        unique_natural_ids = [
            len(partition.unique("natural_id")) for partition in partitions
        ]
        self.assertEqual(unique_natural_ids, expected_num_unique_natural_ids)

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_natural_ids, group_size, expected_num_partitions,
        # expected_num_unique_natural_ids
        [
            [10, 10, 2, [2, 2, 2, 2, 2]],
            [100, 10, 2, [2, 2, 2, 2, 2]],
            [12, 6, 3, [3, 3]],
            # The cases in which the partitions should be smaller
            [10, 7, 2, [3, 2, 2]],
            [10, 3, 2, [3]],
        ]
    )
    def test_allow_bigger_mode_num_partitions_and_partition_sizes(
        self,
        num_rows: int,
        num_unique_natural_id: int,
        group_size: int,
        expected_num_unique_natural_ids: list[int],
    ) -> None:
        """Test allow-bigger mode handles the remainder correctly."""
        dataset = _create_dataset(num_rows, num_unique_natural_id)
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="natural_id",
            group_size=group_size,
            mode="allow-bigger",
            sort_unique_ids=self.sort_unique_ids,
        )
        partitioner.dataset = dataset
        # Trigger partitioning
        partitions = [
            partitioner.load_partition(i) for i in range(partitioner.num_partitions)
        ]
        unique_natural_ids = [
            len(partition.unique("natural_id")) for partition in partitions
        ]
        self.assertEqual(unique_natural_ids, expected_num_unique_natural_ids)

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_natural_ids, group_size, expected_num_partitions,
        # expected_num_unique_natural_ids
        [
            [10, 10, 2, [2, 2, 2, 2, 2]],
            [100, 10, 2, [2, 2, 2, 2, 2]],
            [12, 6, 3, [3, 3]],
            # The cases in which the partitions should be smaller
            [10, 7, 2, [2, 2, 2]],
            [10, 3, 2, [2]],
        ]
    )
    def test_drop_reminder_mode_num_partitions_and_partition_sizes(
        self,
        num_rows: int,
        num_unique_natural_id: int,
        group_size: int,
        expected_num_unique_natural_ids: list[int],
    ) -> None:
        """Test drop reminder mode."""
        dataset = _create_dataset(num_rows, num_unique_natural_id)
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="natural_id",
            group_size=group_size,
            mode="drop-reminder",
            sort_unique_ids=self.sort_unique_ids,
        )
        partitioner.dataset = dataset
        # Trigger partitioning
        partitions = [
            partitioner.load_partition(i) for i in range(partitioner.num_partitions)
        ]
        unique_natural_ids = [
            len(partition.unique("natural_id")) for partition in partitions
        ]
        self.assertEqual(unique_natural_ids, expected_num_unique_natural_ids)

    @parameterized.expand(  # type: ignore
        # mode, num_rows, num_unique_natural_ids, group_size
        [
            ["strict", 10, 10, 2],
            ["allow-smaller", 10, 7, 2],
            ["allow-bigger", 10, 7, 2],
            ["drop-reminder", 10, 7, 2],
            ["strict", 12, 6, 3],
            ["allow-smaller", 12, 6, 3],
            ["allow-bigger", 12, 6, 3],
            ["drop-reminder", 12, 6, 3],
            ["allow-smaller", 10, 2, 3],
        ]
    )
    def test_no_overlapping_natural_ids(
        self,
        mode: Literal["allow-smaller", "allow-bigger", "drop-reminder", "strict"],
        num_rows: int,
        num_unique_natural_id: int,
        group_size: int,
    ) -> None:
        """Test that no natural_ids overlap across partitions."""
        dataset = _create_dataset(num_rows, num_unique_natural_id)
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="natural_id",
            group_size=group_size,
            mode=mode,
            sort_unique_ids=self.sort_unique_ids,
        )
        partitioner.dataset = dataset

        # Trigger partitioning
        partitions = [
            partitioner.load_partition(i) for i in range(partitioner.num_partitions)
        ]

        # Check for overlaps between partitions
        seen_natural_ids: set[str] = set()
        for partition in partitions:
            natural_ids_in_partition = set(partition.unique("natural_id"))

            # Check if there is any overlap with previously seen natural IDs
            overlap = seen_natural_ids.intersection(natural_ids_in_partition)
            self.assertTrue(
                len(overlap) == 0,
                f"Overlapping natural IDs found between partitions in mode: {mode}. "
                f"Overlapping IDs: {overlap}",
            )

            # Add the natural IDs from this partition to the seen set
            seen_natural_ids.update(natural_ids_in_partition)

    def test_group_size_bigger_than_num_unique_natural_ids_allow_smaller(self) -> None:
        """Test the allow-smaller mode with group size > number of unique natural ids.

        That's the only mode that should work in this scenario.
        """
        dataset = _create_dataset(num_rows=10, n_unique_natural_ids=2)
        expected_num_unique_natural_ids = [2]
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="natural_id",
            group_size=3,
            mode="allow-smaller",
            sort_unique_ids=self.sort_unique_ids,
        )
        partitioner.dataset = dataset
        # Trigger partitioning
        partitions = [
            partitioner.load_partition(i) for i in range(partitioner.num_partitions)
        ]
        unique_natural_ids = [
            len(partition.unique("natural_id")) for partition in partitions
        ]

        self.assertEqual(unique_natural_ids, expected_num_unique_natural_ids)

    def test_strict_mode_with_invalid_group_size(self) -> None:
        """Test strict mode raises if group_size does not divide unique IDs evenly."""
        dataset = _create_dataset(num_rows=10, n_unique_natural_ids=3)
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="natural_id",
            group_size=2,
            mode="strict",
            sort_unique_ids=self.sort_unique_ids,
        )
        partitioner.dataset = dataset
        with self.assertRaises(ValueError) as context:
            _ = partitioner.load_partition(0)
        self.assertIn(
            "Strict mode requires that the number of unique natural ids is perfectly "
            "divisible by the group size.",
            str(context.exception),
        )

    def test_too_big_group_size(self) -> None:
        """Test raises if the group size > than the number of unique natural ids."""
        n_unique_natural_ids = 3
        dataset = _create_dataset(
            num_rows=10, n_unique_natural_ids=n_unique_natural_ids
        )
        partitioner = GroupedNaturalIdPartitioner(
            partition_by="natural_id",
            group_size=n_unique_natural_ids + 1,
            mode="allow-bigger",
            sort_unique_ids=self.sort_unique_ids,
        )
        partitioner.dataset = dataset
        with self.assertRaises(ValueError) as context:
            _ = partitioner.load_partition(0)
        self.assertIn(
            "The group size needs to be smaller than the number of the unique "
            "natural ids unless you are using allow-smaller mode which will "
            "result in a single partition.",
            str(context.exception),
        )


if __name__ == "__main__":
    unittest.main()
