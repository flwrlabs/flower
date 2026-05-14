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
"""VerticalSizePartitioner class tests."""

# mypy: disable-error-code=arg-type
# pylint: disable=R0902, R0913
import unittest

import numpy as np

from datasets import Dataset
from flwr_datasets.partitioner.vertical_size_partitioner import VerticalSizePartitioner


def _create_dummy_dataset(column_names: list[str], num_rows: int = 100) -> Dataset:
    """Create a dataset with random integer data."""
    rng = np.random.default_rng(seed=42)
    data = {col: rng.integers(0, 100, size=num_rows).tolist() for col in column_names}
    return Dataset.from_dict(data)


class TestVerticalSizePartitioner(unittest.TestCase):
    """Tests for VerticalSizePartitioner."""

    def test_init_invalid_partition_sizes_type(self) -> None:
        """Check ValueError if partition_sizes is not a list."""
        with self.assertRaises(ValueError):
            VerticalSizePartitioner(partition_sizes="not_a_list")

    def test_init_mixed_partition_sizes_types(self) -> None:
        """Check ValueError if partition_sizes mix int and float."""
        with self.assertRaises(ValueError):
            VerticalSizePartitioner(partition_sizes=[0.5, 1])

    def test_init_float_partitions_sum_not_one(self) -> None:
        """Check ValueError if float partitions do not sum to 1."""
        with self.assertRaises(ValueError):
            VerticalSizePartitioner(partition_sizes=[0.3, 0.3])

    def test_init_float_partitions_out_of_range(self) -> None:
        """Check ValueError if any float partition <0 or >1."""
        with self.assertRaises(ValueError):
            VerticalSizePartitioner(partition_sizes=[-0.5, 1.5])

    def test_init_int_partitions_negative(self) -> None:
        """Check ValueError if any int partition size is negative."""
        with self.assertRaises(ValueError):
            VerticalSizePartitioner(partition_sizes=[5, -1])

    def test_init_invalid_mode(self) -> None:
        """Check ValueError if active_party_columns_mode is invalid."""
        with self.assertRaises(ValueError):
            VerticalSizePartitioner(
                partition_sizes=[2, 2], active_party_columns_mode="invalid"
            )

    def test_init_active_party_column_invalid_type(self) -> None:
        """Check ValueError if active_party_column is not str/list."""
        with self.assertRaises(TypeError):
            VerticalSizePartitioner(partition_sizes=[2, 2], active_party_columns=123)

    def test_partitioning_with_int_sizes(self) -> None:
        """Check correct partitioning with integer sizes."""
        columns = ["f1", "f2", "f3", "f4", "f5"]
        dataset = _create_dummy_dataset(columns)
        partitioner = VerticalSizePartitioner(partition_sizes=[2, 3], shuffle=False)
        partitioner.dataset = dataset
        p0 = partitioner.load_partition(0)
        p1 = partitioner.load_partition(1)
        self.assertEqual(len(p0.column_names), 2)
        self.assertEqual(len(p1.column_names), 3)

    def test_partitioning_with_fraction_sizes(self) -> None:
        """Check correct partitioning with fraction sizes."""
        columns = ["f1", "f2", "f3", "f4"]
        dataset = _create_dummy_dataset(columns)
        partitioner = VerticalSizePartitioner(partition_sizes=[0.5, 0.5], shuffle=False)
        partitioner.dataset = dataset
        p0 = partitioner.load_partition(0)
        p1 = partitioner.load_partition(1)
        self.assertEqual(len(p0.column_names), 2)
        self.assertEqual(len(p1.column_names), 2)

    def test_partitioning_with_drop_columns(self) -> None:
        """Check dropping specified columns before partitioning."""
        columns = ["f1", "drop_me", "f2", "f3"]
        dataset = _create_dummy_dataset(columns)
        partitioner = VerticalSizePartitioner(
            partition_sizes=[2, 1], drop_columns=["drop_me"], shuffle=False
        )
        partitioner.dataset = dataset
        p0 = partitioner.load_partition(0)
        p1 = partitioner.load_partition(1)
        all_cols = p0.column_names + p1.column_names
        self.assertNotIn("drop_me", all_cols)

    def test_partitioning_with_shared_columns(self) -> None:
        """Check shared columns added to every partition."""
        columns = ["f1", "f2", "shared"]
        dataset = _create_dummy_dataset(columns)
        partitioner = VerticalSizePartitioner(
            partition_sizes=[1, 1], shared_columns=["shared"], shuffle=False
        )
        partitioner.dataset = dataset
        p0 = partitioner.load_partition(0)
        p1 = partitioner.load_partition(1)
        self.assertIn("shared", p0.column_names)
        self.assertIn("shared", p1.column_names)

    def test_partitioning_with_active_party_add_to_last(self) -> None:
        """Check active party columns added to the last partition."""
        columns = ["f1", "f2", "label"]
        dataset = _create_dummy_dataset(columns)
        partitioner = VerticalSizePartitioner(
            partition_sizes=[2],
            active_party_columns="label",
            active_party_columns_mode="add_to_last",
            shuffle=False,
        )
        partitioner.dataset = dataset
        p0 = partitioner.load_partition(0)
        self.assertIn("label", p0.column_names)

    def test_partitioning_with_active_party_create_as_first(self) -> None:
        """Check creating a new first partition for active party cols."""
        columns = ["f1", "f2", "label"]
        dataset = _create_dummy_dataset(columns)
        partitioner = VerticalSizePartitioner(
            partition_sizes=[2],
            active_party_columns="label",
            active_party_columns_mode="create_as_first",
            shuffle=False,
        )
        partitioner.dataset = dataset
        self.assertEqual(partitioner.num_partitions, 2)
        p0 = partitioner.load_partition(0)
        p1 = partitioner.load_partition(1)
        self.assertEqual(p0.column_names, ["label"])
        self.assertIn("f1", p1.column_names)
        self.assertIn("f2", p1.column_names)

    def test_partitioning_with_nonexistent_shared_column(self) -> None:
        """Check ValueError if shared column does not exist."""
        columns = ["f1", "f2"]
        dataset = _create_dummy_dataset(columns)
        partitioner = VerticalSizePartitioner(
            partition_sizes=[1], shared_columns=["nonexistent"], shuffle=False
        )
        partitioner.dataset = dataset
        with self.assertRaises(ValueError):
            partitioner.load_partition(0)

    def test_partitioning_with_nonexistent_active_party_column(self) -> None:
        """Check ValueError if active party column does not exist."""
        columns = ["f1", "f2"]
        dataset = _create_dummy_dataset(columns)
        partitioner = VerticalSizePartitioner(
            partition_sizes=[1], active_party_columns="missing_label", shuffle=False
        )
        partitioner.dataset = dataset
        with self.assertRaises(ValueError):
            partitioner.load_partition(0)

    def test_sum_of_int_partition_sizes_exceeds_num_columns(self) -> None:
        """Check ValueError if sum of int sizes > total columns."""
        columns = ["f1", "f2"]
        dataset = _create_dummy_dataset(columns)
        partitioner = VerticalSizePartitioner(partition_sizes=[3], shuffle=False)
        partitioner.dataset = dataset
        with self.assertRaises(ValueError):
            partitioner.load_partition(0)

    def test_sum_of_int_partition_sizes_indirectly_exceeds_num_columns(self) -> None:
        """Check ValueError if sum of int sizes > total columns."""
        columns = ["f1", "f2", "f3"]
        dataset = _create_dummy_dataset(columns)
        partitioner = VerticalSizePartitioner(
            partition_sizes=[1, 1], drop_columns=["f3", "f2"], shuffle=False
        )
        partitioner.dataset = dataset
        with self.assertRaises(ValueError):
            partitioner.load_partition(0)

    def test_sum_of_int_partition_sizes_is_smaller_than_num_columns(self) -> None:
        """Check ValueError if sum of int sizes < total columns."""
        columns = ["f1", "f2", "f3"]
        dataset = _create_dummy_dataset(columns)
        partitioner = VerticalSizePartitioner(partition_sizes=[2], shuffle=False)
        partitioner.dataset = dataset
        with self.assertRaises(ValueError):
            partitioner.load_partition(0)


if __name__ == "__main__":
    unittest.main()
