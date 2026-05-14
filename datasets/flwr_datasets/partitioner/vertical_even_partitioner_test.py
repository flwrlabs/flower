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
"""VerticalEvenPartitioner class tests."""

# mypy: disable-error-code=list-item
import unittest

import numpy as np

from datasets import Dataset
from flwr_datasets.partitioner.vertical_even_partitioner import VerticalEvenPartitioner


def _create_dummy_dataset(column_names: list[str], num_rows: int = 100) -> Dataset:
    """Create a dummy dataset with random data for testing."""
    data = {}
    rng = np.random.default_rng(seed=42)
    for col in column_names:
        # Just numeric data; could also be strings, categoricals, etc.
        data[col] = rng.integers(0, 100, size=num_rows).tolist()
    return Dataset.from_dict(data)


class TestVerticalEvenPartitioner(unittest.TestCase):
    """Unit tests for VerticalEvenPartitioner."""

    def test_init_with_invalid_num_partitions(self) -> None:
        """Test that initializing with an invalid number of partitions."""
        with self.assertRaises(ValueError):
            VerticalEvenPartitioner(num_partitions=0)

    def test_init_with_invalid_active_party_mode(self) -> None:
        """Test initialization with invalid active_party_columns_mode."""
        with self.assertRaises(ValueError):
            VerticalEvenPartitioner(
                num_partitions=2,
                active_party_columns_mode="invalid_mode",  # type: ignore[arg-type]
            )

    def test_init_with_non_string_drop_columns(self) -> None:
        """Test initialization with non-string elements in drop_columns."""
        with self.assertRaises(TypeError):
            VerticalEvenPartitioner(num_partitions=2, drop_columns=[1, "a", 3])

    def test_init_with_non_string_shared_columns(self) -> None:
        """Test initialization with non-string elements in shared_columns."""
        with self.assertRaises(TypeError):
            VerticalEvenPartitioner(num_partitions=2, shared_columns=["col1", 123])

    def test_init_with_non_string_active_party_column(self) -> None:
        """Test initialization with non-string elements in active_party_column."""
        with self.assertRaises(TypeError):
            VerticalEvenPartitioner(
                num_partitions=2, active_party_columns=["col1", None]
            )

    def test_partitioning_basic(self) -> None:
        """Test basic partitioning with no special columns or dropping."""
        columns = ["feature1", "feature2", "feature3", "feature4"]
        dataset = _create_dummy_dataset(columns, num_rows=50)
        partitioner = VerticalEvenPartitioner(num_partitions=2, shuffle=False)
        partitioner.dataset = dataset

        self.assertEqual(partitioner.num_partitions, 2)

        p0 = partitioner.load_partition(0)
        p1 = partitioner.load_partition(1)

        self.assertEqual(len(p0.column_names), 2)
        self.assertEqual(len(p1.column_names), 2)
        self.assertIn("feature1", p0.column_names)
        self.assertIn("feature2", p0.column_names)
        self.assertIn("feature3", p1.column_names)
        self.assertIn("feature4", p1.column_names)

    def test_partitioning_with_drop_columns(self) -> None:
        """Test partitioning while dropping some columns."""
        columns = ["feature1", "feature2", "drop_me", "feature3", "feature4"]
        dataset = _create_dummy_dataset(columns, num_rows=50)
        partitioner = VerticalEvenPartitioner(
            num_partitions=2, drop_columns=["drop_me"], shuffle=False, seed=42
        )
        partitioner.dataset = dataset

        p0 = partitioner.load_partition(0)
        p1 = partitioner.load_partition(1)
        all_partition_columns = p0.column_names + p1.column_names

        # The drop_me should not be in any partition
        self.assertNotIn("drop_me", all_partition_columns)
        # The rest of columns should be distributed
        self.assertIn("feature1", all_partition_columns)
        self.assertIn("feature2", all_partition_columns)
        self.assertIn("feature3", all_partition_columns)
        self.assertIn("feature4", all_partition_columns)

    def test_partitioning_with_shared_columns(self) -> None:
        """Test that shared columns are present in all partitions."""
        columns = ["f1", "f2", "f3", "f4", "shared_col"]
        dataset = _create_dummy_dataset(columns, num_rows=50)
        partitioner = VerticalEvenPartitioner(
            num_partitions=2, shared_columns=["shared_col"], shuffle=False, seed=42
        )
        partitioner.dataset = dataset

        p0 = partitioner.load_partition(0)
        p1 = partitioner.load_partition(1)

        self.assertIn("shared_col", p0.column_names)
        self.assertIn("shared_col", p1.column_names)

    def test_partitioning_with_active_party_columns_add_to_last(self) -> None:
        """Test active party columns are appended to the last partition."""
        columns = ["f1", "f2", "f3", "f4", "income"]
        dataset = _create_dummy_dataset(columns, num_rows=50)
        partitioner = VerticalEvenPartitioner(
            num_partitions=2,
            active_party_columns=["income"],
            active_party_columns_mode="add_to_last",
            shuffle=False,
            seed=42,
        )
        partitioner.dataset = dataset

        p0 = partitioner.load_partition(0)
        p1 = partitioner.load_partition(1)

        # The income should be only in the last partition
        self.assertNotIn("income", p0.column_names)
        self.assertIn("income", p1.column_names)

    def test_partitioning_with_active_party_columns_create_as_first(self) -> None:
        """Test creating a new partition solely for active party columns."""
        columns = ["f1", "f2", "f3", "f4", "income"]
        dataset = _create_dummy_dataset(columns, num_rows=50)
        partitioner = VerticalEvenPartitioner(
            num_partitions=2,
            active_party_columns=["income"],
            active_party_columns_mode="create_as_first",
            shuffle=False,
        )
        partitioner.dataset = dataset

        # The first partition should be just the active party columns
        # and then two more partitions from original splitting.
        self.assertEqual(partitioner.num_partitions, 3)

        p0 = partitioner.load_partition(0)  # active party partition
        p1 = partitioner.load_partition(1)
        p2 = partitioner.load_partition(2)

        self.assertEqual(p0.column_names, ["income"])
        self.assertIn("f1", p1.column_names)
        self.assertIn("f2", p1.column_names)
        self.assertIn("f3", p2.column_names)
        self.assertIn("f4", p2.column_names)

    def test_partitioning_with_nonexistent_active_party_column(self) -> None:
        """Test that a ValueError is raised if active party column does not exist."""
        columns = ["f1", "f2", "f3", "f4"]
        dataset = _create_dummy_dataset(columns, num_rows=50)
        partitioner = VerticalEvenPartitioner(
            num_partitions=2,
            active_party_columns=["income"],  # Not present in dataset
            active_party_columns_mode="add_to_last",
            shuffle=False,
        )
        partitioner.dataset = dataset

        with self.assertRaises(ValueError) as context:
            partitioner.load_partition(0)
        self.assertIn("Active party column 'income' not found", str(context.exception))

    def test_partitioning_with_nonexistent_shared_columns(self) -> None:
        """Test that a ValueError is raised if shared column does not exist."""
        columns = ["f1", "f2", "f3"]
        dataset = _create_dummy_dataset(columns, num_rows=50)
        partitioner = VerticalEvenPartitioner(
            num_partitions=2, shared_columns=["nonexistent_col"], shuffle=False
        )
        partitioner.dataset = dataset

        with self.assertRaises(ValueError) as context:
            partitioner.load_partition(0)
        self.assertIn(
            "Shared column 'nonexistent_col' not found", str(context.exception)
        )


if __name__ == "__main__":
    unittest.main()
