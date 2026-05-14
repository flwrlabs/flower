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
"""Tests for vertical partitioner utilities."""

import unittest
from typing import Any, Literal

from flwr_datasets.partitioner.vertical_partitioner_utils import (
    _add_active_party_columns,
    _list_split,
)


class TestVerticalPartitionerUtils(unittest.TestCase):
    """Tests for _list_split and _add_active_party_columns utilities."""

    def test_list_split_basic_splitting(self) -> None:
        """Check equal splitting with divisible lengths."""
        lst = [1, 2, 3, 4, 5, 6]
        result = _list_split(lst, 3)
        expected = [[1, 2], [3, 4], [5, 6]]
        self.assertEqual(result, expected)

    def test_list_split_uneven_splitting(self) -> None:
        """Check uneven splitting with non-divisible lengths."""
        lst = [10, 20, 30, 40, 50]
        result = _list_split(lst, 2)
        expected = [[10, 20, 30], [40, 50]]
        self.assertEqual(result, expected)

    def test_list_split_single_sublist(self) -> None:
        """Check that single sublist returns the full list."""
        lst = [1, 2, 3]
        result = _list_split(lst, 1)
        expected = [[1, 2, 3]]
        self.assertEqual(result, expected)

    def test_list_split_more_sublists_than_elements(self) -> None:
        """Check extra sublists are empty when count exceeds length."""
        lst = [42]
        result = _list_split(lst, 3)
        expected = [[42], [], []]
        self.assertEqual(result, expected)

    def test_list_split_empty_list(self) -> None:
        """Check splitting empty list produces empty sublists."""
        lst: list[Any] = []
        result = _list_split(lst, 3)
        expected: list[list[Any]] = [[], [], []]
        self.assertEqual(result, expected)

    def test_list_split_invalid_num_sublists(self) -> None:
        """Check ValueError when sublist count is zero or negative."""
        lst = [1, 2, 3]
        with self.assertRaises(ValueError):
            _list_split(lst, 0)

    def test_add_to_first(self) -> None:
        """Check adding active cols to the first partition."""
        partition_columns = [["col1", "col2"], ["col3"], ["col4"]]
        active_party_columns = ["active1", "active2"]
        mode: Literal["add_to_first"] = "add_to_first"
        result = _add_active_party_columns(
            active_party_columns, mode, partition_columns
        )
        self.assertEqual(
            result, [["col1", "col2", "active1", "active2"], ["col3"], ["col4"]]
        )

    def test_add_to_last(self) -> None:
        """Check adding active cols to the last partition."""
        partition_columns = [["col1", "col2"], ["col3"], ["col4"]]
        active_party_columns = ["active"]
        mode: Literal["add_to_last"] = "add_to_last"
        result = _add_active_party_columns(
            active_party_columns, mode, partition_columns
        )
        self.assertEqual(result, [["col1", "col2"], ["col3"], ["col4", "active"]])

    def test_create_as_first(self) -> None:
        """Check creating a new first partition for active cols."""
        partition_columns = [["col1"], ["col2"]]
        active_party_columns = ["active1", "active2"]
        mode: Literal["create_as_first"] = "create_as_first"
        result = _add_active_party_columns(
            active_party_columns, mode, partition_columns
        )
        self.assertEqual(result, [["active1", "active2"], ["col1"], ["col2"]])

    def test_create_as_last(self) -> None:
        """Check creating a new last partition for active cols."""
        partition_columns = [["col1"], ["col2"]]
        active_party_columns = ["active1", "active2"]
        mode: Literal["create_as_last"] = "create_as_last"
        result = _add_active_party_columns(
            active_party_columns, mode, partition_columns
        )
        self.assertEqual(result, [["col1"], ["col2"], ["active1", "active2"]])

    def test_add_to_all(self) -> None:
        """Check adding active cols to all partitions."""
        partition_columns = [["col1"], ["col2", "col3"], ["col4"]]
        active_party_columns = ["active"]
        mode: Literal["add_to_all"] = "add_to_all"
        result = _add_active_party_columns(
            active_party_columns, mode, partition_columns
        )
        self.assertEqual(
            result, [["col1", "active"], ["col2", "col3", "active"], ["col4", "active"]]
        )

    def test_add_to_specific_partition_valid_index(self) -> None:
        """Check adding active cols to a specific valid partition."""
        partition_columns = [["col1"], ["col2"], ["col3"]]
        active_party_columns = ["active1", "active2"]
        mode: int = 1
        result = _add_active_party_columns(
            active_party_columns, mode, partition_columns
        )
        self.assertEqual(result, [["col1"], ["col2", "active1", "active2"], ["col3"]])

    def test_add_to_specific_partition_invalid_index(self) -> None:
        """Check ValueError when partition index is invalid."""
        partition_columns = [["col1"], ["col2"]]
        active_party_columns = ["active"]
        mode: int = 5
        with self.assertRaises(ValueError) as context:
            _add_active_party_columns(active_party_columns, mode, partition_columns)
        self.assertIn("Invalid partition index", str(context.exception))


if __name__ == "__main__":
    unittest.main()
