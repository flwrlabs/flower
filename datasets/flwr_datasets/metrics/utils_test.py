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
"""Tests for metrics utils."""


import unittest

import pandas as pd
from parameterized import parameterized, parameterized_class

import datasets
from datasets import ClassLabel
from flwr_datasets.metrics.utils import (
    _compute_counts,
    _compute_frequencies,
    compute_counts,
    compute_frequencies,
)
from flwr_datasets.partitioner import IidPartitioner


@parameterized_class(
    ("dataset", "result"),
    [
        (
            datasets.Dataset.from_dict({"feature": list(range(10)), "label": [0] * 10}),
            pd.DataFrame([[5], [5]], index=pd.Index([0, 1], name="Partition ID")),
        ),
        (
            datasets.Dataset.from_dict(
                {"feature": list(range(10)), "label": [0] * 5 + [1] * 5}
            ),
            pd.DataFrame([[5, 0], [0, 5]], index=pd.Index([0, 1], name="Partition ID")),
        ),
        (
            datasets.Dataset.from_dict(
                {"feature": list(range(10)), "label": [0, 0, 0, 1, 1] + [1, 1, 1, 1, 2]}
            ),
            pd.DataFrame(
                [[3, 2, 0], [0, 4, 1]], index=pd.Index([0, 1], name="Partition ID")
            ),
        ),
    ],
)
class TestPublicMetricsUtils(unittest.TestCase):
    """Test metrics utils."""

    dataset: datasets.Dataset
    result: pd.DataFrame

    def test_compute_counts(self) -> None:
        """Test if the counts are computed correctly."""
        iid_partitioner = IidPartitioner(num_partitions=2)
        iid_partitioner.dataset = self.dataset
        count = compute_counts(iid_partitioner, column_name="label")
        pd.testing.assert_frame_equal(count, self.result)

    def test_compute_frequencies(self) -> None:
        """Test if the frequencies are computed correctly."""
        iid_partitioner = IidPartitioner(num_partitions=2)
        iid_partitioner.dataset = self.dataset
        frequencies = compute_frequencies(iid_partitioner, column_name="label")
        result = self.result.div(self.result.sum(axis=1), axis=0)
        pd.testing.assert_frame_equal(frequencies, result)

    def test_compute_counts_with_verbose_label(self) -> None:
        """Test if the counts are computed correctly."""
        iid_partitioner = IidPartitioner(num_partitions=2)
        dataset = self.dataset
        new_col_names = [
            str(col_id) for col_id in range(len(self.dataset.unique("label")))
        ]
        dataset = dataset.cast_column(
            "label",
            ClassLabel(
                num_classes=len(self.dataset.unique("label")), names=new_col_names
            ),
        )
        iid_partitioner.dataset = dataset
        result = self.result.copy()
        result.columns = new_col_names
        count = compute_counts(iid_partitioner, column_name="label", verbose_names=True)
        pd.testing.assert_frame_equal(count, result)

    def test_compute_frequencies_with_verbose_label(self) -> None:
        """Test if the frequencies are computed correctly."""
        iid_partitioner = IidPartitioner(num_partitions=2)
        dataset = self.dataset
        new_col_names = [
            str(col_id) for col_id in range(len(self.dataset.unique("label")))
        ]
        dataset = dataset.cast_column(
            "label",
            ClassLabel(
                num_classes=len(self.dataset.unique("label")), names=new_col_names
            ),
        )
        iid_partitioner.dataset = dataset
        result = self.result.copy()
        result.columns = new_col_names
        result = result.div(result.sum(axis=1), axis=0)
        frequencies = compute_frequencies(
            iid_partitioner, column_name="label", verbose_names=True
        )
        pd.testing.assert_frame_equal(frequencies, result)

    def test_compute_count_with_smaller_max_partitions(self) -> None:
        """Test is compute_count works when the max_partitions<total partitions."""
        iid_partitioner = IidPartitioner(num_partitions=2)
        iid_partitioner.dataset = self.dataset
        count = compute_counts(
            iid_partitioner, column_name="label", max_num_partitions=1
        )
        pd.testing.assert_frame_equal(count, self.result.iloc[:1])

    def test_compute_count_with_bigger_max_partitions(self) -> None:
        """Test is compute_count works when the max_partitions>total partitions."""
        iid_partitioner = IidPartitioner(num_partitions=2)
        iid_partitioner.dataset = self.dataset
        count = compute_counts(
            iid_partitioner, column_name="label", max_num_partitions=3
        )
        pd.testing.assert_frame_equal(count, self.result)


class TestPrivateMetricsUtils(unittest.TestCase):
    """Test metrics utils."""

    @parameterized.expand(  # type: ignore
        [
            ([1, 2, 2, 3], [1, 2, 3, 4], pd.Series([1, 2, 1, 0], index=[1, 2, 3, 4])),
            ([], [1, 2, 3], pd.Series([0, 0, 0], index=[1, 2, 3])),
            ([1, 1, 2], [1, 2, 3, 4], pd.Series([2, 1, 0, 0], index=[1, 2, 3, 4])),
        ]
    )
    def test__compute_counts(self, labels, unique_labels, expected) -> None:
        """Test if the counts are computed correctly."""
        result = _compute_counts(labels, unique_labels)
        pd.testing.assert_series_equal(result, expected)

    @parameterized.expand(  # type: ignore
        [
            (
                [1, 1, 2, 2, 2, 3],
                [1, 2, 3, 4],
                pd.Series([0.3333, 0.5, 0.1667, 0.0], index=[1, 2, 3, 4]),
            ),
            ([], [1, 2, 3], pd.Series([0.0, 0.0, 0.0], index=[1, 2, 3])),
            (
                ["a", "b", "b", "c"],
                ["a", "b", "c", "d"],
                pd.Series([0.25, 0.50, 0.25, 0.0], index=["a", "b", "c", "d"]),
            ),
        ]
    )
    def test_compute_distribution(self, labels, unique_labels, expected) -> None:
        """Test if the distributions are computed correctly."""
        result = _compute_frequencies(labels, unique_labels)
        pd.testing.assert_series_equal(result, expected, atol=0.001)

    @parameterized.expand(  # type: ignore
        [
            (["a", "b", "b", "c"], ["a", "b", "c"]),
            ([1, 2, 2, 3, 3, 3, 4], [1, 2, 3, 4]),
        ]
    )
    def test_distribution_sum_to_one(self, labels, unique_labels) -> None:
        """Test if distributions sum up to one."""
        result = _compute_frequencies(labels, unique_labels)
        self.assertAlmostEqual(result.sum(), 1.0)

    def test_compute_counts_non_unique_labels(self) -> None:
        """Test if not having the unique labels raises ValueError."""
        labels = [1, 2, 3]
        unique_labels = [1, 2, 2, 3]
        with self.assertRaises(ValueError):
            _compute_counts(labels, unique_labels)

    def test_compute_distribution_non_unique_labels(self) -> None:
        """Test if not having the unique labels raises ValueError."""
        labels = [1, 1, 2, 3]
        unique_labels = [1, 1, 2, 3]
        with self.assertRaises(ValueError):
            _compute_frequencies(labels, unique_labels)


if __name__ == "__main__":
    unittest.main()
