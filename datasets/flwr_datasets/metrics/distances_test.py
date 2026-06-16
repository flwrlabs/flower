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
"""Tests for distribution distance metrics."""


import unittest
from math import log2, sqrt

import pandas as pd

import datasets
from flwr_datasets.metrics import (
    compute_hellinger_distances,
    compute_jensen_shannon_distances,
)
from flwr_datasets.partitioner import IidPartitioner


class TestDistributionDistanceMetrics(unittest.TestCase):
    """Test distribution distance metrics."""

    def test_distances_are_zero_for_matching_partition_distributions(self) -> None:
        """Test distances are zero when partitions match the full distribution."""
        dataset = datasets.Dataset.from_dict(
            {"feature": list(range(4)), "label": [0, 1, 0, 1]}
        )
        iid_partitioner = IidPartitioner(num_partitions=2)
        iid_partitioner.dataset = dataset
        expected_hellinger = pd.Series(
            [0.0, 0.0],
            index=pd.Index([0, 1], name="Partition ID"),
            name="Hellinger distance",
        )
        expected_jensen_shannon = pd.Series(
            [0.0, 0.0],
            index=pd.Index([0, 1], name="Partition ID"),
            name="Jensen-Shannon distance",
        )

        pd.testing.assert_series_equal(
            compute_hellinger_distances(iid_partitioner, column_name="label"),
            expected_hellinger,
        )
        pd.testing.assert_series_equal(
            compute_jensen_shannon_distances(iid_partitioner, column_name="label"),
            expected_jensen_shannon,
        )

    def test_hellinger_distances_for_label_skew(self) -> None:
        """Test Hellinger distance for fully skewed label partitions."""
        dataset = datasets.Dataset.from_dict(
            {"feature": list(range(20)), "label": [0] * 10 + [1] * 10}
        )
        iid_partitioner = IidPartitioner(num_partitions=2)
        iid_partitioner.dataset = dataset
        expected_distance = sqrt((1 - sqrt(0.5)) ** 2 + (sqrt(0.5)) ** 2) / sqrt(2)
        expected = pd.Series(
            [expected_distance, expected_distance],
            index=pd.Index([0, 1], name="Partition ID"),
            name="Hellinger distance",
        )

        distances = compute_hellinger_distances(iid_partitioner, column_name="label")

        pd.testing.assert_series_equal(distances, expected)

    def test_jensen_shannon_distances_for_label_skew(self) -> None:
        """Test Jensen-Shannon distance for fully skewed label partitions."""
        dataset = datasets.Dataset.from_dict(
            {"feature": list(range(20)), "label": [0] * 10 + [1] * 10}
        )
        iid_partitioner = IidPartitioner(num_partitions=2)
        iid_partitioner.dataset = dataset
        divergence = 0.5 * (
            log2(1 / 0.75) + 0.5 * log2(0.5 / 0.75) + 0.5 * log2(0.5 / 0.25)
        )
        expected_distance = sqrt(divergence)
        expected = pd.Series(
            [expected_distance, expected_distance],
            index=pd.Index([0, 1], name="Partition ID"),
            name="Jensen-Shannon distance",
        )

        distances = compute_jensen_shannon_distances(
            iid_partitioner, column_name="label"
        )

        pd.testing.assert_series_equal(distances, expected)

    def test_distances_respect_max_num_partitions(self) -> None:
        """Test max_num_partitions limits the returned partition distances."""
        dataset = datasets.Dataset.from_dict(
            {"feature": list(range(20)), "label": [0] * 10 + [1] * 10}
        )
        iid_partitioner = IidPartitioner(num_partitions=2)
        iid_partitioner.dataset = dataset

        distances = compute_hellinger_distances(
            iid_partitioner, column_name="label", max_num_partitions=1
        )

        self.assertEqual(list(distances.index), [0])

    def test_distances_reject_invalid_max_num_partitions(self) -> None:
        """Test invalid max_num_partitions raises ValueError."""
        dataset = datasets.Dataset.from_dict(
            {"feature": list(range(4)), "label": [0, 1, 0, 1]}
        )
        iid_partitioner = IidPartitioner(num_partitions=2)
        iid_partitioner.dataset = dataset

        with self.assertRaises(ValueError):
            compute_hellinger_distances(
                iid_partitioner, column_name="label", max_num_partitions=0
            )


if __name__ == "__main__":
    unittest.main()
