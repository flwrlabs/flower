# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Test ShardPartitioner."""

# pylint: disable=W0212, R0913
import unittest

from datasets import Dataset
from flwr_datasets.partitioner.shard_partitioner import ShardPartitioner


def _dummy_setup(  # pylint: disable=R0917
    num_rows: int,
    partition_by: str,
    num_partitions: int,
    num_shards_per_partition: int | None,
    shard_size: int | None,
    keep_incomplete_shard: bool = False,
) -> tuple[Dataset, ShardPartitioner]:
    """Create a dummy dataset for testing."""
    data = {
        partition_by: [i % 3 for i in range(num_rows)],
        "features": list(range(num_rows)),
    }
    dataset = Dataset.from_dict(data)
    partitioner = ShardPartitioner(
        num_partitions=num_partitions,
        num_shards_per_partition=num_shards_per_partition,
        partition_by=partition_by,
        shard_size=shard_size,
        keep_incomplete_shard=keep_incomplete_shard,
    )
    partitioner.dataset = dataset
    return dataset, partitioner


class TestShardPartitionerSpec1(unittest.TestCase):
    """Test first possible initialization of ShardPartitioner.

    Specify num_shards_per_partition and shard_size arguments.
    """

    def test_correct_num_partitions(self) -> None:
        """Test the correct number of partitions is created."""
        partition_by = "label"
        num_rows = 113
        num_partitions = 3
        num_shards_per_partition = 3
        shard_size = 10
        keep_incomplete_shard = False
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        _ = partitioner.load_partition(0)
        num_partitions_created = len(partitioner._partition_id_to_indices.keys())
        self.assertEqual(num_partitions_created, num_partitions)

    def test_correct_partition_sizes(self) -> None:
        """Test if the partitions sizes are as theoretically calculated."""
        partition_by = "label"
        num_rows = 113
        num_partitions = 3
        num_shards_per_partition = 3
        shard_size = 10
        keep_incomplete_shard = False
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        sizes = [len(partitioner.load_partition(i)) for i in range(num_partitions)]
        sizes = sorted(sizes)
        self.assertEqual(sizes, [30, 30, 30])

    def test_unique_samples(self) -> None:
        """Test if each partition has unique samples.

        (No duplicates along partitions).
        """
        partition_by = "label"
        num_rows = 113
        num_partitions = 3
        num_shards_per_partition = 3
        shard_size = 10
        keep_incomplete_shard = False
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        partitions = [
            partitioner.load_partition(i)["features"] for i in range(num_partitions)
        ]
        combined_list = [item for sublist in partitions for item in sublist]
        combined_set = set(combined_list)
        self.assertEqual(len(combined_list), len(combined_set))


class TestShardPartitionerSpec2(unittest.TestCase):
    """Test second possible initialization of ShardPartitioner.

    Specify shard_size and keep_incomplete_shard=False. This setting creates partitions
    that might have various sizes (each shard is same size).
    """

    def test_correct_num_partitions(self) -> None:
        """Test the correct number of partitions is created."""
        partition_by = "label"
        num_rows = 113
        num_partitions = 3
        num_shards_per_partition = None
        shard_size = 10
        keep_incomplete_shard = False
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        _ = partitioner.load_partition(0)
        num_partitions_created = len(partitioner._partition_id_to_indices.keys())
        self.assertEqual(num_partitions_created, num_partitions)

    def test_correct_partition_sizes(self) -> None:
        """Test if the partitions sizes are as theoretically calculated."""
        partition_by = "label"
        num_rows = 113
        num_partitions = 3
        num_shards_per_partition = None
        shard_size = 10
        keep_incomplete_shard = False
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        sizes = [len(partitioner.load_partition(i)) for i in range(num_partitions)]
        sizes = sorted(sizes)
        self.assertEqual(sizes, [30, 40, 40])

    def test_unique_samples(self) -> None:
        """Test if each partition has unique samples.

        (No duplicates along partitions).
        """
        partition_by = "label"
        num_rows = 113
        num_partitions = 3
        num_shards_per_partition = None
        shard_size = 10
        keep_incomplete_shard = False
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        partitions = [
            partitioner.load_partition(i)["features"] for i in range(num_partitions)
        ]
        combined_list = [item for sublist in partitions for item in sublist]
        combined_set = set(combined_list)
        self.assertEqual(len(combined_list), len(combined_set))


class TestShardPartitionerSpec3(unittest.TestCase):
    """Test third possible initialization of ShardPartitioner.

    Specify shard_size and keep_incomplete_shard=True. This setting creates partitions
    that might have various sizes (each shard is same size).
    """

    def test_correct_num_partitions(self) -> None:
        """Test the correct number of partitions is created."""
        partition_by = "label"
        num_rows = 113
        num_partitions = 3
        num_shards_per_partition = None
        shard_size = 10
        keep_incomplete_shard = True
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        _ = partitioner.load_partition(0)
        num_partitions_created = len(partitioner._partition_id_to_indices.keys())
        self.assertEqual(num_partitions_created, num_partitions)

    def test_correct_partition_sizes(self) -> None:
        """Test if the partitions sizes are as theoretically calculated."""
        partition_by = "label"
        num_rows = 113
        num_partitions = 3
        num_shards_per_partition = None
        shard_size = 10
        keep_incomplete_shard = True
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        sizes = [len(partitioner.load_partition(i)) for i in range(num_partitions)]
        sizes = sorted(sizes)
        self.assertEqual(sizes, [33, 40, 40])

    def test_unique_samples(self) -> None:
        """Test if each partition has unique samples.

        (No duplicates along partitions).
        """
        partition_by = "label"
        num_rows = 113
        num_partitions = 3
        num_shards_per_partition = None
        shard_size = 10
        keep_incomplete_shard = True
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        partitions = [
            partitioner.load_partition(i)["features"] for i in range(num_partitions)
        ]
        combined_list = [item for sublist in partitions for item in sublist]
        combined_set = set(combined_list)
        self.assertEqual(len(combined_list), len(combined_set))


class TestShardPartitionerSpec4(unittest.TestCase):
    """Test fourth possible initialization of ShardPartitioner.

    Specify num_shards_per_partition but not shard_size arguments.
    """

    def test_correct_num_partitions(self) -> None:
        """Test the correct number of partitions is created."""
        partition_by = "label"
        num_rows = 113
        num_partitions = 3
        num_shards_per_partition = 3
        shard_size = None
        keep_incomplete_shard = False
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        _ = partitioner.load_partition(0)
        num_partitions_created = len(partitioner._partition_id_to_indices.keys())
        self.assertEqual(num_partitions_created, num_partitions)

    def test_correct_partition_sizes(self) -> None:
        """Test if the partitions sizes are as theoretically calculated."""
        partition_by = "label"
        num_rows = 113
        num_partitions = 3
        num_shards_per_partition = 3
        shard_size = None
        keep_incomplete_shard = False
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        sizes = [len(partitioner.load_partition(i)) for i in range(num_partitions)]
        sizes = sorted(sizes)
        self.assertEqual(sizes, [36, 36, 36])

    def test_unique_samples(self) -> None:
        """Test if each partition has unique samples.

        (No duplicates along partitions).
        """
        partition_by = "label"
        num_rows = 113
        num_partitions = 3
        num_shards_per_partition = 3
        shard_size = None
        keep_incomplete_shard = False
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        partitions = [
            partitioner.load_partition(i)["features"] for i in range(num_partitions)
        ]
        combined_list = [item for sublist in partitions for item in sublist]
        combined_set = set(combined_list)
        self.assertEqual(len(combined_list), len(combined_set))


class TestShardPartitionerIncorrectSpec(unittest.TestCase):
    """Test the incorrect specification cases.

    The lack of correctness can be caused by the num_partitions, shard_size and
    num_shards_per_partition can create.
    """

    def test_incorrect_specification(self) -> None:
        """Test if the given specification makes the partitioning possible."""
        partition_by = "label"
        num_rows = 10
        num_partitions = 3
        num_shards_per_partition = 2
        shard_size = 10
        keep_incomplete_shard = False
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        with self.assertRaises(ValueError):
            _ = partitioner.load_partition(0)

    def test_too_big_shard_size(self) -> None:
        """Test if it is impossible to create an empty partition."""
        partition_by = "label"
        num_rows = 20
        num_partitions = 3
        num_shards_per_partition = None
        shard_size = 10
        keep_incomplete_shard = False
        _, partitioner = _dummy_setup(
            num_rows,
            partition_by,
            num_partitions,
            num_shards_per_partition,
            shard_size,
            keep_incomplete_shard,
        )
        with self.assertRaises(ValueError):
            _ = partitioner.load_partition(2).num_rows


if __name__ == "__main__":
    unittest.main()
