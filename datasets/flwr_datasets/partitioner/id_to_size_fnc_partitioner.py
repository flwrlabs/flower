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
"""IdToSizeFncPartitioner class."""

from collections.abc import Callable

import numpy as np

import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class IdToSizeFncPartitioner(Partitioner):
    """Base class for the deterministic size partitioning based on the `partition_id`.

    The client with `partition_id` has the following relationship regarding the number
    of samples.

    `partition_id_to_size_fn(partition_id)` ~ number of samples for `partition_id`

    If the function doesn't transform the `partition_id` it's a linear correlation
    between the number of sample for the partition and the value of `partition_id`. For
    instance, if the partition ids range from 1 to M, partition with id 1 gets 1 unit of
    data, client 2 gets 2 units, and so on, up to partition M which gets M units.

    Note that size corresponding to the `partition_id` is deterministic, yet in case of
    different dataset shuffling the assignment of samples to `partition_id` will vary.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    partition_id_to_size_fn : Callable
        Function that defines the relationship between partition id and the number of
        samples.
    """

    def __init__(
        self,
        num_partitions: int,
        partition_id_to_size_fn: Callable,  # type: ignore[type-arg]
    ) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
        self._num_partitions = num_partitions
        self._partition_id_to_size_fn = partition_id_to_size_fn

        self._partition_id_to_size: dict[int, int] = {}
        self._partition_id_to_indices: dict[int, list[int]] = {}
        # A flag to perform only a single compute to determine the indices
        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a single partition based on the partition index.

        The number of samples is dependent on the partition partition_id.

        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition: Dataset
            single dataset partition
        """
        # The partitioning is done lazily - only when the first partition is requested.
        # A single run creates the indices assignments for all the partition indices.
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    @property
    def partition_id_to_size(self) -> dict[int, int]:
        """Node id to the number of samples."""
        return self._partition_id_to_size

    @property
    def partition_id_to_indices(self) -> dict[int, list[int]]:
        """Node id to the list of indices."""
        return self._partition_id_to_indices

    def _determine_partition_id_to_size(self) -> None:
        """Determine data quantity associated with partition indices."""
        data_division_in_units = self._partition_id_to_size_fn(
            np.linspace(start=1, stop=self._num_partitions, num=self._num_partitions)
        )
        total_units: int | float = data_division_in_units.sum()
        # Normalize the units to get the fraction total dataset
        partition_sizes_as_fraction = data_division_in_units / total_units
        # Calculate the number of samples
        partition_sizes_as_num_of_samples = np.array(
            partition_sizes_as_fraction * len(self.dataset), dtype=np.int64
        )
        # Check if any sample is not allocated because of multiplication with fractions.
        assigned_samples = np.sum(partition_sizes_as_num_of_samples)
        left_unassigned_samples = len(self.dataset) - assigned_samples
        # If there is any sample(s) left unassigned, assign it to the largest partition.
        partition_sizes_as_num_of_samples[-1] += left_unassigned_samples
        for idx, partition_size in enumerate(partition_sizes_as_num_of_samples):
            self._partition_id_to_size[idx] = partition_size

        self._check_if_partition_id_to_size_possible()

    def _determine_partition_id_to_indices_if_needed(self) -> None:
        """Create an assignment of indices to the partition indices.."""
        if self._partition_id_to_indices_determined is True:
            return
        self._determine_partition_id_to_size()
        total_samples_assigned = 0
        for idx, quantity in self._partition_id_to_size.items():
            self._partition_id_to_indices[idx] = list(
                range(total_samples_assigned, total_samples_assigned + quantity)
            )
            total_samples_assigned += quantity
        self._partition_id_to_indices_determined = True

    def _check_if_partition_id_to_size_possible(self) -> None:
        all_positive = all(value >= 1 for value in self.partition_id_to_size.values())
        if not all_positive:
            raise ValueError(
                f"The given specification of the parameter num_partitions"
                f"={self._num_partitions} for the given dataset results "
                f"in the partitions sizes that are not greater than 0."
            )
