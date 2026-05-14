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
"""SizePartitioner class."""

import warnings
from collections.abc import Sequence

import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class SizePartitioner(Partitioner):
    """Partitioner that creates each partition with the size specified by a user.

    Parameters
    ----------
    partition_sizes : Sequence[int]
        The size of each partition. partition_id 0 will have partition_sizes[0]
        samples, partition_id 1 will have partition_sizes[1] samples, etc.

    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import SizePartitioner
    >>>
    >>> partition_sizes = [15_000, 5_000, 30_000]
    >>> partitioner = SizePartitioner(partition_sizes)
    >>> fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    """

    def __init__(self, partition_sizes: Sequence[int]) -> None:
        super().__init__()
        self._pre_ds_validate_partition_sizes(partition_sizes)
        self._partition_sizes = partition_sizes
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a single partition of the size of partition_sizes[partition_id].

        For example if given partition_sizes=[20_000, 10_000, 30_000],
        then partition_id=0 will return a partition of size 20_000,
        partition_id=1 will return a partition of size 10_000, etc.

        Parameters
        ----------
        partition_id : int
            The index that corresponds to the requested partition.

        Returns
        -------
        dataset_partition : Dataset
            Single dataset partition.
        """
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._determine_partition_id_to_indices_if_needed()
        return len(self._partition_sizes)

    @property
    def partition_id_to_indices(self) -> dict[int, list[int]]:
        """Partition id to indices (the result of partitioning)."""
        self._determine_partition_id_to_indices_if_needed()
        return self._partition_id_to_indices

    def _determine_partition_id_to_indices_if_needed(
        self,
    ) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return
        self._post_ds_validate_partition_sizes()
        start = 0
        end = 0
        for partition_id, partition_size in enumerate(self._partition_sizes):
            end += partition_size
            indices = list(range(start, end))
            self._partition_id_to_indices[partition_id] = indices
            start = end
        self._partition_id_to_indices_determined = True

    def _pre_ds_validate_partition_sizes(self, partition_sizes: Sequence[int]) -> None:
        """Check if the partition sizes are valid (no information about the dataset)."""
        if not isinstance(partition_sizes, Sequence):
            raise ValueError("Partition sizes must be a sequence.")
        if len(partition_sizes) == 0:
            raise ValueError("Partition sizes must not be empty.")
        if not all(
            isinstance(partition_size, int) for partition_size in partition_sizes
        ):
            raise ValueError("All partition sizes must be integers.")
        if not all(partition_size > 0 for partition_size in partition_sizes):
            raise ValueError("All partition sizes must be greater than zero.")

    def _post_ds_validate_partition_sizes(self) -> None:
        """Validate the partition sizes against the dataset size."""
        desired_partition_sizes = sum(self._partition_sizes)
        dataset_size = len(self.dataset)
        if desired_partition_sizes > dataset_size:
            raise ValueError(
                f"The sum of partition sizes sum({self._partition_sizes})"
                f"= {desired_partition_sizes} is greater than the size of"
                f" the dataset {dataset_size}."
            )
        if desired_partition_sizes < dataset_size:
            warnings.warn(
                f"The sum of partition sizes is {desired_partition_sizes}, which is"
                f"smaller than the size of the dataset: {dataset_size}. "
                f"Ignore this warning if it is the desired behavior.",
                stacklevel=1,
            )
