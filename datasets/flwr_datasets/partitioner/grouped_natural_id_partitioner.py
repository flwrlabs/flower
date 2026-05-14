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
"""Grouped natural id partitioner class that works with Hugging Face Datasets."""

from typing import Any, Literal

import numpy as np

import datasets
from flwr_datasets.common.typing import NDArrayInt
from flwr_datasets.partitioner.partitioner import Partitioner


class GroupedNaturalIdPartitioner(Partitioner):
    """Partition dataset by creating groups of natural ids.

    Conceptually, you can think of this partitioner as a way of creating an organization
    of x users instead of each user represetning a separate partition. You can change
    the nature of the problem from cross-device to cross-silo (cross organization).

    Parameters
    ----------
    partition_by: str
        The name of the column that contains the unique values of partitions.
    group_size: int
        The number of unique ids that will be placed in a single group.
    mode: Literal["allow-smaller", "allow-bigger", "drop-reminder", ""strict"]
        The mode that will be used to handle the remainder of the unique ids.
        - "allow-smaller": The last group can be smaller than the group_size.
        - "allow-bigger": The first group can be bigger than the group_size.
        - "drop-reminder": The last group will be dropped if it is smaller than the
        group_size.
        - "strict": Raises a ValueError if the remainder is not zero. In this mode, you
        expect each group to have the same size.
    sort_unique_ids: bool
        If True, the unique natural ids will be sorted before creating the groups.

    Examples
    --------
    Partition users in the "sentiment140" (aka Twitter) dataset into groups of two
    users following the default mode:

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import GroupedNaturalIdPartitioner
    >>>
    >>> partitioner = GroupedNaturalIdPartitioner(partition_by="user", group_size=2)
    >>> fds = FederatedDataset(dataset="sentiment140",
    >>>                        partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    """

    def __init__(
        self,
        partition_by: str,
        group_size: int,
        mode: Literal[
            "allow-smaller", "allow-bigger", "drop-reminder", "strict"
        ] = "allow-smaller",
        sort_unique_ids: bool = False,
    ) -> None:
        super().__init__()
        self._partition_id_to_natural_ids: dict[int, list[Any]] = {}
        self._natural_id_to_partition_id: dict[Any, int] = {}
        self._partition_id_to_indices: dict[int, NDArrayInt] = {}
        self._partition_by = partition_by
        self._mode = mode
        self._sort_unique_ids = sort_unique_ids

        if group_size < 0:
            raise ValueError("group_size must be a positive integer")
        self._group_size = group_size

    def _create_int_partition_id_to_natural_id(self) -> None:
        """Create a mapping from int indices to unique client ids from dataset.

        Natural ids come from the column specified in `partition_by`.
        """
        unique_natural_ids = self.dataset.unique(self._partition_by)
        if self._mode != "allow-smaller" and self._group_size > len(unique_natural_ids):
            raise ValueError(
                "The group size needs to be smaller than the number of the unique "
                "natural ids unless you are using allow-smaller mode which will "
                "result in a single partition."
            )
        if self._sort_unique_ids:
            unique_natural_ids = sorted(unique_natural_ids)
        num_unique_natural_ids = len(unique_natural_ids)
        remainder = num_unique_natural_ids % self._group_size
        num_groups = num_unique_natural_ids // self._group_size
        if num_groups == 0 and self._mode == "allow-smaller":
            num_groups = 1
            remainder = 0
        # Note that the number of groups might be different that this number
        # due to certain modes, it's a base value.

        if self._mode == "allow-bigger":
            groups_of_natural_ids = np.array_split(unique_natural_ids, num_groups)
        elif self._mode == "drop-reminder":
            # Narrow down the unique_natural_ids to not have a bigger group
            # which is the behavior of the np.array_split
            unique_natural_ids = unique_natural_ids[
                : int(num_groups * self._group_size)
            ]
            groups_of_natural_ids = np.array_split(unique_natural_ids, num_groups)
        elif self._mode == "allow-smaller":
            if remainder > 0:
                last_group_ids = unique_natural_ids[-remainder:]
            unique_natural_ids = unique_natural_ids[
                : int(num_groups * self._group_size)
            ]
            groups_of_natural_ids = np.array_split(unique_natural_ids, num_groups)
            if remainder > 0:
                groups_of_natural_ids.append(np.array(last_group_ids))
        elif self._mode == "strict":
            if remainder != 0:
                raise ValueError(
                    "Strict mode requires that the number of unique natural ids is "
                    "perfectly divisible by the group size. "
                    f"Found remainder: {remainder}. Please pass the group_size that "
                    f"enables strict mode or relax the mode parameter. Refer to the "
                    f"documentation of the mode parameter for the available modes."
                )
            groups_of_natural_ids = np.array_split(unique_natural_ids, num_groups)
        else:
            raise ValueError(
                f"Given {self._mode} is not a valid mode. Refer to the documentation of"
                " the mode parameter for the available modes."
            )

        self._partition_id_to_natural_ids = {}
        for group_of_natural_ids_id, group_of_natural_ids in enumerate(
            groups_of_natural_ids
        ):
            self._partition_id_to_natural_ids[group_of_natural_ids_id] = (
                group_of_natural_ids.tolist()
            )

    def _create_natural_id_to_int_partition_id(self) -> None:
        """Create a mapping from unique client ids from dataset to int indices.

        Natural ids come from the column specified in `partition_by`. This object is
        inverse of the `self._partition_id_to_natural_id`. This method assumes that
        `self._partition_id_to_natural_id` already exists.
        """
        self._natural_id_to_partition_id = {}
        for partition_id, natural_ids in self._partition_id_to_natural_ids.items():
            for natural_id in natural_ids:
                self._natural_id_to_partition_id[natural_id] = partition_id

    def _create_partition_id_to_indices(self) -> None:
        natural_id_to_indices = {}  # type: ignore
        natural_ids = np.array(self.dataset[self._partition_by])

        for index, natural_id in enumerate(natural_ids):
            if natural_id not in natural_id_to_indices:
                natural_id_to_indices[natural_id] = []
            natural_id_to_indices[natural_id].append(index)

        self._partition_id_to_indices = {}
        for partition_id, natural_id_group in self._partition_id_to_natural_ids.items():
            indices = []
            for natural_id in natural_id_group:
                indices.extend(natural_id_to_indices[natural_id])
            self._partition_id_to_indices[partition_id] = np.array(indices)

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a single partition corresponding to a single `partition_id`.

        The choice of the partition is based on unique integers assigned to each
        natural id present in the dataset in the `partition_by` column.


        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """
        if len(self._partition_id_to_natural_ids) == 0:
            self._create_int_partition_id_to_natural_id()
            self._create_natural_id_to_int_partition_id()

        if len(self._partition_id_to_indices) == 0:
            self._create_partition_id_to_indices()

        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        if len(self._partition_id_to_natural_ids) == 0:
            self._create_int_partition_id_to_natural_id()
            self._create_natural_id_to_int_partition_id()
        return len(self._partition_id_to_natural_ids)

    @property
    def partition_id_to_natural_ids(self) -> dict[int, list[Any]]:
        """Partition id to the corresponding group of natural ids present.

        Natural ids are the unique values in `partition_by` column in dataset.
        """
        return self._partition_id_to_natural_ids

    @property
    def natural_id_to_partition_id(self) -> dict[Any, int]:
        """Natural id to the corresponding partition id."""
        return self._natural_id_to_partition_id
