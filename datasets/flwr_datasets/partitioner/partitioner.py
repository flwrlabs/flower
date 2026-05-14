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
"""Partitioner class that works with Hugging Face Datasets."""

from abc import ABC, abstractmethod

from datasets import Dataset


class Partitioner(ABC):
    """The base partitioner class that enables obtaining federated partitions.

    The initialization is intended to take all necessary arguments such that the call to
    the `load_partition` method can be used in the same way for all partitioners.
    """

    def __init__(self) -> None:
        self._dataset: Dataset | None = None

    @property
    def dataset(self) -> Dataset:
        """Dataset property."""
        if self._dataset is None:
            raise AttributeError(
                "The dataset field should be set before using it (directly, via "
                "`load_partition` or some other method). "
            )
        return self._dataset

    @dataset.setter
    def dataset(self, value: Dataset) -> None:
        """Set the dataset property."""
        if self._dataset is not None:
            raise ValueError(
                "The dataset should be assigned only once to the partitioner."
                "This operation might also wipe out the saved references to the "
                "created partitions (in case the partitioning scheme needs to create "
                "the full partitioning also in order to return a single partition)."
            )
        if not isinstance(value, Dataset):
            raise TypeError(
                f"The dataset object you want to assign to the partitioner should be "
                f"of type `datasets.Dataset` but given {type(value)}."
            )
        self._dataset = value

    @abstractmethod
    def load_partition(self, partition_id: int) -> Dataset:
        """Load a single partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """

    def is_dataset_assigned(self) -> bool:
        """Check if a dataset has been assigned to the partitioner.

        This method returns True if a dataset is already set for the partitioner,
        otherwise, it returns False.

        Returns
        -------
        dataset_assigned : bool
            True if a dataset is assigned, otherwise False.
        """
        return self._dataset is not None

    @property
    @abstractmethod
    def num_partitions(self) -> int:
        """Total number of partitions."""
