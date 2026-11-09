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
"""IID partitioner class that works with Hugging Face Datasets."""


import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class IidPartitioner(Partitioner):
    """Partitioner creates IID partitions from a dataset.

    By default, partitions are contiguous shards of the dataset. Set ``shuffle=True``
    to shuffle the dataset once before sharding. This is useful for local datasets
    sorted by class or another target column.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    shuffle : bool
        Whether to shuffle the dataset before sharding. The default is ``False``.
    seed : Optional[int]
        Seed used for dataset shuffling when ``shuffle`` is set to ``True``.

    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import IidPartitioner
    >>>
    >>> partitioner = IidPartitioner(num_partitions=10)
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    """

    def __init__(
        self, num_partitions: int, shuffle: bool = False, seed: int | None = 42
    ) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
        self._num_partitions = num_partitions
        self._shuffle = shuffle
        self._seed = seed
        self._shuffled_dataset: datasets.Dataset | None = None

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a single IID partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """
        dataset = self._dataset_to_partition()
        return dataset.shard(
            num_shards=self._num_partitions, index=partition_id, contiguous=True
        )

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        return self._num_partitions

    def _dataset_to_partition(self) -> datasets.Dataset:
        """Return the dataset used for sharding."""
        if not self._shuffle:
            return self.dataset
        if self._shuffled_dataset is None:
            self._shuffled_dataset = self.dataset.shuffle(seed=self._seed)
        return self._shuffled_dataset
