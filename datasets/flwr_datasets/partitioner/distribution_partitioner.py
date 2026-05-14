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
"""Distribution partitioner class that works with Hugging Face Datasets."""

from collections import Counter

import numpy as np

import datasets
from flwr_datasets.common.typing import NDArray, NDArrayFloat, NDArrayInt
from flwr_datasets.partitioner.partitioner import Partitioner


class DistributionPartitioner(Partitioner):  # pylint: disable=R0902
    """Partitioner based on a distribution.

    Inspired from implementations of Li et al. Federated Optimization in
    Heterogeneous Networks (2020) https://arxiv.org/abs/1812.06127.

    Given a 2-dimensional user-specified distribution, the algorithm splits the dataset
    for each unique label per partition where each label is assigned to the partitions
    in a deterministic pathological manner. The 1st dimension is the number of unique
    labels and the 2nd-dimension is the number of buckets into which the samples
    associated with each label will be divided. That is, given a distribution array of
    shape,::

                            `num_unique_labels_per_partition` x `num_partitions`
        ( `num_unique_labels`, ---------------------------------------------------- ),
                                            `num_unique_labels`
        the label_id at the i'th row is assigned to the partition_id based on the
        following approach.

        First, for an i'th row, generate a list of `id`s according to the formula:
            id = alpha + beta
        where,
            alpha = (i - num_unique_labels_per_partition + 1) +
                    + (j % num_unique_labels_per_partition),
            alpha = alpha + (alpha >= 0 ? 0 : num_unique_labels),
            beta = num_unique_labels * (j // num_unique_labels_per_partition)
    and j in {0, 1, 2, ..., `num_columns`}. Then, sort the list of `id` s in ascending
    order. The j'th index in this sorted list corresponds to the partition_id that the
    i'th unique label (and the underlying distribution array value) will be assigned to.
    So, for a dataset with 10 unique labels and a configuration with 20 partitions and
    2 unique labels per partition, the 0'th row of the distribution array (corresponding
    to class 0) will be assigned to partitions [0, 9, 10, 19], 1st row (class 1) to
    [0, 1, 10, 11], 2nd row (class 2) to [1, 2, 11, 12], 3rd row (class 3) to
    [2, 3, 12, 13], etc ... . Alternatively, the distribution can be interpreted as
    partition 0 having classes 0 and 1, partition 1 having classes 1 and 2, partition 2
    having classes 2 and 3, etc ... The list representing the unique labels is sorted
    in ascending order.

    Parameters
    ----------
    distribution_array : Union[NDArrayInt, NDArrayFloat]
        A 2-dimensional numpy array of the probability distribution of samples
        for all labels in all partitions. The array shape should be
        (`num_unique_labels`,
        `num_unique_labels_per_partition*num_partitions/num_unique_labels`),
        such that the first row of the array corresponds to the sample distribution
        of the first unique label (in ascending order). The values may be scaled per
        label such that the sum of the label distributions across all partitions are
        equal to the original unpartitioned label distribution
        - see the `rescale` argument.
    num_partitions : int
        The total number of partitions that the data will be divided into. The number of
        partitions must be an integer multiple of the number of unique labels in the
        dataset.
    num_unique_labels_per_partition : int
        Number of unique labels assigned to a single partition.
    partition_by : str
        Column name of the labels (targets) based on which sampling works.
    preassigned_num_samples_per_label : int
        The number of samples that each unique label in each partition will first
        be assigned before the `distribution_array` values are assigned. This
        value has no effect if `rescale` is set to False.
    rescale : bool, default=True
        Whether to partition samples according to the values in
        `distribution_array` or rescale based on the original unpartitioned class label
        distribution. `float` values are rounded to the nearest `int`. All samples for
        any label_id are exhausted during the partitioning by randomly assigning any
        unassigned samples from round-off errors to one of the label_id's partition_ids.
    shuffle : bool, default=True
        Whether to randomize the order of samples. Shuffling applied after the
        samples assignment to nodes.
    seed : int, default=42
        Seed used for initializing the random number generator (RNG),
        which affects dataset shuffling (if `shuffle` is True)
        and random assignment of any unassigned samples that resulted from
        rounding errors during label rescaling (if `rescale` is True).

    Examples
    --------
    In order to reproduce the power-law distribution of the paper, follow this setup:

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DistributionPartitioner
    >>> from pprint import pprint
    >>> import numpy as np
    >>>
    >>> num_partitions = 1_000
    >>> num_unique_labels_per_partition = 2
    >>> num_unique_labels = 10
    >>> preassigned_num_samples_per_label = 5
    >>>
    >>> # Generate a vector from a log-normal probability distribution
    >>> rng = np.random.default_rng(2024)
    >>> mu, sigma = 0., 2.
    >>> distribution_array = rng.lognormal(
    >>>     mu,
    >>>     sigma,
    >>>     (num_partitions*num_unique_labels_per_partition),
    >>> )
    >>> distribution_array = distribution_array.reshape((num_unique_labels, -1))
    >>>
    >>> partitioner = DistributionPartitioner(
    >>>     distribution_array=distribution_array,
    >>>     num_partitions=num_partitions,
    >>>     num_unique_labels_per_partition=num_unique_labels_per_partition,
    >>>     partition_by="label",  # MNIST dataset has a target column `label`
    >>>     preassigned_num_samples_per_label=preassigned_num_samples_per_label,
    >>> )
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    >>> print(partition[0])  # Print the first example
    {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x169DD54D0>,
    'label': 0}
    >>> distributions = {
    >>>     partition_id: fds.load_partition(partition_id=partition_id)
    >>>     .to_pandas()["label"]
    >>>     .value_counts()
    >>>     .to_dict()
    >>>     for partition_id in range(10)
    >>> }
    >>> pprint(distributions)
    {0: {0: 40, 1: 5},
     1: {2: 36, 1: 5},
     2: {3: 52, 2: 7},
     3: {3: 14, 4: 6},
     4: {4: 47, 5: 28},
     5: {6: 30, 5: 5},
     6: {6: 19, 7: 11},
     7: {8: 22, 7: 11},
     8: {9: 11, 8: 5},
     9: {0: 124, 9: 13}}
    """

    def __init__(  # pylint: disable=R0913, R0917
        self,
        distribution_array: NDArrayInt | NDArrayFloat,
        num_partitions: int,
        num_unique_labels_per_partition: int,
        partition_by: str,
        preassigned_num_samples_per_label: int,
        rescale: bool = True,
        shuffle: bool = True,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        # Attributes based on the constructor
        self._distribution_array = distribution_array
        self._num_partitions = num_partitions
        self._num_unique_labels_per_partition = num_unique_labels_per_partition
        self._partition_by = partition_by
        self._preassigned_num_samples_per_label = preassigned_num_samples_per_label
        self._rescale = rescale
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)  # NumPy random generator

        # Utility attributes
        # The attributes below are determined during the first call to load_partition
        self._num_unique_labels: int = 0
        self._num_columns: int = 0
        self._partition_id_to_indices_determined = False
        self._partition_id_to_indices: dict[int, list[int]] = {}

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single partition of a dataset
        """
        # The partitioning is done lazily - only when the first partition is
        # requested. Only the first call creates the indices assignments for all the
        # partition indices.
        self._check_distribution_array_shape_if_needed()
        self._check_num_unique_labels_per_partition_if_needed()
        self._check_distribution_array_sum_if_needed()
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        return self._num_partitions

    def _determine_partition_id_to_indices_if_needed(  # pylint: disable=R0914
        self,
    ) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return

        # Compute the label distribution from the dataset
        unique_labels = sorted(self.dataset.unique(self._partition_by))
        labels = np.asarray(self.dataset[self._partition_by])
        unique_label_to_indices = {}
        unique_label_distribution = {}

        for unique_label in unique_labels:
            unique_label_to_indices[unique_label] = np.where(labels == unique_label)[0]
            unique_label_distribution[unique_label] = len(
                unique_label_to_indices[unique_label]
            )

        if self._rescale:
            # Compute the normalized distribution for each class label
            self._distribution_array = self._distribution_array / np.sum(
                self._distribution_array, axis=-1, keepdims=True
            )

            # Compute the total preassigned number of samples per label for all labels
            # and partitions. This sum will be subtracted from the label distribution
            # of the original dataset, and added back later. It ensures that
            # (1) each partition will have at least
            #     `self._preassigned_num_samples_per_label`, and
            # (2) there is sufficient indices to sample from the dataset.
            total_preassigned_samples = int(
                self._preassigned_num_samples_per_label * self._num_columns
            )

            label_distribution = np.fromiter(
                unique_label_distribution.values(),
                dtype=float,
            )

            self._check_total_preassigned_samples_within_limit(
                label_distribution, total_preassigned_samples
            )

            # Subtract the preassigned total amount from the label distribution,
            # we'll add these back later.
            label_distribution -= total_preassigned_samples

            # Rescale normalized distribution with the actual label distribution.
            # Each row represents the number of samples to be taken for that class label
            # and the sum of each row equals the total of each class label.
            label_sampling_matrix = np.floor(
                self._distribution_array * label_distribution[:, np.newaxis]
            ).astype(int)

            # Add back the preassigned total amount
            label_sampling_matrix += self._preassigned_num_samples_per_label
        else:
            label_sampling_matrix = self._distribution_array.astype(int)

        # Create the label sampling dictionary
        label_samples = dict(
            zip(unique_label_distribution.keys(), label_sampling_matrix, strict=False)
        )

        # Create indices split from dataset
        split_indices_per_label = {}
        for unique_label in unique_labels:
            # Compute cumulative sum of samples to identify splitting points
            cumsum_division_numbers = np.cumsum(label_samples[unique_label])
            split_indices = np.split(
                unique_label_to_indices[unique_label], cumsum_division_numbers
            )
            if self._rescale:
                # Randomly append unassigned samples (which are in the last split that
                # exceeds `self._num_columns`) to one of the `self._num_columns`
                # partitions. Unassigned samples originate from float-to-int rounding
                # errors of the normalizing algorithm.
                if len(split_indices) > self._num_columns:
                    last_split = split_indices.pop()
                    random_index = self._rng.integers(0, self._num_columns)
                    split_indices[random_index] = np.append(
                        split_indices[random_index], last_split
                    )
                assert len(split_indices) == self._num_columns
            split_indices_per_label[unique_label] = split_indices

        # Initialize sampling tracker. Keys are the unique class labels.
        # Values are the smallest indices of each array in `label_samples`
        # which will be sampled next. Once a sample is taken from a label/key,
        # increment the value (index) by 1.
        index_tracker = dict.fromkeys(unique_labels, 0)

        # Prepare data structure to store indices assigned to partition ids
        self._partition_id_to_indices = {
            partition_id: [] for partition_id in range(self._num_partitions)
        }

        for partition_id in range(self._num_partitions):
            # Get the `num_unique_labels_per_partition` labels for each partition. Use
            # `numpy.roll` to get indices of adjacent sorted labels for pathological
            # label distributions.
            labels_per_client = np.roll(unique_labels, -partition_id)[
                : self._num_unique_labels_per_partition
            ]
            for label in labels_per_client:
                index_to_sample = index_tracker[label]
                self._partition_id_to_indices[partition_id].extend(
                    split_indices_per_label[label][index_to_sample]
                )
                index_tracker[label] += 1

        # Shuffle the indices to avoid datasets with targets in sequences like
        # [00000, 11111, ...]) if the shuffle is True
        if self._shuffle:
            for indices in self._partition_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)
        self._partition_id_to_indices_determined = True

    def _check_distribution_array_shape_if_needed(self) -> None:
        """Test distribution array shape correctness."""
        if not self._partition_id_to_indices_determined:
            if not isinstance(self._distribution_array, np.ndarray):
                raise TypeError("Input must be a NumPy array.")

            if self._distribution_array.ndim != 2:
                raise ValueError("The distribution array is not 2-dimensional.")

            self._num_unique_labels = len(self.dataset.unique(self._partition_by))
            self._num_columns = int(
                self._num_unique_labels_per_partition
                * self._num_partitions
                / self._num_unique_labels
            )

            if self._distribution_array.shape[0] != self._num_unique_labels:
                raise ValueError(
                    "The expected number of rows in `distribution_array` must equal to "
                    "the number of unique labels in the dataset, which is "
                    f"{self._num_unique_labels}, but the number of rows in "
                    f"`distribution_array` is {self._distribution_array.shape[0]}."
                )

            if self._distribution_array.shape[1] != self._num_columns:
                raise ValueError(
                    "The expected number of columns in `distribution_array` is "
                    f"{self._num_columns} (refer to the documentation for the "
                    "expression), but the number of columns in `distribution_array` "
                    f"is {self._distribution_array.shape[1]}."
                )

    def _check_num_unique_labels_per_partition_if_needed(self) -> None:
        """Test number of unique labels do not exceed self.num_unique_labels."""
        if self._num_unique_labels_per_partition > self._num_unique_labels:
            raise ValueError(
                "The specified `num_unique_labels_per_partition`"
                f"={self._num_unique_labels_per_partition} is greater than the number "
                f"of unique labels in the given dataset={self._num_unique_labels} "
                f"as specified by the label column `{self._partition_by}`."
                "Reduce the `num_unique_labels_per_partition` or make use of a "
                "different dataset to apply this partitioning."
            )

    def _check_distribution_array_sum_if_needed(self) -> None:
        """Test correctness of distribution array sum."""
        if not self._partition_id_to_indices_determined and not self._rescale:
            labels = self.dataset[self._partition_by]
            unique_labels_counter = sorted(Counter(labels).items())
            unique_labels_counter_vals = [v for _, v in unique_labels_counter]

            if any(self._distribution_array.sum(1) > unique_labels_counter_vals):
                raise ValueError(
                    "The sum of at least one unique label distribution array "
                    "exceeds that of the unique labels counter in the given dataset= "
                    f"{dict(unique_labels_counter)}."
                )

    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test num_partitions when the dataset is given."""
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    f"The number of partitions ({self._num_partitions}) needs to be "
                    "smaller than the number of samples in the dataset "
                    f"({self.dataset.num_rows})."
                )
            if self._num_partitions % self._num_unique_labels != 0:
                raise ValueError(
                    f"The number of partitions ({self._num_partitions}) is not "
                    f"divisible by the number of unique labels "
                    f"{({self._num_unique_labels})}."
                )
            if not self._num_partitions > 0:
                raise ValueError(
                    "The number of partitions needs to be greater than zero."
                )

    def _check_total_preassigned_samples_within_limit(
        self, label_distribution: NDArray, total_preassigned_samples: int
    ) -> None:
        """Test total preassigned samples do not exceed minimum allowable."""
        if any(label_distribution - total_preassigned_samples < self._num_columns):
            raise ValueError(
                "There is insufficient samples to partition by applying the specified "
                "`preassigned_num_samples_per_label`"
                f"={self._preassigned_num_samples_per_label}. Reduce the "
                "`preassigned_num_samples_per_label` or use a different dataset with "
                "more samples to apply this partition."
            )
