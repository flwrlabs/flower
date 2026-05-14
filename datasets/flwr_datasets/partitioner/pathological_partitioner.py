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
"""Pathological partitioner class that works with Hugging Face Datasets."""

import warnings
from typing import Any, Literal

import numpy as np

import datasets
from flwr_datasets.common.typing import NDArray
from flwr_datasets.partitioner.partitioner import Partitioner


# pylint: disable=too-many-arguments, too-many-instance-attributes
class PathologicalPartitioner(Partitioner):
    """Partition dataset such that each partition has a chosen number of classes.

    Implementation based on Federated Learning on Non-IID Data Silos: An Experimental
    Study https://arxiv.org/pdf/2102.02079.

    The algorithm firstly determines which classes will be assigned to which partitions.
    For each partition `num_classes_per_partition` are sampled in a way chosen in
    `class_assignment_mode`. Given the information about the required classes for each
    partition, it is determined into how many parts the samples corresponding to this
    label should be divided. Such division is performed for each class.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    partition_by : str
        Column name of the labels (targets) based on which partitioning works.
    num_classes_per_partition: int
        The (exact) number of unique classes that each partition will have.
    class_assignment_mode: Literal["random", "deterministic", "first-deterministic"]
        The way how the classes are assigned to the partitions. The default is "random".
        The possible values are:

        - "random": Randomly assign classes to the partitions. For each partition choose
          the `num_classes_per_partition` classes without replacement.
        - "first-deterministic": Assign the first class for each partition in a
          deterministic way (class id is the partition_id % num_unique_classes).
          The rest of the classes are assigned randomly. In case the number of
          partitions is smaller than the number of unique classes, not all classes will
          be used in the first iteration, otherwise all the classes will be used (such
          it will be present in at least one partition).
        - "deterministic": Assign all the classes to the partitions in a deterministic
          way. Classes are assigned based on the formula: partion_id has classes
          identified by the index: (partition_id + i) % num_unique_classes
          where i in {0, ..., num_classes_per_partition}. So, partition 0 will have
          classes 0, 1, 2, ..., `num_classes_per_partition`-1, partition 1 will have
          classes 1, 2, 3, ...,`num_classes_per_partition`, ....

        The list representing the unique labels is sorted in ascending order. In case
        of numbers starting from zero the class id corresponds to the number itself.
        `class_assignment_mode="first-deterministic"` was used in the original paper,
        here we provide the option to use the other modes as well.
    shuffle: bool
        Whether to randomize the order of samples. Shuffling applied after the
        samples assignment to partitions.
    seed: int
        Seed used for initializing the random number generator (RNG),
        which affects the ranodm assignment of labels/classes to partitions
        and dataset shuffling (if `shuffle` is True).

    Examples
    --------
    In order to mimic the original behavior of the paper follow the setup below
    (the `class_assignment_mode="first-deterministic"`):

    >>> from flwr_datasets.partitioner import PathologicalPartitioner
    >>> from flwr_datasets import FederatedDataset
    >>>
    >>> partitioner = PathologicalPartitioner(
    >>>     num_partitions=10,
    >>>     partition_by="label",
    >>>     num_classes_per_partition=2,
    >>>     class_assignment_mode="first-deterministic"
    >>> )
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    """

    def __init__(  # pylint: disable=R0917
        self,
        num_partitions: int,
        partition_by: str,
        num_classes_per_partition: int,
        class_assignment_mode: Literal[
            "random", "deterministic", "first-deterministic"
        ] = "random",
        shuffle: bool = True,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._partition_by = partition_by
        self._num_classes_per_partition = num_classes_per_partition
        self._class_assignment_mode = class_assignment_mode
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

        # Utility attributes
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_unique_labels: dict[int, list[Any]] = {
            pid: [] for pid in range(self._num_partitions)
        }
        self._unique_labels: list[Any] = []
        # Count in how many partitions the label is used
        self._unique_label_to_times_used_counter: dict[Any, int] = {}
        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            The index that corresponds to the requested partition.

        Returns
        -------
        dataset_partition : Dataset
            Single partition of a dataset.
        """
        # The partitioning is done lazily - only when the first partition is
        # requested. Only the first call creates the indices assignments for all the
        # partition indices.
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    def _determine_partition_id_to_indices_if_needed(self) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return
        self._determine_partition_id_to_unique_labels()
        assert self._unique_labels is not None
        self._count_partitions_having_each_unique_label()

        labels = np.asarray(self.dataset[self._partition_by])
        self._check_correctness_of_unique_label_to_times_used_counter(labels)
        for partition_id in range(self._num_partitions):
            self._partition_id_to_indices[partition_id] = []

        unused_labels = []
        for unique_label in self._unique_labels:
            if self._unique_label_to_times_used_counter[unique_label] == 0:
                unused_labels.append(unique_label)
                continue
            # Get the indices in the original dataset where the y == unique_label
            unique_label_to_indices = np.where(labels == unique_label)[0]

            split_unique_labels_to_indices = np.array_split(
                unique_label_to_indices,
                self._unique_label_to_times_used_counter[unique_label],
            )

            split_index = 0
            for partition_id in range(self._num_partitions):
                if unique_label in self._partition_id_to_unique_labels[partition_id]:
                    self._partition_id_to_indices[partition_id].extend(
                        split_unique_labels_to_indices[split_index]
                    )
                    split_index += 1

        if len(unused_labels) >= 1:
            warnings.warn(
                f"Classes: {unused_labels} will NOT be used due to the chosen "
                f"configuration. If it is undesired behavior consider setting"
                f" 'first_class_deterministic_assignment=True' which in case when"
                f" the number of classes is smaller than the number of partitions will "
                f"utilize all the classes for the created partitions.",
                stacklevel=1,
            )
        if self._shuffle:
            for indices in self._partition_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)

        self._partition_id_to_indices_determined = True

    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test num_partitions when the dataset is given (in load_partition)."""
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of "
                    "samples in the dataset."
                )

    def _determine_partition_id_to_unique_labels(self) -> None:
        """Determine the assignment of unique labels to the partitions."""
        self._unique_labels = sorted(self.dataset.unique(self._partition_by))
        num_unique_classes = len(self._unique_labels)

        if self._num_classes_per_partition > num_unique_classes:
            raise ValueError(
                f"The specified `num_classes_per_partition`"
                f"={self._num_classes_per_partition} is greater than the number "
                f"of unique classes in the given dataset={num_unique_classes}. "
                f"Reduce the `num_classes_per_partition` or make use different dataset "
                f"to apply this partitioning."
            )
        if self._class_assignment_mode == "first-deterministic":
            # if self._first_class_deterministic_assignment:
            for partition_id in range(self._num_partitions):
                label = self._unique_labels[partition_id % num_unique_classes]
                self._partition_id_to_unique_labels[partition_id].append(label)

                while (
                    len(self._partition_id_to_unique_labels[partition_id])
                    < self._num_classes_per_partition
                ):
                    label = self._rng.choice(self._unique_labels, size=1)[0]
                    if label not in self._partition_id_to_unique_labels[partition_id]:
                        self._partition_id_to_unique_labels[partition_id].append(label)
        elif self._class_assignment_mode == "deterministic":
            for partition_id in range(self._num_partitions):
                labels = []
                for i in range(self._num_classes_per_partition):
                    label = self._unique_labels[
                        (partition_id + i) % len(self._unique_labels)
                    ]
                    labels.append(label)
                self._partition_id_to_unique_labels[partition_id] = labels
        elif self._class_assignment_mode == "random":
            for partition_id in range(self._num_partitions):
                labels = self._rng.choice(
                    self._unique_labels,
                    size=self._num_classes_per_partition,
                    replace=False,
                ).tolist()
                self._partition_id_to_unique_labels[partition_id] = labels
        else:
            raise ValueError(
                f"The supported class_assignment_mode are: 'random', 'deterministic', "
                f"'first-deterministic'. You provided: {self._class_assignment_mode}."
            )

    def _count_partitions_having_each_unique_label(self) -> None:
        """Count the number of partitions that have each unique label.

        This computation is based on the assignment of the label to the partition_id in
        the `_determine_partition_id_to_unique_labels` method.
        Given:
        * partition 0 has only labels: 0,1 (not necessarily just two samples it can have
          many samples but either from 0 or 1)
        *  partition 1 has only labels: 1, 2 (same count note as above)
        * and there are only two partitions then the following will be computed:
        {
          0: 1,
          1: 2,
          2: 1
        }
        """
        for unique_label in self._unique_labels:
            self._unique_label_to_times_used_counter[unique_label] = 0
        for unique_labels in self._partition_id_to_unique_labels.values():
            for unique_label in unique_labels:
                self._unique_label_to_times_used_counter[unique_label] += 1

    def _check_correctness_of_unique_label_to_times_used_counter(
        self, labels: NDArray
    ) -> None:
        """Check if partitioning is possible given the presence requirements.

        The number of times the label can be used must be smaller or equal to the number
        of times that the label is present in the dataset.
        """
        for unique_label in self._unique_labels:
            num_unique = np.sum(labels == unique_label)
            if self._unique_label_to_times_used_counter[unique_label] > num_unique:
                raise ValueError(
                    f"Label: {unique_label} is needed to be assigned to more "
                    f"partitions "
                    f"({self._unique_label_to_times_used_counter[unique_label]})"
                    f" than there are samples (corresponding to this label) in the "
                    f"dataset ({num_unique}). Please decrease the `num_partitions`, "
                    f"`num_classes_per_partition` to avoid this situation, "
                    f"or try `class_assignment_mode='deterministic'` to create a more "
                    f"even distribution of classes along the partitions. "
                    f"Alternatively use a different dataset if you can not adjust"
                    f" the any of these parameters."
                )
