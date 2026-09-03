"""floco: A Flower Baseline."""

from typing import Optional, Union

import numpy as np
from flwr_datasets.partitioner.partitioner import Partitioner

import datasets


# pylint: disable=R0902, R0912, R0914


class FoldPartitioner(Partitioner):
    """Partitioner based on data Folds split.

    Implementation based on https://arxiv.org/pdf/2007.03797

    This procedure partitions clients in equally sized groups and assigns each group
    a set of primary classes. Every client gets q * 100 % of its data from its group’s
    primary classes and (100 − q) % from the remaining classes.
    We apply this method with q = 80 for five groups and refer to this split as 5-Fold.
    For example, in CIFAR-10 5-Fold, 20 % of the clients get assigned 80 % samples
    from classes 1-2 and 20 % from classes 3-10.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    partition_by : str
        Column name of the labels (targets) based on which Dirichlet sampling works.
    num_folds : float
        The number of folds to split the dataset into groups of clients that share
        the primary classes.
    q : float
        The percentage of samples that each client gets from the primary classes.
    min_partition_size : int
        The minimum number of samples that each partitions will have (the sampling
        process is repeated if any partition is too small).
    shuffle: bool
        Whether to randomize the order of samples. Shuffling applied after the
        samples assignment to partitions.
    seed: int
        Seed used for dataset shuffling. It has no effect if `shuffle` is False.
    """

    def __init__(  # pylint: disable=R0913,R0917
        self,
        num_partitions: int,
        partition_by: str,
        num_folds: float,
        q: float,
        min_partition_size: int = 10,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        # Attributes based on the constructor
        self._num_partitions = num_partitions
        self._check_num_partitions_greater_than_zero()
        self._partition_by = partition_by
        self._num_folds = int(num_folds)
        self._q = float(q)
        self._min_partition_size: int = min_partition_size
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)  # NumPy random generator

        # Utility attributes
        # The attributes below are determined during the first call to load_partition
        self._avg_num_of_samples_per_partition: Optional[float] = None
        self._unique_classes: Optional[Union[list[int], list[str]]] = None
        self._partition_id_to_indices: dict[int, np.ndarray] = {}
        self._partition_id_to_indices_determined = False

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
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    def _determine_partition_id_to_indices_if_needed(
        self,
    ) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return

        # Generate information needed for Dirichlet partitioning
        self._unique_classes = self.dataset.unique(self._partition_by)
        assert self._unique_classes is not None
        # This is needed only if self._self_balancing is True (the default option)
        self._avg_num_of_samples_per_partition = (
            self.dataset.num_rows / self._num_partitions
        )

        # Change targets list data type to numpy
        targets = np.array(self.dataset[self._partition_by])

        # Repeat the sampling procedure based on the Dirichlet distribution until the
        # min_partition_size is reached.
        np.random.seed(self._seed)
        s = self._q / 100
        folds = np.array_split(
            ary=np.arange(len(self._unique_classes)),
            indices_or_sections=self._num_folds,
        )
        # -------------------------------------------------------
        # divide the first dataset that all clients share (includes all classes)
        num_imgs_iid = int(self._min_partition_size * s)
        num_imgs_noniid = self._min_partition_size - num_imgs_iid
        partition_id_to_indices = {i: np.array([]) for i in range(self._num_partitions)}
        num_samples = len(targets)
        num_per_label_total = int(num_samples / len(self._unique_classes))
        labels1 = np.array(targets)
        idxs1 = np.arange(len(targets))
        # iid labels
        idxs_labels = np.vstack((idxs1, labels1))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        # label available
        label_list = list(range(len(self._unique_classes)))
        # number of imgs has allocated per label
        label_used = [2000 for i in range(len(self._unique_classes))]
        iid_per_label = int(num_imgs_iid / len(self._unique_classes))
        iid_per_label_last = (
            num_imgs_iid - (len(self._unique_classes) - 1) * iid_per_label
        )
        np.random.seed(self._seed)
        for i in range(self._num_partitions):
            # allocate iid idxs
            label_cnt = 0
            for y in label_list:
                label_cnt = label_cnt + 1
                iid_num = iid_per_label
                start = y * num_per_label_total + label_used[y]
                if label_cnt == len(self._unique_classes):
                    iid_num = iid_per_label_last
                if (label_used[y] + iid_num) > num_per_label_total:
                    start = y * num_per_label_total
                    label_used[y] = 0
                partition_id_to_indices[i] = np.concatenate(
                    (partition_id_to_indices[i], idxs[start : start + iid_num]), axis=0
                )
                label_used[y] = label_used[y] + iid_num
            # allocate noniid idxs
            # rand_label = np.random.choice(label_list, 3, replace=False)
            rand_label = folds[i % len(folds)]
            noniid_labels = len(rand_label)
            noniid_per_num = int(num_imgs_noniid / noniid_labels)
            noniid_per_num_last = num_imgs_noniid - noniid_per_num * (noniid_labels - 1)
            label_cnt = 0
            for y in rand_label:
                label_cnt = label_cnt + 1
                noniid_num = noniid_per_num
                start = y * num_per_label_total + label_used[y]
                if label_cnt == noniid_labels:
                    noniid_num = noniid_per_num_last
                if (label_used[y] + noniid_num) > num_per_label_total:
                    start = y * num_per_label_total
                    label_used[y] = 0
                partition_id_to_indices[i] = np.concatenate(
                    (partition_id_to_indices[i], idxs[start : start + noniid_num]),
                    axis=0,
                )
                label_used[y] = label_used[y] + noniid_num
            partition_id_to_indices[i] = partition_id_to_indices[i].astype(int)

        # Shuffle the indices not to have the datasets with targets in sequences like
        # [00000, 11111, ...]) if the shuffle is True
        if self._shuffle:
            for indices in partition_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)
        self._partition_id_to_indices = partition_id_to_indices
        self._partition_id_to_indices_determined = True

    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test num_partitions when the dataset is given (in load_partition)."""
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of "
                    "samples in the dataset."
                )

    def _check_num_partitions_greater_than_zero(self) -> None:
        """Test num_partition left sides correctness."""
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")
