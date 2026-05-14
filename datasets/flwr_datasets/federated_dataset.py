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
"""FederatedDataset."""

from typing import Any

import datasets
from datasets import Dataset, DatasetDict
from flwr_datasets.common import EventType, event
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.preprocessor import Preprocessor
from flwr_datasets.utils import _instantiate_merger_if_needed, _instantiate_partitioners


# noqa: E501
# pylint: disable=line-too-long
class FederatedDataset:
    """Representation of a dataset for federated learning/evaluation/analytics.

    Download, partition data among clients (edge devices), or load full dataset.

    Partitions are created per-split-basis using Partitioners from
    `flwr_datasets.partitioner` specified in `partitioners` (see `partitioners`
    parameter for more information).

    Parameters
    ----------
    dataset : str
        The name of the dataset in the Hugging Face Hub.
    subset : str
        Secondary information regarding the dataset, most often subset or version
        (that is passed to the name in datasets.load_dataset).
    preprocessor : Optional[Union[Preprocessor, Dict[str, Tuple[str, ...]]]]
        `Callable` that transforms `DatasetDict` by resplitting, removing
        features, creating new features, performing any other preprocessing operation,
        or configuration dict for `Merger`. Applied after shuffling. If None,
        no operation is applied.
    partitioners : Dict[str, Union[Partitioner, int]]
        A dictionary mapping the Dataset split (a `str`) to a `Partitioner` or an `int`
        (representing the number of IID partitions that this split should be
        partitioned into, i.e., using the default partitioner
        `IidPartitioner <https://flower.ai/docs/datasets/ref-api/flwr_
        datasets.partitioner.IidPartitioner.html>`_). One or multiple `Partitioner`
        objects can be specified in that manner, but at most, one per split.
    shuffle : bool
        Whether to randomize the order of samples. Applied prior to preprocessing
        operations, speratelly to each of the present splits in the dataset. It uses
        the `seed` argument. Defaults to True.
    seed : Optional[int]
        Seed used for dataset shuffling. It has no effect if `shuffle` is False. The
        seed cannot be set in the later stages. If `None`, then fresh, unpredictable
        entropy will be pulled from the OS. Defaults to 42.
    load_dataset_kwargs : Any
        Additional keyword arguments passed to `datasets.load_dataset` function.
        Currently used parameters used are dataset => path (in load_dataset),
        subset => name (in load_dataset). You can pass e.g., `num_proc=4`,
        `trust_remote_code=True`. Do not pass any parameters that modify the
        return type such as another type than DatasetDict is returned.

    Examples
    --------
    Use MNIST dataset for Federated Learning with 100 clients (edge devices):

    >>> from flwr_datasets import FederatedDataset
    >>>
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
    >>> # Load partition for a client with ID 10.
    >>> partition = fds.load_partition(10)
    >>> # Use test split for centralized evaluation.
    >>> centralized = fds.load_split("test")

    Use CIFAR10 dataset for Federated Laerning with 100 clients:

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>>
    >>> partitioner = DirichletPartitioner(num_partitions=10, partition_by="label",
    >>>                                    alpha=0.5, min_partition_size=10)
    >>> fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(partition_id=0)

    Visualize the partitioned datasets:

    >>> from flwr_datasets.visualization import plot_label_distributions
    >>>
    >>> _ = plot_label_distributions(
    >>>     partitioner=fds.partitioners["train"],
    >>>     label_name="label",
    >>>     legend=True,
    >>> )
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(
        self,
        *,
        dataset: str,
        subset: str | None = None,
        preprocessor: Preprocessor | dict[str, tuple[str, ...]] | None = None,
        partitioners: dict[str, Partitioner | int],
        shuffle: bool = True,
        seed: int | None = 42,
        **load_dataset_kwargs: Any,
    ) -> None:
        self._dataset_name: str = dataset
        self._subset: str | None = subset
        self._preprocessor: Preprocessor | None = _instantiate_merger_if_needed(
            preprocessor
        )
        self._partitioners: dict[str, Partitioner] = _instantiate_partitioners(
            partitioners
        )
        self._check_partitioners_correctness()
        self._shuffle = shuffle
        self._seed = seed
        #  _dataset is prepared lazily on the first call to `load_partition`
        #  or `load_split`. See _prepare_datasets for more details
        self._dataset: DatasetDict | None = None
        # Indicate if the dataset is prepared for `load_partition` or `load_split`
        self._dataset_prepared: bool = False
        self._event = {
            "load_partition": dict.fromkeys(self._partitioners, False),
        }
        self._load_dataset_kwargs = load_dataset_kwargs

    def load_partition(
        self,
        partition_id: int,
        split: str | None = None,
    ) -> Dataset:
        """Load the partition specified by the idx in the selected split.

        The dataset is downloaded only when the first call to `load_partition` or
        `load_split` is made.

        Parameters
        ----------
        partition_id : int
            Partition index for the selected split, idx in {0, ..., num_partitions - 1}.
        split : Optional[str]
            Name of the (partitioned) split (e.g. "train", "test"). You can skip this
            parameter if there is only one partitioner for the dataset. The name will be
            inferred automatically. For example, if `partitioners={"train": 10}`, you do
            not need to provide this argument, but if `partitioners={"train": 10,
            "test": 100}`, you need to set it to differentiate which partitioner should
            be used.
            The split names you can choose from vary from dataset to dataset. You need
            to check the dataset on the `Hugging Face Hub`<https://huggingface.co/
            datasets>_ to see which splits are available. You can resplit the dataset
            by using the `preprocessor` parameter (to rename, merge, divide, etc. the
            available splits).

        Returns
        -------
        partition : Dataset
            Single partition from the dataset split.
        """
        if not self._dataset_prepared:
            self._prepare_dataset()
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        if split is None:
            self._check_if_no_split_keyword_possible()
            split = list(self._partitioners.keys())[0]
        self._check_if_split_present(split)
        self._check_if_split_possible_to_federate(split)
        partitioner: Partitioner = self._partitioners[split]
        self._assign_dataset_to_partitioner(split)
        partition = partitioner.load_partition(partition_id)
        if not self._event["load_partition"][split]:
            event(
                EventType.LOAD_PARTITION_CALLED,
                {
                    "federated_dataset_id": id(self),
                    "dataset_name": self._dataset_name,
                    "split": split,
                    "partitioner": partitioner.__class__.__name__,
                    "num_partitions": partitioner.num_partitions,
                },
            )
            self._event["load_partition"][split] = True
        return partition

    def load_split(self, split: str) -> Dataset:
        """Load the full split of the dataset.

        The dataset is downloaded only when the first call to `load_partition` or
        `load_split` is made.

        Parameters
        ----------
        split : str
            Split name of the downloaded dataset (e.g. "train", "test").
            The split names you can choose from vary from dataset to dataset. You need
            to check the dataset on the `Hugging Face Hub`<https://huggingface.co/
            datasets>_ to see which splits are available. You can resplit the dataset
            by using the `preprocessor` parameter (to rename, merge, divide, etc. the
            available splits).

        Returns
        -------
        dataset_split : Dataset
            Part of the dataset identified by its split name.
        """
        if not self._dataset_prepared:
            self._prepare_dataset()
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        self._check_if_split_present(split)
        dataset_split = self._dataset[split]

        if not self._event["load_split"][split]:
            event(
                EventType.LOAD_SPLIT_CALLED,
                {
                    "federated_dataset_id": id(self),
                    "dataset_name": self._dataset_name,
                    "split": split,
                },
            )
            self._event["load_split"][split] = True

        return dataset_split

    @property
    def partitioners(self) -> dict[str, Partitioner]:
        """Dictionary mapping each split to its associated partitioner.

        The returned partitioners have the splits of the dataset assigned to them.
        """
        # This function triggers the dataset download (lazy download) and checks
        # the partitioner specification correctness (which can also happen lazily only
        # after the dataset download).
        if not self._dataset_prepared:
            self._prepare_dataset()
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        partitioners_keys = list(self._partitioners.keys())
        for split in partitioners_keys:
            self._check_if_split_present(split)
            self._assign_dataset_to_partitioner(split)
        return self._partitioners

    def _check_if_split_present(self, split: str) -> None:
        """Check if the split (for partitioning or full return) is in the dataset."""
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        available_splits = list(self._dataset.keys())
        if split not in available_splits:
            raise ValueError(
                f"The given split: '{split}' is not present in the dataset's splits: "
                f"'{available_splits}'."
            )

    def _check_if_split_possible_to_federate(self, split: str) -> None:
        """Check if the split has corresponding partitioner."""
        partitioners_keys = list(self._partitioners.keys())
        if split not in partitioners_keys:
            raise ValueError(
                f"The given split: '{split}' does not have a partitioner to perform "
                f"partitioning. Partitioners were specified for the following splits:"
                f"'{partitioners_keys}'."
            )

    def _assign_dataset_to_partitioner(self, split: str) -> None:
        """Assign the corresponding split of the dataset to the partitioner.

        Assign only if the dataset is not assigned yet.
        """
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        if not self._partitioners[split].is_dataset_assigned():
            self._partitioners[split].dataset = self._dataset[split]

    def _prepare_dataset(self) -> None:
        """Prepare the dataset (prior to partitioning) by download, shuffle, replit.

        Run only ONCE when triggered by load_* function. (In future more control whether
        this should happen lazily or not can be added). The operations done here should
        not happen more than once.

        It is controlled by a single flag, `_dataset_prepared` that is set True at the
        end of the function.

        Notes
        -----
        The shuffling should happen before the resplitting. Here is the explanation.
        If the dataset has a non-random order of samples e.g. each split has first
        only label 0, then only label 1. Then in case of resplitting e.g.
        someone creates: "train" train[:int(0.75 * len(train))], test: concat(
        train[int(0.75 * len(train)):], test). The new test took the 0.25 of e.g.
        the train that is only label 0 (assuming the equal count of labels).
        Therefore, for such edge cases (for which we have split) the split should
        happen before the resplitting.
        """
        self._dataset = datasets.load_dataset(
            path=self._dataset_name, name=self._subset, **self._load_dataset_kwargs
        )
        if not isinstance(self._dataset, datasets.DatasetDict):
            raise ValueError(
                "Probably one of the specified parameter in `load_dataset_kwargs` "
                "change the return type of the datasets.load_dataset function. "
                "Make sure to use parameter such that the return type is DatasetDict. "
                f"The return type is currently: {type(self._dataset)}."
            )
        if self._shuffle:
            # Note it shuffles all the splits. The self._dataset is DatasetDict
            # so e.g. {"train": train_data, "test": test_data}. All splits get shuffled.
            self._dataset = self._dataset.shuffle(seed=self._seed)
        if self._preprocessor:
            self._dataset = self._preprocessor(self._dataset)
        available_splits = list(self._dataset.keys())
        self._event["load_split"] = dict.fromkeys(available_splits, False)
        self._dataset_prepared = True

    def _check_if_no_split_keyword_possible(self) -> None:
        if len(self._partitioners) != 1:
            raise ValueError(
                "Please set the `split` argument. You can only omit the split keyword "
                "if there is exactly one partitioner specified."
            )

    def _check_partitioners_correctness(self) -> None:
        """Check if the partitioners are correctly specified.

        Check if each partitioner is a different Python object. Using the same
        partitioner for different splits is not allowed.
        """
        partitioners_keys = list(self._partitioners.keys())
        for i, first_split in enumerate(partitioners_keys):
            for j in range(i + 1, len(partitioners_keys)):
                second_split = partitioners_keys[j]
                if self._partitioners[first_split] is self._partitioners[second_split]:
                    raise ValueError(
                        f"The same partitioner object is used for multiple splits: "
                        f"('{first_split}', '{second_split}'). "
                        "Each partitioner should be a separate object."
                    )
