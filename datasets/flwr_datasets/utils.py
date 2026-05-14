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
"""Utils for FederatedDataset."""

import warnings
from typing import cast

from datasets import Dataset, DatasetDict, concatenate_datasets
from flwr_datasets.partitioner import IidPartitioner, Partitioner
from flwr_datasets.preprocessor import Preprocessor
from flwr_datasets.preprocessor.merger import Merger

tested_datasets = [
    "mnist",
    "ylecun/mnist",
    "cifar10",
    "uoft-cs/cifar10",
    "fashion_mnist",
    "zalando-datasets/fashion_mnist",
    "sasha/dog-food",
    "zh-plus/tiny-imagenet",
    "scikit-learn/adult-census-income",
    "cifar100",
    "uoft-cs/cifar100",
    "svhn",
    "ufldl-stanford/svhn",
    "sentiment140",
    "stanfordnlp/sentiment140",
    "speech_commands",
    "LIUM/tedlium",
    "flwrlabs/femnist",
    "flwrlabs/ucf101",
    "flwrlabs/ambient-acoustic-context",
    "jlh/uci-mushrooms",
    "Mike0307/MNIST-M",
    "flwrlabs/usps",
    "scikit-learn/iris",
    "flwrlabs/pacs",
    "flwrlabs/cinic10",
    "flwrlabs/caltech101",
    "flwrlabs/office-home",
    "flwrlabs/fed-isic2019",
]


def _instantiate_partitioners(
    partitioners: dict[str, Partitioner | int],
) -> dict[str, Partitioner]:
    """Transform the partitioners from the initial format to instantiated objects.

    Parameters
    ----------
    partitioners : Dict[str, Union[Partitioner, int]]
        Dataset split to the Partitioner or a number of IID partitions.

    Returns
    -------
    partitioners : Dict[str, Partitioner]
        Partitioners specified as split to Partitioner object.
    """
    instantiated_partitioners: dict[str, Partitioner] = {}
    if isinstance(partitioners, dict):
        for split, partitioner in partitioners.items():
            if isinstance(partitioner, Partitioner):
                instantiated_partitioners[split] = partitioner
            elif isinstance(partitioner, int):
                instantiated_partitioners[split] = IidPartitioner(
                    num_partitions=partitioner
                )
            else:
                raise ValueError(
                    f"Incorrect type of the 'partitioners' value encountered. "
                    f"Expected Partitioner or int. Given {type(partitioner)}"
                )
    else:
        raise ValueError(
            f"Incorrect type of the 'partitioners' encountered. "
            f"Expected Dict[str, Union[int, Partitioner]]. "
            f"Given {type(partitioners)}."
        )
    return instantiated_partitioners


def _instantiate_merger_if_needed(
    merger: Preprocessor | dict[str, tuple[str, ...]] | None,
) -> Preprocessor | None:
    """Instantiate `Merger` if preprocessor is merge_config."""
    if merger and isinstance(merger, dict):
        merger = Merger(merge_config=merger)
    return cast(Preprocessor | None, merger)


def divide_dataset(
    dataset: Dataset, division: list[float] | tuple[float, ...] | dict[str, float]
) -> list[Dataset] | DatasetDict:
    """Divide the dataset according to the `division`.

    The division support varying number of splits, which you can name. The splits are
    created from the beginning of the dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset to be divided.
    division: Union[List[float], Tuple[float, ...], Dict[str, float]]
        Configuration specifying how the dataset is divided. Each fraction has to be
        >0 and <=1. They have to sum up to at most 1 (smaller sum is possible).

    Returns
    -------
    divided_dataset : Union[List[Dataset], DatasetDict]
        If `division` is `List` or `Tuple` then `List[Dataset]` is returned else if
        `division` is `Dict` then `DatasetDict` is returned.

    Examples
    --------
    Use `divide_dataset` with division specified as a list.

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.utils import divide_dataset
    >>>
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
    >>> partition = fds.load_partition(0)
    >>> division = [0.8, 0.2]
    >>> train, test = divide_dataset(dataset=partition, division=division)

    Use `divide_dataset` with division specified as a dict
    (this accomplishes the same goal as the example with a list above).

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.utils import divide_dataset
    >>>
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
    >>> partition = fds.load_partition(0)
    >>> division = {"train": 0.8, "test": 0.2}
    >>> train_test = divide_dataset(dataset=partition, division=division)
    >>> train, test = train_test["train"], train_test["test"]
    """
    _check_division_config_correctness(division)
    dataset_length = len(dataset)
    ranges = _create_division_indices_ranges(dataset_length, division)
    if isinstance(division, (list | tuple)):
        split_partition: list[Dataset] = []
        for single_range in ranges:
            split_partition.append(dataset.select(single_range))
        return split_partition
    if isinstance(division, dict):
        split_partition_dict: dict[str, Dataset] = {}
        for split_name, single_range in zip(division.keys(), ranges, strict=False):
            split_partition_dict[split_name] = dataset.select(single_range)
        return DatasetDict(split_partition_dict)
    raise TypeError(
        f"The type of the `division` should be dict, "
        f"tuple or list but is {type(division)} instead."
    )


def _create_division_indices_ranges(
    dataset_length: int,
    division: list[float] | tuple[float, ...] | dict[str, float],
) -> list[range]:
    ranges = []
    if isinstance(division, (list | tuple)):
        start_idx = 0
        end_idx = 0
        for fraction in division:
            end_idx += int(dataset_length * fraction)
            ranges.append(range(start_idx, end_idx))
            start_idx = end_idx
    elif isinstance(division, dict):
        ranges = []
        start_idx = 0
        end_idx = 0
        for fraction in division.values():
            end_idx += int(dataset_length * fraction)
            ranges.append(range(start_idx, end_idx))
            start_idx = end_idx
    else:
        raise TypeError(
            f"The type of the `division` should be dict, "
            f"tuple or list but is {type(division)} instead. "
        )
    return ranges


def _check_division_config_types_correctness(
    division: list[float] | tuple[float, ...] | dict[str, float],
) -> None:
    if isinstance(division, (list | tuple)):
        if not all(isinstance(x, float) for x in division):
            raise TypeError(
                "List or tuple values of `division` must contain only floats, "
                "other types are not allowed."
            )
    elif isinstance(division, dict):
        if not all(isinstance(x, float) for x in division.values()):
            raise TypeError(
                "Dict values of `division` must be only floats, "
                "other types are not allowed."
            )
    else:
        raise TypeError("`division` must be a list, tuple, or dict.")


def _check_division_config_values_correctness(
    division: list[float] | tuple[float, ...] | dict[str, float],
) -> None:
    if isinstance(division, (list | tuple)):
        if not all(0 < x <= 1 for x in division):
            raise ValueError(
                "All fractions for the division must be greater than 0 and smaller or "
                "equal to 1."
            )
        fraction_sum_from_list_tuple = sum(division)
        if fraction_sum_from_list_tuple > 1:
            raise ValueError("Sum of fractions for division must not exceed 1.")
        if fraction_sum_from_list_tuple < 1:
            warnings.warn(
                f"Sum of fractions for division is {sum(division)}, which is below 1. "
                f"Make sure that's the desired behavior. Some data will not be used "
                f"in the current specification.",
                stacklevel=1,
            )
    elif isinstance(division, dict):
        values = list(division.values())
        if not all(0 < x <= 1 for x in values):
            raise ValueError(
                "All fractions must be greater than 0 and smaller or equal to 1."
            )
        if sum(values) > 1:
            raise ValueError("Sum of fractions must not exceed 1.")
        if sum(values) < 1:
            warnings.warn(
                f"Sum of fractions in `division` is {values}, which is below 1. "
                f"Make sure that's the desired behavior. Some data will not be used "
                f"in the current specification.",
                stacklevel=1,
            )
    else:
        raise TypeError("`division` must be a list, tuple, or dict.")


def _check_division_config_correctness(
    division: list[float] | tuple[float, ...] | dict[str, float],
) -> None:
    _check_division_config_types_correctness(division)
    _check_division_config_values_correctness(division)


def concatenate_divisions(
    partitioner: Partitioner,
    partition_division: list[float] | tuple[float, ...] | dict[str, float],
    division_id: int | str,
) -> Dataset:
    """Create a dataset by concatenation of divisions from all partitions.

    The divisions are created based on the `partition_division` and accessed based
    on the `division_id`. This function can be used to create e.g. centralized dataset
    from federated on-edge test sets.

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner object with assigned dataset.
    partition_division : Union[List[float], Tuple[float, ...], Dict[str, float]]
        Fractions specifying the division of the partitions of a `partitioner`. You can
        think of this as on-edge division of the data into multiple divisions
        (e.g. into train and validation). E.g. [0.8, 0.2] or
        {"partition_train": 0.8, "partition_test": 0.2}.
    division_id : Union[int, str]
        The way to access the division (from a List or DatasetDict). If your
        `partition_division` is specified as a list, then `division_id` represents an
        index to an element in that list. If `partition_division` is passed as a
        `Dict`, then `division_id` is a key of such dictionary.

    Returns
    -------
    concatenated_divisions : Dataset
        A dataset created as concatenation of the divisions from all partitions.

    Examples
    --------
    Use `concatenate_divisions` with division specified as a list.

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.utils import concatenate_divisions
    >>>
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
    >>> concatenated_divisions = concatenate_divisions(
    ...     partitioner=fds.partitioners["train"],
    ...     partition_division=[0.8, 0.2],
    ...     division_id=1
    ... )
    >>> print(concatenated_divisions)

    Use `concatenate_divisions` with division specified as a dict.
    This accomplishes the same goal as the example with a list above.

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.utils import concatenate_divisions
    >>>
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
    >>> concatenated_divisions = concatenate_divisions(
    ...     partitioner=fds["train"],
    ...     partition_division={"train": 0.8, "test": 0.2},
    ...     division_id="test"
    ... )
    >>> print(concatenated_divisions)
    """
    _check_division_config_correctness(partition_division)
    divisions = []
    zero_len_divisions = 0
    for partition_id in range(partitioner.num_partitions):
        partition = partitioner.load_partition(partition_id)
        if isinstance(partition_division, (list | tuple)):
            if not isinstance(division_id, int):
                raise TypeError(
                    "The `division_id` needs to be an int in case of "
                    "`partition_division` specification as List."
                )
            partition = divide_dataset(partition, partition_division)
            division = partition[division_id]
        elif isinstance(partition_division, dict):
            partition = divide_dataset(partition, partition_division)
            division = partition[division_id]
        else:
            raise TypeError(
                "The type of partition needs to be List of DatasetDict in this "
                "context."
            )
        if len(division) == 0:
            zero_len_divisions += 1
        divisions.append(division)

    if zero_len_divisions == partitioner.num_partitions:
        raise ValueError(
            "The concatenated dataset is of length 0. Please change the "
            "`partition_division` parameter to change this behavior."
        )
    if zero_len_divisions != 0:
        warnings.warn(
            f"{zero_len_divisions} division(s) have length zero.", stacklevel=1
        )
    return concatenate_datasets(divisions)
