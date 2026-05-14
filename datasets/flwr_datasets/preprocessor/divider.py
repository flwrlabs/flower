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
"""Divider class for Flower Datasets."""

import collections
import warnings
from typing import cast

import datasets
from datasets import DatasetDict


# noqa: E501
# pylint: disable=line-too-long
class Divider:
    """Dive existing split(s) of the dataset and assign them custom names.

    Create new `DatasetDict` with new split names with corresponding percentages of data
    and custom names.

    Parameters
    ----------
    divide_config: Union[Dict[str, int], Dict[str, float], Dict[str, Dict[str, int]], Dict[str, Dict[str, float]]]
        If single level dictionary, keys represent the split names. If values are: int,
        they represent the number of samples in each split; float, they represent the
        fraction of the total samples assigned to that split. These fractions do not
        have to sum up to 1.0. The order of values (either int or float) matter: the
        first key will get the first split starting from the beginning of the dataset,
        and so on.
        If two level dictionary (dictionary of dictionaries) then the first keys are
        the split names that will be divided into different splits. It's an alternative
        to specifying `divide_split` if you need to divide many splits.
    divide_split: Optional[str]
        In case of single level dictionary specification of `divide_config`, specifies
        the split name that will be divided. Might be left None in case of a single-
        split dataset (it will be automatically inferred). Ignored in case of
        multi-split configuration.
    drop_remaining_splits: bool
        In case of single level dictionary specification of `divide_config`, specifies
        if the splits that are not divided are dropped.

    Raises
    ------
    ValuesError if the specified name of a new split is already present in the dataset
    and the `drop_remaining_splits` is False.

    Examples
    --------
    Create new `DatasetDict` with a divided split "train" into "train" and "valid"
    splits by using 80% and 20% correspondingly. Keep the "test" split.

    1) Using the `divide_split` parameter and "smaller" (i.e. single-level) divide_config

    >>> # Assuming there is a dataset_dict of type `DatasetDict`
    >>> # dataset_dict is {"train": train-data, "test": test-data}
    >>> divider = Divider(
    >>>     divide_config={
    >>>         "train": 0.8,
    >>>         "valid": 0.2,
    >>>     }
    >>>     divide_split="train",
    >>> )
    >>> new_dataset_dict = divider(dataset_dict)
    >>> # new_dataset_dict is
    >>> # {"train": 80% of train, "valid": 20% of train, "test": test-data}

    1) Using "bigger" (i.e. two-level dict) version of divide_config and no
    `divide_split` to accomplish the same (splitting train into train, valid with 80%,
    20% correspondingly) and additionally dividing the test set.

    >>> # Assuming there is a dataset_dict of type `DatasetDict`
    >>> # dataset_dict is {"train": train-data, "test": test-data}
    >>> divider = Divider(
    >>>     divide_config={
    >>>         "train": {
    >>>             "train": 0.8,
    >>>             "valid": 0.2,
    >>>         },
    >>>         "test": {"test-a": 0.4, "test-b": 0.6 }
    >>>     }
    >>> )
    >>> new_dataset_dict = divider(dataset_dict)
    >>> # new_dataset_dict is
    >>> # {"train": 80% of train, "valid": 20% of train,
    >>> # "test-a": 40% of test, "test-b": 60% of test}
    """

    def __init__(
        self,
        divide_config: (
            dict[str, float]
            | dict[str, int]
            | dict[str, dict[str, float]]
            | dict[str, dict[str, int]]
        ),
        divide_split: str | None = None,
        drop_remaining_splits: bool = False,
    ) -> None:
        self._single_split_config: dict[str, float] | dict[str, int]
        self._multiple_splits_config: (
            dict[str, dict[str, float]] | dict[str, dict[str, int]]
        )

        self._config_type = _determine_config_type(divide_config)
        self._check_type_correctness(divide_config)
        if self._config_type == "single-split":
            self._single_split_config = cast(
                dict[str, float] | dict[str, int], divide_config
            )
        else:
            self._multiple_splits_config = cast(
                dict[str, dict[str, float]] | dict[str, dict[str, int]],
                divide_config,
            )
        self._divide_split = divide_split
        self._drop_remaining_splits = drop_remaining_splits
        self._check_duplicate_splits_in_config()
        self._warn_on_potential_misuse_of_divide_split()

    def __call__(self, dataset: DatasetDict) -> DatasetDict:
        """Resplit the dataset according to the configuration."""
        if self._drop_remaining_splits is False:
            dataset_splits = list(dataset.keys())
            self._check_duplicate_splits_in_config_and_original_dataset(dataset_splits)
        return self.resplit(dataset)

    # pylint: disable=too-many-branches
    def resplit(self, dataset: DatasetDict) -> DatasetDict:
        """Resplit the dataset according to the configuration."""
        resplit_dataset = {}
        dataset_splits: list[str] = list(dataset.keys())
        # Change the "single-split" config to look like "multiple-split" config
        if self._config_type == "single-split":
            # First, if the `divide_split` is None determine the split
            if self._divide_split is None:
                if len(dataset_splits) != 1:
                    raise ValueError(
                        "When giving the config that is single level and working with "
                        "dataset with more than one split you need to specify the "
                        "`divide_split` but current value is None."
                    )
                self._divide_split = dataset_splits[0]
            self._multiple_splits_config = cast(
                dict[str, dict[str, float]] | dict[str, dict[str, int]],
                {self._divide_split: self._single_split_config},
            )

        self._check_size_values(dataset)
        # Continue with the resplitting process
        # Move the non-split splits if they exist
        if self._drop_remaining_splits is False:
            if len(dataset_splits) >= 2:
                split_splits = set(self._multiple_splits_config.keys())
                non_split_splits = list(set(dataset_splits) - split_splits)
                for non_split_split in non_split_splits:
                    resplit_dataset[non_split_split] = dataset[non_split_split]
        else:
            # The remaining data is not kept (by simply not coping it=the reference)
            pass

        # Split the splits
        for split_from, new_splits_dict in self._multiple_splits_config.items():
            start_index = 0
            end_index = 0
            split_data = dataset[split_from]
            for new_split_name, size in new_splits_dict.items():
                if isinstance(size, float):
                    end_index += int(len(split_data) * size)
                elif isinstance(size, int):
                    end_index += size
                else:
                    raise ValueError(
                        "The type of size value for the divide config must "
                        "be int or float."
                    )
                if end_index > len(split_data):
                    raise ValueError(
                        "The size specified in the `divide_config` is greater than "
                        "the size of the dataset."
                    )
                if end_index == start_index:
                    raise ValueError(
                        f"The size specified in the `divide_config` results in the "
                        f"dataset of size 0. The problem occurred in {new_splits_dict}."
                        f"Please make sure to provide sizes that do not produce empty"
                        f"datasets."
                    )
                resplit_dataset[new_split_name] = split_data.select(
                    range(start_index, end_index)
                )
                start_index = end_index
        return datasets.DatasetDict(resplit_dataset)

    def _check_duplicate_splits_in_config(self) -> None:
        """Check if the new split names are duplicated in `divide_config`."""
        if self._config_type == "single-split":
            new_splits = list(self._single_split_config.keys())
        elif self._config_type == "multiple-splits":
            new_splits = []
            for new_splits_dict in self._multiple_splits_config.values():
                new_values = list(new_splits_dict.keys())
                new_splits.extend(new_values)
        else:
            raise ValueError("Incorrect type of config.")

        duplicates = [
            item for item, count in collections.Counter(new_splits).items() if count > 1
        ]
        if duplicates:
            raise ValueError(
                f"`divide_config` contains duplicates ({duplicates}). Please specify"
                "unique values for each new split."
            )

    def _check_duplicate_splits_in_config_and_original_dataset(
        self, dataset_splits: list[str]
    ) -> None:
        """Check duplicates along the new split values and dataset splits.

        This check can happen only at the time this class is called (it does not have
        access to the dataset prior to that).
        """
        if self._config_type == "single-split":
            new_splits = list(self._single_split_config.keys())
            all_splits = dataset_splits + new_splits
            assert self._divide_split is not None
            all_splits.pop(all_splits.index(self._divide_split))
        elif self._config_type == "multiple-splits":
            new_splits = []
            for new_splits_dict in self._multiple_splits_config.values():
                new_splits.extend(list(new_splits_dict.keys()))
            all_splits = dataset_splits + new_splits
            for used_split in self._multiple_splits_config.keys():
                all_splits.pop(all_splits.index(used_split))
        else:
            raise ValueError("Incorrect type of config.")

        duplicates = [
            item for item, count in collections.Counter(all_splits).items() if count > 1
        ]
        if duplicates:
            raise ValueError(
                "The specified values of the new splits in "
                f"`divide_config` are duplicated ({duplicates}) with the split names of"
                " the datasets. Please specify unique values for each new split."
            )

    def _check_size_values(self, dataset: DatasetDict) -> None:
        # It should be called after the `divide_config` is in the multiple-splits format
        assert self._multiple_splits_config is not None
        for split_from, new_split_dict in self._multiple_splits_config.items():
            if all(isinstance(x, float) for x in new_split_dict.values()):
                if not all(0 < x <= 1 for x in new_split_dict.values()):
                    raise ValueError(
                        "All fractions in `divide_config` must be greater than 0 and "
                        "smaller or equal to 1."
                    )
                if sum(new_split_dict.values()) > 1.0:
                    raise ValueError(
                        "The sum of the fractions in `divide_config` must be smaller "
                        "than 1.0."
                    )

            elif all(isinstance(x, int) for x in new_split_dict.values()):
                dataset_len = len(dataset[split_from])
                len_from_divide_resplit = sum(new_split_dict.values())
                if len_from_divide_resplit > dataset_len:
                    raise ValueError(
                        f"The sum of the sample numbers in `divide_config` must be "
                        f"smaller than the split size. This is not the case for "
                        f"{split_from} split which is of length {dataset_len} and the "
                        f"sum in the supplied `divide_config` is "
                        f"{len_from_divide_resplit}."
                    )
            else:
                raise TypeError(
                    "The values in `divide_config` must be either ints or floats. "
                    "The mix of them or other types are not allowed."
                )

    def _warn_on_potential_misuse_of_divide_split(self) -> None:
        if self._config_type == "multiple-splits" and self._divide_split is not None:
            warnings.warn(
                "The `divide_split` was specified but the multiple split "
                "configuration was given. The `divide_split` will be "
                "ignored.",
                stacklevel=1,
            )

    def _check_type_correctness(
        self,
        divide_config: (
            dict[str, float]
            | dict[str, int]
            | dict[str, dict[str, float]]
            | dict[str, dict[str, int]]
        ),
    ) -> None:
        assert self._config_type in [
            "single-split",
            "multiple-splits",
        ], "Incorrect config type"
        if self._config_type == "single-split":
            if all(
                isinstance(key, str) and isinstance(value, float)
                for key, value in divide_config.items()
            ):
                return
            if all(
                isinstance(key, str) and isinstance(value, int)
                for key, value in divide_config.items()
            ):
                return

            raise ValueError(
                "Dictionary for single-split config does not match required type "
                "Dict[str, float] or Dict[str, int]"
            )

        # multiple-splits
        if all(
            isinstance(key, str)
            and isinstance(value, dict)
            and all(
                isinstance(k, str) and isinstance(v, float) for k, v in value.items()
            )
            for key, value in divide_config.items()
        ):
            return
        if all(
            isinstance(key, str)
            and isinstance(value, dict)
            and all(isinstance(k, str) and isinstance(v, int) for k, v in value.items())
            for key, value in divide_config.items()
        ):
            return

        raise ValueError(
            "Multi-split dictionary does not match required type "
            "Dict[str, Dict[str, float]] or Dict[str, Dict[str, int]]"
        )


def _determine_config_type(
    config: (
        dict[str, float]
        | dict[str, int]
        | dict[str, dict[str, float]]
        | dict[str, dict[str, int]]
    ),
) -> str:
    """Determine configuration type of `divide_config` based on the dict structure.

    Two possible configuration are possible: 1) single-split single-level (works
    together with `divide_split`), 2) nested/two-level that works with multiple
    splits (`divide_split` is ignored).

    Returns
    -------
    config_type: str
        "single-split" or "multiple-splits"
    """
    if not isinstance(config, dict):
        raise ValueError("Provided input dictionary is not a dictionary")
    for value in config.values():
        # Check if the value is a dictionary
        if isinstance(value, dict):
            return "multiple-splits"
    # If no dictionary values are found, it is single-level
    return "single-split"
