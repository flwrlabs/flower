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
"""Merger class for Flower Datasets."""

import collections
import warnings
from functools import reduce

import datasets
from datasets import Dataset, DatasetDict


class Merger:
    """Merge existing splits of the dataset and assign them custom names.

    Create new `DatasetDict` with new split names corresponding to the merged existing
    splits (e.g. "train", "valid" and "test").

    Parameters
    ----------
    merge_config : Dict[str, Tuple[str, ...]]
        Dictionary with keys - the desired split names to values - tuples of the current
        split names that will be merged together

    Examples
    --------
    Create new `DatasetDict` with a split name "new_train" that is created as a merger
    of the "train" and "valid" splits. Keep the "test" split.

    >>> # Assuming there is a dataset_dict of type `DatasetDict`
    >>> # dataset_dict is {"train": train-data, "valid": valid-data, "test": test-data}
    >>> merger = Merger(
    >>>     merge_config={
    >>>         "new_train": ("train", "valid"),
    >>>         "test": ("test", )
    >>>     }
    >>> )
    >>> new_dataset_dict = merger(dataset_dict)
    >>> # new_dataset_dict is
    >>> # {"new_train": concatenation of train-data and valid-data, "test": test-data}
    """

    def __init__(
        self,
        merge_config: dict[str, tuple[str, ...]],
    ) -> None:
        self._merge_config: dict[str, tuple[str, ...]] = merge_config
        self._check_duplicate_merge_splits()

    def __call__(self, dataset: DatasetDict) -> DatasetDict:
        """Resplit the dataset according to the `merge_config`."""
        self._check_correct_keys_in_merge_config(dataset)
        return self.resplit(dataset)

    def resplit(self, dataset: DatasetDict) -> DatasetDict:
        """Resplit the dataset according to the `merge_config`."""
        resplit_dataset = {}
        for divide_to, divided_from__list in self._merge_config.items():
            datasets_from_list: list[Dataset] = []
            for divide_from in divided_from__list:
                datasets_from_list.append(dataset[divide_from])
            if len(datasets_from_list) > 1:
                resplit_dataset[divide_to] = datasets.concatenate_datasets(
                    datasets_from_list
                )
            else:
                resplit_dataset[divide_to] = datasets_from_list[0]
        return datasets.DatasetDict(resplit_dataset)

    def _check_correct_keys_in_merge_config(self, dataset: DatasetDict) -> None:
        """Check if the keys in merge_config are existing dataset splits."""
        dataset_keys = dataset.keys()
        specified_dataset_keys = self._merge_config.values()
        for key_list in specified_dataset_keys:
            for key in key_list:
                if key not in dataset_keys:
                    raise ValueError(
                        f"The given dataset key '{key}' is not present in the given "
                        f"dataset object. Make sure to use only the keywords that are "
                        f"available in your dataset."
                    )

    def _check_duplicate_merge_splits(self) -> None:
        """Check if the original splits are duplicated for new splits creation."""
        merge_splits = reduce(lambda x, y: x + y, self._merge_config.values())
        duplicates = [
            item
            for item, count in collections.Counter(merge_splits).items()
            if count > 1
        ]
        if duplicates:
            warnings.warn(
                f"More than one desired splits used '{duplicates[0]}' in "
                f"`merge_config`. Make sure that is the intended behavior.",
                stacklevel=1,
            )
