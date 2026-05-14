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
"""VerticalEvenPartitioner class."""

# noqa: E501
# pylint: disable=C0301, R0902, R0913
from typing import Literal

import numpy as np

import datasets
from flwr_datasets.partitioner.partitioner import Partitioner
from flwr_datasets.partitioner.vertical_partitioner_utils import (
    _add_active_party_columns,
    _init_optional_str_or_list_str,
    _list_split,
)


class VerticalEvenPartitioner(Partitioner):
    """Partitioner that splits features (columns) evenly into vertical partitions.

    Enables selection of "active party" column(s) and placement into
    a specific partition or creation of a new partition just for it.
    Also enables dropping columns and sharing specified columns across
    all partitions.

    Parameters
    ----------
    num_partitions : int
        Number of partitions to create.
    active_party_column : Optional[Union[str, list[str]]]
        Column(s) (typically representing labels) associated with the
        "active party" (which can be the server).
    active_party_columns_mode : Union[Literal[["add_to_first", "add_to_last", "create_as_first", "create_as_last", "add_to_all"], int]
        Determines how to assign the active party columns:

        - `"add_to_first"`: Append active party columns to the first partition.
        - `"add_to_last"`: Append active party columns to the last partition.
        - `"create_as_first"`: Create a new partition at the start containing only these columns.
        - `"create_as_last"`: Create a new partition at the end containing only these columns.
        - `"add_to_all"`: Append active party columns to all partitions.
        - int: Append active party columns to the specified partition index.
    drop_columns : Optional[Union[str, list[str]]]
        Columns to remove entirely from the dataset before partitioning.
    shared_columns : Optional[Union[str, list[str]]]
        Columns to duplicate into every partition after initial partitioning.
    shuffle : bool
        Whether to shuffle the order of columns before partitioning.
    seed : Optional[int]
        Random seed for shuffling columns. Has no effect if `shuffle=False`.

    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import VerticalEvenPartitioner
    >>>
    >>> partitioner = VerticalEvenPartitioner(
    ...     num_partitions=3,
    ...     active_party_columns="income",
    ...     active_party_columns_mode="add_to_last",
    ...     shuffle=True,
    ...     seed=42
    ... )
    >>> fds = FederatedDataset(
    ...     dataset="scikit-learn/adult-census-income",
    ...     partitioners={"train": partitioner}
    ... )
    >>> partitions = [fds.load_partition(i) for i in range(fds.partitioners["train"].num_partitions)]
    >>> print([partition.column_names for partition in partitions])
    """

    def __init__(  # pylint: disable=R0917
        self,
        num_partitions: int,
        active_party_columns: str | list[str] | None = None,
        active_party_columns_mode: (
            Literal[
                "add_to_first",
                "add_to_last",
                "create_as_first",
                "create_as_last",
                "add_to_all",
            ]
            | int
        ) = "add_to_last",
        drop_columns: str | list[str] | None = None,
        shared_columns: str | list[str] | None = None,
        shuffle: bool = True,
        seed: int | None = 42,
    ) -> None:
        super().__init__()

        self._num_partitions = num_partitions
        self._active_party_columns = _init_optional_str_or_list_str(
            active_party_columns
        )
        self._active_party_columns_mode = active_party_columns_mode
        self._drop_columns = _init_optional_str_or_list_str(drop_columns)
        self._shared_columns = _init_optional_str_or_list_str(shared_columns)
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

        self._partition_columns: list[list[str]] | None = None
        self._partitions_determined = False

        self._validate_parameters_in_init()

    def _determine_partitions_if_needed(self) -> None:
        if self._partitions_determined:
            return

        if self.dataset is None:
            raise ValueError("No dataset is set for this partitioner.")

        all_columns = list(self.dataset.column_names)
        self._validate_parameters_while_partitioning(
            all_columns, self._shared_columns, self._active_party_columns
        )
        columns = [column for column in all_columns if column not in self._drop_columns]
        columns = [column for column in columns if column not in self._shared_columns]
        columns = [
            column for column in columns if column not in self._active_party_columns
        ]

        if self._shuffle:
            self._rng.shuffle(columns)
        partition_columns = _list_split(columns, self._num_partitions)
        partition_columns = _add_active_party_columns(
            self._active_party_columns,
            self._active_party_columns_mode,
            partition_columns,
        )

        # Add shared columns to all partitions
        for partition in partition_columns:
            for column in self._shared_columns:
                partition.append(column)

        self._partition_columns = partition_columns
        self._partitions_determined = True

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
        self._determine_partitions_if_needed()
        assert self._partition_columns is not None
        if partition_id < 0 or partition_id >= len(self._partition_columns):
            raise ValueError(f"Invalid partition_id {partition_id}.")
        columns = self._partition_columns[partition_id]
        return self.dataset.select_columns(columns)

    @property
    def num_partitions(self) -> int:
        """Number of partitions."""
        self._determine_partitions_if_needed()
        assert self._partition_columns is not None
        return len(self._partition_columns)

    def _validate_parameters_in_init(self) -> None:
        if self._num_partitions < 1:
            raise ValueError("`column_distribution` as int must be >= 1.")

        valid_modes = {
            "add_to_first",
            "add_to_last",
            "create_as_first",
            "create_as_last",
            "add_to_all",
        }
        if not (
            isinstance(self._active_party_columns_mode, int)
            or self._active_party_columns_mode in valid_modes
        ):
            raise ValueError(
                "`active_party_column_mode` must be an int or one of "
                "'add_to_first', 'add_to_last', 'create_as_first', 'create_as_last', "
                "'add_to_all'."
            )

    def _validate_parameters_while_partitioning(
        self,
        all_columns: list[str],
        shared_columns: list[str],
        active_party_column: str | list[str],
    ) -> None:
        if isinstance(active_party_column, str):
            active_party_column = [active_party_column]
        # Shared columns existence check
        for column in shared_columns:
            if column not in all_columns:
                raise ValueError(f"Shared column '{column}' not found in the dataset.")
        # Active party columns existence check
        for column in active_party_column:
            if column not in all_columns:
                raise ValueError(
                    f"Active party column '{column}' not found in the dataset."
                )
