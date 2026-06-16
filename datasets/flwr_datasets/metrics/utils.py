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
"""Utils for metrics computation."""


import warnings

import pandas as pd

from flwr_datasets.partitioner import Partitioner


def compute_counts(
    partitioner: Partitioner,
    column_name: str,
    verbose_names: bool = False,
    max_num_partitions: int | None = None,
) -> pd.DataFrame:
    """Compute the counts of unique values in a given column in the partitions.

    Take into account all possible labels in dataset when computing count for each
    partition (assign 0 as the size when there are no values for a label in the
    partition).

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner with an assigned dataset.
    column_name : str
        Column name identifying label based on which the count will be calculated.
    verbose_names : bool
        Whether to use verbose versions of the values in the column specified by
        `column_name`. The verbose values are possible to extract if the column is a
        feature of type `ClassLabel`.
    max_num_partitions : Optional[int]
        The maximum number of partitions that will be used. If greater than the
        total number of partitions in a partitioner, it won't have an effect. If left
        as None, then all partitions will be used.

    Returns
    -------
    dataframe: pd.DataFrame
        DataFrame where the row index represent the partition id and the column index
        represent the unique values found in column specified by `column_name`
        (e.g. representing the labels). The value of the dataframe.loc[i, j] represents
        the count of the label j, in the partition of index i.

    Examples
    --------
    Generate DataFrame with label counts resulting from DirichletPartitioner on cifar10

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>> from flwr_datasets.metrics import compute_counts
    >>>
    >>> fds = FederatedDataset(
    >>>     dataset="cifar10",
    >>>     partitioners={
    >>>         "train": DirichletPartitioner(
    >>>             num_partitions=20,
    >>>             partition_by="label",
    >>>             alpha=0.3,
    >>>             min_partition_size=0,
    >>>         ),
    >>>     },
    >>> )
    >>> partitioner = fds.partitioners["train"]
    >>> counts_dataframe = compute_counts(
    >>>     partitioner=partitioner,
    >>>     column_name="label"
    >>> )
    """
    if column_name not in partitioner.dataset.column_names:
        raise ValueError(
            f"The specified 'column_name': '{column_name}' is not present in the "
            f"dataset. The dataset contains columns {partitioner.dataset.column_names}."
        )

    if max_num_partitions is None:
        max_num_partitions = partitioner.num_partitions
    else:
        max_num_partitions = min(max_num_partitions, partitioner.num_partitions)
    assert isinstance(max_num_partitions, int)
    partition = partitioner.load_partition(0)

    try:
        # Unique labels are needed to represent the correct count of each class
        # (some of the classes can have zero samples that's why this
        # adjustment is needed)
        unique_labels = partition.features[column_name].str2int(
            partition.features[column_name].names
        )
    except AttributeError:  # If the column_name is not formally a Label
        unique_labels = partitioner.dataset.unique(column_name)

    partition_id_to_label_absolute_size = {}
    for partition_id in range(max_num_partitions):
        partition = partitioner.load_partition(partition_id)
        partition_id_to_label_absolute_size[partition_id] = _compute_counts(
            partition[column_name], unique_labels
        )

    dataframe = pd.DataFrame.from_dict(
        partition_id_to_label_absolute_size, orient="index"
    )
    dataframe.index.name = "Partition ID"

    if verbose_names:
        # Adjust the column name values of the dataframe
        current_labels = dataframe.columns
        try:
            legend_names = partitioner.dataset.features[column_name].int2str(
                [int(v) for v in current_labels]
            )
            dataframe.columns = legend_names
        except AttributeError:
            warnings.warn(
                "The verbose names can not be established. "
                "The column specified by 'column_name' needs to be of type "
                "'ClassLabel' to create a verbose names. "
                "The available names will used.",
                stacklevel=1,
            )
    return dataframe


def compute_frequencies(
    partitioner: Partitioner,
    column_name: str,
    verbose_names: bool = False,
    max_num_partitions: int | None = None,
) -> pd.DataFrame:
    """Compute the frequencies of unique values in a given column in the partitions.

    The frequencies sum up to 1 for a given partition id. This function takes into
    account all possible labels in the dataset when computing the count for each
    partition (assign 0 as the size when there are no values for a label in the
    partition).

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner with an assigned dataset.
    column_name : str
        Column name identifying label based on which the count will be calculated.
    verbose_names : bool
        Whether to use verbose versions of the values in the column specified by
        `column_name`. The verbose value are possible to extract if the column is a
        feature of type `ClassLabel`.
    max_num_partitions : Optional[int]
        The maximum number of partitions that will be used. If greater than the
        total number of partitions in a partitioner, it won't have an effect. If left
        as None, then all partitions will be used.

    Returns
    -------
    dataframe: pd.DataFrame
        DataFrame where the row index represent the partition id and the column index
        represent the unique values found in column specified by `column_name`
        (e.g. representing the labels). The value of the dataframe.loc[i, j] represent
        the ratio of the label j to the total number of sample of in partition i.

    Examples
    --------
    Generate DataFrame with label counts resulting from DirichletPartitioner on cifar10

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>> from flwr_datasets.metrics import compute_frequencies
    >>>
    >>> fds = FederatedDataset(
    >>>     dataset="cifar10",
    >>>     partitioners={
    >>>         "train": DirichletPartitioner(
    >>>             num_partitions=20,
    >>>             partition_by="label",
    >>>             alpha=0.3,
    >>>             min_partition_size=0,
    >>>         ),
    >>>     },
    >>> )
    >>> partitioner = fds.partitioners["train"]
    >>> counts_dataframe = compute_frequencies(
    >>>     partitioner=partitioner,
    >>>     column_name="label"
    >>> )
    """
    dataframe = compute_counts(
        partitioner, column_name, verbose_names, max_num_partitions
    )
    dataframe = dataframe.div(dataframe.sum(axis=1), axis=0)
    return dataframe


def _compute_counts(
    labels: list[int] | list[str], unique_labels: list[int] | list[str]
) -> pd.Series:
    """Compute the count of labels when taking into account all possible labels.

    Also known as absolute frequency.

    Parameters
    ----------
    labels: Union[List[int], List[str]]
        The labels from the datasets.
    unique_labels: Union[List[int], List[str]]
        The reference all unique label. Needed to avoid missing any label, instead
        having the value equal to zero for them.

    Returns
    -------
    label_counts: pd.Series
        The pd.Series with label as indices and counts as values.
    """
    if len(unique_labels) != len(set(unique_labels)):
        raise ValueError("unique_labels must contain unique elements only.")
    labels_series = pd.Series(labels)
    label_counts = labels_series.value_counts()
    label_counts_with_zeros = pd.Series(index=unique_labels, data=0)
    label_counts_with_zeros = label_counts_with_zeros.add(
        label_counts, fill_value=0
    ).astype(int)
    return label_counts_with_zeros


def _compute_frequencies(
    labels: list[int] | list[str], unique_labels: list[int] | list[str]
) -> pd.Series:
    """Compute the distribution of labels when taking into account all possible labels.

    Also known as relative frequency.

    Parameters
    ----------
    labels: Union[List[int], List[str]]
        The labels from the datasets.
    unique_labels: Union[List[int], List[str]]
        The reference all unique label. Needed to avoid missing any label, instead
        having the value equal to zero for them.

    Returns
    -------
        The pd.Series with label as indices and probabilities as values.
    """
    counts = _compute_counts(labels, unique_labels)
    if len(labels) == 0:
        frequencies = counts.astype(float)
        return frequencies
    frequencies = counts.divide(len(labels))
    return frequencies
