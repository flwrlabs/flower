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
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

from flwr_datasets.common.typing import NDArray
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


def compute_hellinger_distances(
    partitioner: Partitioner,
    column_name: str,
    max_num_partitions: int | None = None,
    bins: int | Sequence[float] | None = None,
) -> pd.Series:
    """Compute Hellinger distances between partitions and the full dataset.

    The distance is computed between each partition's distribution over
    ``column_name`` and the full dataset distribution over the same column. Values are
    in the interval ``[0, 1]``. A value of ``0`` means that the partition distribution
    exactly matches the full dataset distribution.

    For continuous target columns, pass ``bins`` to discretize values before computing
    the distributions.

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner with an assigned dataset.
    column_name : str
        Column name identifying the values used to compute the distributions.
    max_num_partitions : Optional[int]
        The maximum number of partitions that will be used. If greater than the
        total number of partitions in a partitioner, it won't have an effect. If left
        as None, then all partitions will be used.
    bins : Optional[Union[int, Sequence[float]]]
        Bin specification passed to ``pandas.cut``. Use this when ``column_name``
        contains continuous values.

    Returns
    -------
    distances : pd.Series
        Hellinger distance for each partition id.
    """
    frequencies, reference_frequency = _compute_partition_and_reference_frequencies(
        partitioner=partitioner,
        column_name=column_name,
        max_num_partitions=max_num_partitions,
        bins=bins,
    )
    distances = pd.Series(
        [
            _hellinger_distance(
                partition_frequency.to_numpy(dtype=float),
                reference_frequency.to_numpy(dtype=float),
            )
            for _, partition_frequency in frequencies.iterrows()
        ],
        index=frequencies.index,
        name="Hellinger distance",
    )
    return distances


def compute_jensen_shannon_distances(
    partitioner: Partitioner,
    column_name: str,
    max_num_partitions: int | None = None,
    bins: int | Sequence[float] | None = None,
    base: float | None = 2.0,
) -> pd.Series:
    """Compute Jensen-Shannon distances between partitions and the full dataset.

    The distance is computed between each partition's distribution over
    ``column_name`` and the full dataset distribution over the same column. With the
    default logarithm base ``2``, values are in the interval ``[0, 1]``. A value of
    ``0`` means that the partition distribution exactly matches the full dataset
    distribution.

    For continuous target columns, pass ``bins`` to discretize values before computing
    the distributions.

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner with an assigned dataset.
    column_name : str
        Column name identifying the values used to compute the distributions.
    max_num_partitions : Optional[int]
        The maximum number of partitions that will be used. If greater than the
        total number of partitions in a partitioner, it won't have an effect. If left
        as None, then all partitions will be used.
    bins : Optional[Union[int, Sequence[float]]]
        Bin specification passed to ``pandas.cut``. Use this when ``column_name``
        contains continuous values.
    base : Optional[float]
        Logarithm base used in the Jensen-Shannon divergence. Use ``None`` for the
        natural logarithm.

    Returns
    -------
    distances : pd.Series
        Jensen-Shannon distance for each partition id.
    """
    if base is not None and (base <= 0 or base == 1):
        raise ValueError("base must be greater than 0 and not equal to 1.")

    frequencies, reference_frequency = _compute_partition_and_reference_frequencies(
        partitioner=partitioner,
        column_name=column_name,
        max_num_partitions=max_num_partitions,
        bins=bins,
    )
    distances = pd.Series(
        [
            _jensen_shannon_distance(
                partition_frequency.to_numpy(dtype=float),
                reference_frequency.to_numpy(dtype=float),
                base=base,
            )
            for _, partition_frequency in frequencies.iterrows()
        ],
        index=frequencies.index,
        name="Jensen-Shannon distance",
    )
    return distances


def _compute_partition_and_reference_frequencies(
    partitioner: Partitioner,
    column_name: str,
    max_num_partitions: int | None,
    bins: int | Sequence[float] | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute partition and full-dataset frequencies over a shared support."""
    max_num_partitions = _resolve_max_num_partitions(partitioner, max_num_partitions)
    if bins is None:
        counts = compute_counts(
            partitioner=partitioner,
            column_name=column_name,
            max_num_partitions=max_num_partitions,
        )
        reference_counts = _compute_counts(
            labels=partitioner.dataset[column_name],
            unique_labels=list(counts.columns),
        )
    else:
        counts, reference_counts = _compute_binned_counts(
            partitioner=partitioner,
            column_name=column_name,
            max_num_partitions=max_num_partitions,
            bins=bins,
        )
    return _normalize_counts(counts), _normalize_count_series(reference_counts)


def _compute_binned_counts(
    partitioner: Partitioner,
    column_name: str,
    max_num_partitions: int,
    bins: int | Sequence[float],
) -> tuple[pd.DataFrame, pd.Series]:
    """Compute counts after binning a continuous column."""
    _check_column_name(partitioner, column_name)
    _check_bins(bins)

    reference_values = pd.Series(partitioner.dataset[column_name])
    reference_binned, bin_edges = pd.cut(
        reference_values, bins=bins, include_lowest=True, retbins=True
    )
    categories = reference_binned.cat.categories
    reference_counts = reference_binned.value_counts(sort=False).reindex(
        categories, fill_value=0
    )

    partition_id_to_counts = {}
    for partition_id in range(max_num_partitions):
        partition = partitioner.load_partition(partition_id)
        partition_values = pd.Series(partition[column_name])
        partition_binned = pd.cut(partition_values, bins=bin_edges, include_lowest=True)
        partition_id_to_counts[partition_id] = partition_binned.value_counts(
            sort=False
        ).reindex(categories, fill_value=0)

    counts = pd.DataFrame.from_dict(partition_id_to_counts, orient="index")
    counts.index.name = "Partition ID"
    return counts, reference_counts


def _normalize_counts(counts: pd.DataFrame) -> pd.DataFrame:
    """Normalize count rows into frequency rows."""
    row_totals = counts.sum(axis=1)
    if (row_totals <= 0).any():
        raise ValueError("Cannot compute distances for empty partitions.")
    return counts.astype(float).div(row_totals, axis=0)


def _normalize_count_series(counts: pd.Series) -> pd.Series:
    """Normalize counts into frequencies."""
    total = counts.sum()
    if total <= 0:
        raise ValueError("Cannot compute distances for an empty dataset.")
    return counts.astype(float).divide(total)


def _resolve_max_num_partitions(
    partitioner: Partitioner, max_num_partitions: int | None
) -> int:
    """Resolve the number of partitions to include."""
    if max_num_partitions is None:
        return partitioner.num_partitions
    if max_num_partitions <= 0:
        raise ValueError("max_num_partitions must be greater than zero.")
    return min(max_num_partitions, partitioner.num_partitions)


def _check_column_name(partitioner: Partitioner, column_name: str) -> None:
    """Check that column_name is present in the dataset."""
    if column_name not in partitioner.dataset.column_names:
        raise ValueError(
            f"The specified 'column_name': '{column_name}' is not present in the "
            f"dataset. The dataset contains columns {partitioner.dataset.column_names}."
        )


def _check_bins(bins: int | Sequence[float]) -> None:
    """Check that bins can define at least one interval."""
    if isinstance(bins, int):
        if bins <= 0:
            raise ValueError("bins must be greater than zero.")
        return
    if len(bins) < 2:
        raise ValueError("bins must contain at least two edges.")


def _hellinger_distance(first: NDArray, second: NDArray) -> float:
    """Compute Hellinger distance between two probability distributions."""
    return float(np.linalg.norm(np.sqrt(first) - np.sqrt(second)) / np.sqrt(2))


def _jensen_shannon_distance(
    first: NDArray, second: NDArray, base: float | None
) -> float:
    """Compute Jensen-Shannon distance between two probability distributions."""
    midpoint = (first + second) / 2
    divergence = (
        _kl_divergence(first, midpoint, base) + _kl_divergence(second, midpoint, base)
    ) / 2
    return float(np.sqrt(divergence))


def _kl_divergence(first: NDArray, second: NDArray, base: float | None) -> float:
    """Compute Kullback-Leibler divergence, ignoring zero-probability terms."""
    non_zero = first > 0
    log_values = np.log(first[non_zero] / second[non_zero])
    if base is not None:
        log_values = log_values / np.log(base)
    return float(np.sum(first[non_zero] * log_values))


def _compute_counts(
    labels: Sequence[Hashable], unique_labels: Sequence[Hashable]
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
    labels: Sequence[Hashable], unique_labels: Sequence[Hashable]
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
