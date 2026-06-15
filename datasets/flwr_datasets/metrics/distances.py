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
"""Distance metrics for partition distributions."""


from collections.abc import Callable

import numpy as np
import pandas as pd

from flwr_datasets.common.typing import NDArray
from flwr_datasets.metrics.utils import _compute_counts, compute_counts
from flwr_datasets.partitioner import Partitioner


def compute_hellinger_distances(
    partitioner: Partitioner,
    column_name: str,
    max_num_partitions: int | None = None,
) -> pd.Series:
    """Compute Hellinger distances between partitions and the full dataset.

    The distance is computed between each partition's categorical distribution over
    ``column_name`` and the full dataset distribution over the same column. Values are
    in the interval ``[0, 1]``. A value of ``0`` means that the partition distribution
    exactly matches the full dataset distribution.

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner with an assigned dataset.
    column_name : str
        Column name identifying the categorical values used to compute the
        distributions.
    max_num_partitions : Optional[int]
        The maximum number of partitions that will be used. If greater than the
        total number of partitions in a partitioner, it won't have an effect. If left
        as None, then all partitions will be used.

    Returns
    -------
    distances : pd.Series
        Hellinger distance for each partition id.
    """
    return _compute_partition_distances(
        partitioner=partitioner,
        column_name=column_name,
        max_num_partitions=max_num_partitions,
        distance_fn=_hellinger_distance,
        name="Hellinger distance",
    )


def compute_jensen_shannon_distances(
    partitioner: Partitioner,
    column_name: str,
    max_num_partitions: int | None = None,
) -> pd.Series:
    """Compute Jensen-Shannon distances between partitions and the full dataset.

    The distance is computed between each partition's categorical distribution over
    ``column_name`` and the full dataset distribution over the same column. Values are
    in the interval ``[0, 1]``. A value of ``0`` means that the partition distribution
    exactly matches the full dataset distribution.

    Parameters
    ----------
    partitioner : Partitioner
        Partitioner with an assigned dataset.
    column_name : str
        Column name identifying the categorical values used to compute the
        distributions.
    max_num_partitions : Optional[int]
        The maximum number of partitions that will be used. If greater than the
        total number of partitions in a partitioner, it won't have an effect. If left
        as None, then all partitions will be used.

    Returns
    -------
    distances : pd.Series
        Jensen-Shannon distance for each partition id.
    """
    return _compute_partition_distances(
        partitioner=partitioner,
        column_name=column_name,
        max_num_partitions=max_num_partitions,
        distance_fn=_jensen_shannon_distance,
        name="Jensen-Shannon distance",
    )


def _compute_partition_distances(
    partitioner: Partitioner,
    column_name: str,
    max_num_partitions: int | None,
    distance_fn: Callable[[NDArray, NDArray], float],
    name: str,
) -> pd.Series:
    """Compute distances from partition distributions to the full distribution."""
    _check_max_num_partitions(max_num_partitions)
    counts = compute_counts(
        partitioner=partitioner,
        column_name=column_name,
        max_num_partitions=max_num_partitions,
    )
    reference_counts = _compute_counts(
        labels=partitioner.dataset[column_name],
        unique_labels=list(counts.columns),
    )
    frequencies = _normalize_counts(counts)
    reference_frequency = _normalize_count_series(reference_counts)

    return pd.Series(
        [
            distance_fn(
                partition_frequency.to_numpy(dtype=float),
                reference_frequency.to_numpy(dtype=float),
            )
            for _, partition_frequency in frequencies.iterrows()
        ],
        index=frequencies.index,
        name=name,
    )


def _check_max_num_partitions(max_num_partitions: int | None) -> None:
    """Check that the optional partition limit is positive."""
    if max_num_partitions is not None and max_num_partitions <= 0:
        raise ValueError("max_num_partitions must be greater than zero.")


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


def _hellinger_distance(first: NDArray, second: NDArray) -> float:
    """Compute Hellinger distance between two probability distributions."""
    return float(np.linalg.norm(np.sqrt(first) - np.sqrt(second)) / np.sqrt(2))


def _jensen_shannon_distance(first: NDArray, second: NDArray) -> float:
    """Compute Jensen-Shannon distance between two probability distributions."""
    midpoint = (first + second) / 2
    divergence = (
        _kl_divergence(first, midpoint) + _kl_divergence(second, midpoint)
    ) / 2
    return float(np.sqrt(divergence))


def _kl_divergence(first: NDArray, second: NDArray) -> float:
    """Compute base-2 Kullback-Leibler divergence, ignoring zero-probability terms."""
    non_zero = first > 0
    return float(np.sum(first[non_zero] * np.log2(first[non_zero] / second[non_zero])))
