# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Continuous partitioner class that works with Hugging Face Datasets."""

# pylint: disable=R0913, R0917

import numpy as np

from datasets import Dataset
from flwr_datasets.partitioner.partitioner import Partitioner


class ContinuousPartitioner(
    Partitioner
):  # pylint: disable=too-many-instance-attributes
    r"""Partitioner based on a real-valued dataset property with adjustable strictness.

    This partitioner enables non-IID partitioning by sorting the dataset according to a
    continuous (i.e., real-valued, not categorical) property and introducing controlled noise
    to adjust the level of heterogeneity.

    To interpolate between IID and non-IID partitioning, a `strictness` parameter
    (𝜎 ∈ [0, 1]) blends a standardized property vector (z ∈ ℝⁿ) with Gaussian noise
    (ε ~ 𝒩(0, I)), producing blended scores:


    .. math::

        b = \sigma \cdot z + (1 - \sigma) \cdot ε


    Samples are then sorted by `b` to assign them to partitions. When `strictness` is 0,
    partitioning is purely random (IID), while a value of 1 strictly follows the property ranking
    (strongly non-IID).

    Parameters
    ----------
    num_partitions : int
        Number of partitions to create.
    partition_by : str
        Name of the continuous feature to partition the dataset on.
    strictness : float
        Controls how strongly the feature influences partitioning (0 = iid, 1 = non-iid).
    shuffle : bool
        Whether to shuffle the indices within each partition (default: True).
    seed : Optional[int]
        Random seed for reproducibility. Used for initializing the random number generator (RNG),
        which affects the generation of the Gaussian noise (related to the `strictness` parameter)
        and dataset shuffling (if `shuffle` is True).


    Examples
    --------
    >>> from datasets import Dataset
    >>> import numpy as np
    >>> import pandas as pd
    >>> from flwr_datasets.partitioner import ContinuousPartitioner
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Create synthetic data
    >>> df = pd.DataFrame({
    >>>     "continuous": np.linspace(0, 10, 10_000),
    >>>     "category": np.random.choice([0, 1, 2, 3], size=10_000)
    >>> })
    >>> hf_dataset = Dataset.from_pandas(df)
    >>>
    >>> # Partition dataset
    >>> partitioner = ContinuousPartitioner(
    >>>     num_partitions=5,
    >>>     partition_by="continuous",
    >>>     strictness=0.7,
    >>>     shuffle=True
    >>> )
    >>> partitioner.dataset = hf_dataset
    >>>
    >>> # Plot partitions
    >>> plt.figure(figsize=(10, 6))
    >>> for i in range(5):
    >>>     plt.hist(
    >>>         partitioner.load_partition(i)["continuous"],
    >>>         bins=64,
    >>>         alpha=0.5,
    >>>         label=f"Partition {i}"
    >>>     )
    >>> plt.legend()
    >>> plt.xlabel("Continuous Value")
    >>> plt.ylabel("Frequency")
    >>> plt.title("Partition distributions")
    >>> plt.grid(True)
    >>> plt.show()
    """

    def __init__(
        self,
        num_partitions: int,
        partition_by: str,
        strictness: float,
        shuffle: bool = True,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        if not 0 <= strictness <= 1:
            raise ValueError("`strictness` must be between 0 and 1")
        if num_partitions <= 0:
            raise ValueError("`num_partitions` must be greater than 0")

        self._num_partitions = num_partitions
        self._partition_by = partition_by
        self._strictness = strictness
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Lazy initialization
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int) -> Dataset:
        """Load a single partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            The index that corresponds to the requested partition.

        Returns
        -------
        dataset_partition : Dataset
            A single dataset partition.
        """
        self._check_and_generate_partitions_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._check_and_generate_partitions_if_needed()
        return self._num_partitions

    @property
    def partition_id_to_indices(self) -> dict[int, list[int]]:
        """Mapping from partition ID to dataset indices."""
        self._check_and_generate_partitions_if_needed()
        return self._partition_id_to_indices

    def _check_and_generate_partitions_if_needed(self) -> None:
        """Lazy evaluation of the partitioning logic."""
        if self._partition_id_to_indices_determined:
            return

        if self._num_partitions > self.dataset.num_rows:
            raise ValueError(
                "Number of partitions must be less than or equal to number of dataset samples."
            )

        # Extract property values
        property_values = np.array(self.dataset[self._partition_by], dtype=np.float32)

        # Check for missing values (None or NaN)
        if np.any(property_values is None) or np.isnan(property_values).any():
            raise ValueError(
                f"The column '{self._partition_by}' contains None or NaN values, "
                f"which are not supported by {self.__class__.__qualname__}. "
                "Please clean or filter your dataset before partitioning."
            )

        # Standardize
        std = np.std(property_values)
        if std < 1e-6 and self._strictness > 0:
            raise ValueError(
                f"Cannot standardize column '{self._partition_by}' "
                f"because it has near-zero std (std={std}). "
                "All values are nearly identical, which prevents meaningful non-IID partitioning. "
                "To resolve this, choose a different partition property "
                "or set strictness to 0 to enable IID partitioning."
            )

        standardized_values = (property_values - np.mean(property_values)) / std

        # Blend noise
        noise = self._rng.normal(loc=0, scale=1, size=len(standardized_values))
        blended_values = (
            self._strictness * standardized_values + (1 - self._strictness) * noise
        )

        # Sort and partition
        sorted_indices = np.argsort(blended_values)
        partition_indices = np.array_split(sorted_indices, self._num_partitions)

        for pid, indices in enumerate(partition_indices):
            indices_list = indices.tolist()
            if self._shuffle:
                self._rng.shuffle(indices_list)
            self._partition_id_to_indices[pid] = indices_list

        self._partition_id_to_indices_determined = True
