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
"""ExponentialPartitioner class."""

import numpy as np

from flwr_datasets.partitioner.id_to_size_fnc_partitioner import IdToSizeFncPartitioner


class ExponentialPartitioner(IdToSizeFncPartitioner):
    """Partitioner creates partitions of size that are correlated with exp(id).

    The amount of data each client gets is correlated with the exponent of partition ID.
    For instance, if the IDs range from 1 to M, client with ID 1 gets e units of
    data, client 2 gets e^2 units, and so on, up to client M which gets e^M units.
    The floor operation is applied on each of these numbers, it means floor(2.71...)
    = 2; e^2 ~ 7.39 floor(7.39) = 7. The number is rounded down = the fraction is
    always cut. The remainders of theses unassigned (fraction) samples is added to the
    biggest partition (the one with the biggest partition_id).

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.

    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import ExponentialPartitioner
    >>>
    >>> partitioner = ExponentialPartitioner(num_partitions=10)
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    """

    def __init__(self, num_partitions: int) -> None:
        super().__init__(num_partitions=num_partitions, partition_id_to_size_fn=np.exp)
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
