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
"""SquarePartitioner class."""

import numpy as np

from flwr_datasets.partitioner.id_to_size_fnc_partitioner import IdToSizeFncPartitioner


class SquarePartitioner(IdToSizeFncPartitioner):
    """Partitioner creates partitions of size that are correlated with squared id.

    The amount of data each client gets is correlated with the squared partition ID.
    For instance, if the IDs range from 1 to M, client with ID 1 gets 1 unit of data,
    client 2 gets 4 units, and so on, up to client M which gets M^2 units.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.

    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import SquarePartitioner
    >>>
    >>> partitioner = SquarePartitioner(num_partitions=10)
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    """

    def __init__(self, num_partitions: int) -> None:
        super().__init__(
            num_partitions=num_partitions, partition_id_to_size_fn=np.square
        )
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
