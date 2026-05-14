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
"""Flower Datasets main package."""

from flwr_datasets import metrics, partitioner, preprocessor
from flwr_datasets import utils as utils
from flwr_datasets import visualization
from flwr_datasets.common.version import package_version as _package_version
from flwr_datasets.federated_dataset import FederatedDataset

__all__ = [
    "FederatedDataset",
    "metrics",
    "partitioner",
    "preprocessor",
    "utils",
    "visualization",
]


__version__ = _package_version
