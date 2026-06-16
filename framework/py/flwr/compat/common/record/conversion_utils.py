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
"""Deprecated conversion utility functions for Records."""


from flwr.app.message import Array
from flwr.common import NDArray
from flwr.common.logger import warn_deprecated_feature

WARN_DEPRECATED_MESSAGE = (
    "`array_from_numpy` is deprecated. Instead, use the `Array(ndarray)` class "
    "directly or `Array.from_numpy_ndarray(ndarray)`."
)


def array_from_numpy(ndarray: NDArray) -> Array:
    """Create Array from NumPy ndarray."""
    warn_deprecated_feature(WARN_DEPRECATED_MESSAGE)
    return Array.from_numpy_ndarray(ndarray)
