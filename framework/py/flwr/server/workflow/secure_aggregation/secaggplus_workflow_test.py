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
"""Tests for the SecAgg+ workflow."""


import pytest

from .secaggplus_workflow import SecAggPlusWorkflow


def test_valid_init_params() -> None:
    """Test that valid init parameters do not raise."""
    SecAggPlusWorkflow(num_shares=3, reconstruction_threshold=2)


@pytest.mark.parametrize("clipping_range", [0.0, -1.0])
def test_non_positive_clipping_range_raises(clipping_range: float) -> None:
    """Test that a non-positive `clipping_range` is rejected at construction.

    A `clipping_range` of 0 would otherwise surface later as an opaque
    ZeroDivisionError inside quantization (`target_range / (2 * clipping_range)`),
    and a negative value would silently produce reversed `np.clip` bounds.
    """
    with pytest.raises(ValueError, match="clipping_range"):
        SecAggPlusWorkflow(
            num_shares=3,
            reconstruction_threshold=2,
            clipping_range=clipping_range,
        )
