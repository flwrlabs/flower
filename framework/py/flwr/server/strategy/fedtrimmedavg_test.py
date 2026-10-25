# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""FedTrimmedAvg tests."""

import pytest

from .fedtrimmedavg import FedTrimmedAvg


@pytest.mark.parametrize("beta", [-0.1, 0.5])
def test_init_rejects_invalid_beta(beta: float) -> None:
    """Test that invalid trimming fractions are rejected during setup."""
    with pytest.raises(ValueError, match=r"\[0, 0\.5\)"):
        FedTrimmedAvg(beta=beta)
