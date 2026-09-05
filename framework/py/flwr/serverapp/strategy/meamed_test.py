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
"""MeaMed tests."""


import numpy as np

from flwr.common import ArrayRecord

from .meamed import MeaMed
from .strategy_utils_test import create_mock_reply


def test_aggregate_fit() -> None:
    """Tests if MeaMed is aggregating correctly.

    With values [0.2, 1.0, 0.5], the median is 0.5.
    With num_closest=2, the 2 closest to 0.5 are 0.2 and 0.5.
    Their mean is 0.35.
    """
    # Prepare
    strategy = MeaMed(num_closest=2)
    replies = [
        create_mock_reply(ArrayRecord([np.array([0.2, 0.2, 0.2, 0.2])]), 5),
        create_mock_reply(ArrayRecord([np.array([1.0, 1.0, 1.0, 1.0])]), 2),
        create_mock_reply(ArrayRecord([np.array([0.5, 0.5, 0.5, 0.5])]), 9),
    ]
    expected = np.array([0.35, 0.35, 0.35, 0.35])

    # Execute
    actual_aggregated, _ = strategy.aggregate_train(1, replies)

    # Assert
    assert actual_aggregated
    actual = actual_aggregated.to_numpy_ndarrays()[0]
    np.testing.assert_allclose(actual, expected)


def test_aggregate_fit_all_closest() -> None:
    """Tests MeaMed with num_closest equal to number of clients (degenerates to mean)."""
    # Prepare
    strategy = MeaMed(num_closest=3)
    replies = [
        create_mock_reply(ArrayRecord([np.array([0.2, 0.2, 0.2, 0.2])]), 5),
        create_mock_reply(ArrayRecord([np.array([1.0, 1.0, 1.0, 1.0])]), 2),
        create_mock_reply(ArrayRecord([np.array([0.5, 0.5, 0.5, 0.5])]), 9),
    ]
    # Mean of all three: (0.2 + 1.0 + 0.5) / 3 ≈ 0.5667
    expected = np.array([0.5666667, 0.5666667, 0.5666667, 0.5666667])

    # Execute
    actual_aggregated, _ = strategy.aggregate_train(1, replies)

    # Assert
    assert actual_aggregated
    actual = actual_aggregated.to_numpy_ndarrays()[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


def test_aggregate_fit_with_scalar_weights() -> None:
    """Tests if MeaMed preserves 0-dim arrays."""
    strategy = MeaMed(num_closest=2)
    replies = [
        create_mock_reply(ArrayRecord([np.array(1.0)]), 1),
        create_mock_reply(ArrayRecord([np.array(3.0)]), 1),
        create_mock_reply(ArrayRecord([np.array(2.0)]), 1),
    ]
    # Median is 2.0, closest 2 are 1.0 and 2.0 (or 2.0 and 3.0), mean = 1.5 or 2.5
    # Actually: |1-2|=1, |3-2|=1, |2-2|=0 → closest 2 are 2.0 and one of {1.0, 3.0}

    actual_aggregated, _ = strategy.aggregate_train(1, replies)

    assert actual_aggregated
    actual = actual_aggregated.to_numpy_ndarrays()[0]
    assert actual.shape == ()
