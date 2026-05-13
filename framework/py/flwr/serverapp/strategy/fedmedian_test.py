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
"""FedMedian tests."""


import numpy as np

from flwr.common import ArrayRecord

from .fedmedian import FedMedian
from .strategy_utils_test import create_mock_reply


def test_aggregate_fit() -> None:
    """Tests if FedMedian is aggregating correctly."""
    # Prepare
    strategy = FedMedian()
    replies = [
        create_mock_reply(ArrayRecord([np.array([1.0, 0.2, 0.5, 1.0])]), 5),
        create_mock_reply(ArrayRecord([np.array([0.2, 0.5, 1.0, 0.2])]), 2),
        create_mock_reply(ArrayRecord([np.array([0.5, 1.0, 0.2, 0.5])]), 9),
    ]
    expected = np.array([0.5, 0.5, 0.5, 0.5])

    # Execute
    actual_aggregated, _ = strategy.aggregate_train(1, replies)

    # Assert
    assert actual_aggregated
    actual = actual_aggregated.to_numpy_ndarrays()[0]
    np.testing.assert_equal(actual, expected)


def test_aggregate_fit_with_scalar_weights() -> None:
    """Tests if FedMedian preserves 0-dim arrays."""
    strategy = FedMedian()
    replies = [
        create_mock_reply(ArrayRecord([np.array(1.0)]), 1),
        create_mock_reply(ArrayRecord([np.array(3.0)]), 1),
        create_mock_reply(ArrayRecord([np.array(2.0)]), 1),
    ]

    actual_aggregated, _ = strategy.aggregate_train(1, replies)

    assert actual_aggregated
    actual = actual_aggregated.to_numpy_ndarrays()[0]
    assert actual.shape == ()
    np.testing.assert_equal(actual, np.array(2.0))
