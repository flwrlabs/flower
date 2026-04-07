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
"""FedAvgM tests."""


import numpy as np

from flwr.common import ArrayRecord, Message, NDArrays

from .fedavgm import FedAvgM
from .strategy_utils_test import create_mock_reply


def _prepare_strategy() -> tuple[FedAvgM, list[Message], NDArrays]:
    """Prepare test.

    Returns
    -------
    tuple[FedAvgM, list[Message], NDArrays]
        A tuple of (strategy, replies, expected_weights_after_aggregation)
    """
    # Prepare: Mock replies from two clients
    weights0_0 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    weights0_1 = np.array([7, 8, 9, 10], dtype=np.float32)
    weights1_0 = np.array([[29, 23, 19], [17, 13, 11]], dtype=np.float32)
    weights1_1 = np.array([7, 5, 3, 2], dtype=np.float32)
    replies = [
        create_mock_reply(ArrayRecord([weights0_0, weights0_1]), num_examples=1),
        create_mock_reply(ArrayRecord([weights1_0, weights1_1]), num_examples=2),
    ]

    # Prepare: Compute expected weights after aggregation
    expected = [
        (weights0_0 * 1 + weights1_0 * 2) / 3,
        (weights0_1 * 1 + weights1_1 * 2) / 3,
    ]

    # Prepare: Create strategy and set initial weights
    initial_weights = [
        np.array([[0, 0, 0], [0, 0, 0]], dtype=np.float32),
        np.array([0, 0, 0, 0], dtype=np.float32),
    ]
    strategy = FedAvgM()
    strategy.current_arrays = ArrayRecord(initial_weights)
    return strategy, replies, expected


def test_aggregate_fit_using_near_one_server_lr_and_no_momentum() -> None:
    """Test aggregate with near-one learning rate and no momentum."""
    # Prepare
    strategy, replies, expected = _prepare_strategy()
    strategy.server_learning_rate = 1.0 + 1e-9

    # Execute
    actual, _ = strategy.aggregate_train(1, replies)

    # Assert
    assert actual is not None
    for w_act, w_exp in zip(actual.to_numpy_ndarrays(), expected, strict=True):
        np.testing.assert_almost_equal(w_act, w_exp, decimal=5)


def test_aggregate_fit_server_learning_rate_and_momentum() -> None:
    """Test aggregate with near-one learning rate and near-zero momentum."""
    # Prepare
    strategy, replies, expected = _prepare_strategy()
    strategy.server_learning_rate = 1.0 + 1e-9
    strategy.server_momentum = 1e-9

    # Execute: First round (activate momentum)
    actual, _ = strategy.aggregate_train(1, replies)

    # Execute: Second round (update momentum)
    actual, _ = strategy.aggregate_train(2, replies)

    # Assert
    assert actual is not None
    for w_act, w_exp in zip(actual.to_numpy_ndarrays(), expected, strict=True):
        np.testing.assert_almost_equal(w_act, w_exp, decimal=5)


def test_aggregate_fit_with_scalar_weights() -> None:
    """Test aggregate with scalar-shaped weights."""
    strategy = FedAvgM(server_learning_rate=1.0 + 1e-9)
    strategy.current_arrays = ArrayRecord([np.array(0.0)])
    replies = [
        create_mock_reply(ArrayRecord([np.array(1.0)]), num_examples=1),
        create_mock_reply(ArrayRecord([np.array(3.0)]), num_examples=2),
    ]

    actual, _ = strategy.aggregate_train(1, replies)

    assert actual is not None
    scalar_weight = actual.to_numpy_ndarrays()[0]
    assert scalar_weight.shape == ()
    np.testing.assert_allclose(scalar_weight, np.array(7.0 / 3.0), rtol=1e-6)
