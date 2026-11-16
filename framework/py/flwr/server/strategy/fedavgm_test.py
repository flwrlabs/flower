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


from unittest.mock import MagicMock

from numpy import array, float32
from numpy.testing import assert_almost_equal

from flwr.common import (
    Code,
    FitRes,
    NDArrays,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from .fedavgm import FedAvgM


def test_aggregate_fit_using_near_one_server_lr_and_no_momentum() -> None:
    """Test aggregate with near-one learning rate and no momentum."""
    # Prepare
    weights0_0 = array([[1, 2, 3], [4, 5, 6]], dtype=float32)
    weights0_1 = array([7, 8, 9, 10], dtype=float32)
    weights1_0 = array([[1, 2, 3], [4, 5, 6]], dtype=float32)
    weights1_1 = array([7, 8, 9, 10], dtype=float32)

    initial_weights: NDArrays = [
        array([[0, 0, 0], [0, 0, 0]], dtype=float32),
        array([0, 0, 0, 0], dtype=float32),
    ]

    results: list[tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters([weights0_0, weights0_1]),
                num_examples=1,
                metrics={},
            ),
        ),
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters([weights1_0, weights1_1]),
                num_examples=2,
                metrics={},
            ),
        ),
    ]
    failures: list[tuple[ClientProxy, FitRes] | BaseException] = []
    expected: NDArrays = [
        array([[1, 2, 3], [4, 5, 6]], dtype=float32),
        array([7, 8, 9, 10], dtype=float32),
    ]

    strategy = FedAvgM(
        initial_parameters=ndarrays_to_parameters(initial_weights),
        server_learning_rate=1.0 + 1e-9,
    )

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual
    for w_act, w_exp in zip(parameters_to_ndarrays(actual), expected, strict=True):
        assert_almost_equal(w_act, w_exp)


def test_aggregate_fit_server_learning_rate_and_momentum() -> None:
    """Test aggregate with near-one learning rate and near-zero momentum."""
    # Prepare
    weights0_0 = array([[1, 2, 3], [4, 5, 6]], dtype=float32)
    weights0_1 = array([7, 8, 9, 10], dtype=float32)
    weights1_0 = array([[1, 2, 3], [4, 5, 6]], dtype=float32)
    weights1_1 = array([7, 8, 9, 10], dtype=float32)

    initial_weights: NDArrays = [
        array([[0, 0, 0], [0, 0, 0]], dtype=float32),
        array([0, 0, 0, 0], dtype=float32),
    ]

    results: list[tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters([weights0_0, weights0_1]),
                num_examples=1,
                metrics={},
            ),
        ),
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters([weights1_0, weights1_1]),
                num_examples=2,
                metrics={},
            ),
        ),
    ]
    failures: list[tuple[ClientProxy, FitRes] | BaseException] = []
    expected: NDArrays = [
        array([[1, 2, 3], [4, 5, 6]], dtype=float32),
        array([7, 8, 9, 10], dtype=float32),
    ]

    strategy = FedAvgM(
        initial_parameters=ndarrays_to_parameters(initial_weights),
        server_learning_rate=1.0 + 1e-9,
        server_momentum=1.0e-9,
    )

    # Execute
    # First round (activate momentum)
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Second round (update momentum)
    actual, _ = strategy.aggregate_fit(2, results, failures)

    # Assert
    assert actual
    for w_act, w_exp in zip(parameters_to_ndarrays(actual), expected, strict=True):
        assert_almost_equal(w_act, w_exp)
