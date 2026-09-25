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
"""FedMom tests."""

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

from .fedmom import FedMom


def test_aggregate_fit_using_one_server_lr_and_no_momentum() -> None:
    """Test aggregate with unit learning rate and no momentum."""
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

    strategy = FedMom(
        initial_parameters=ndarrays_to_parameters(initial_weights),
        server_learning_rate=1.0,
        server_momentum=0.0,
    )

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual
    for w_act, w_exp in zip(parameters_to_ndarrays(actual), expected, strict=True):
        assert_almost_equal(w_act, w_exp)


def test_aggregate_fit_server_learning_rate_and_momentum() -> None:
    """Test aggregate with learning rate and momentum over multiple rounds."""
    # Prepare
    weights0_0 = array([[1, 2, 3], [4, 5, 6]], dtype=float32)
    weights0_1 = array([7, 8, 9, 10], dtype=float32)

    initial_weights: NDArrays = [
        array([[0, 0, 0], [0, 0, 0]], dtype=float32),
        array([0, 0, 0, 0], dtype=float32),
    ]

    results_r1: list[tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters([weights0_0, weights0_1]),
                num_examples=1,
                metrics={},
            ),
        ),
    ]
    failures: list[tuple[ClientProxy, FitRes] | BaseException] = []

    # Round 1 expectation:
    # v1 = w0 - 1.0 * (w0 - w_avg1) = w_avg1
    # w1 = v1 + 0.5 * (v1 - w0) = 1.5 * w_avg1
    expected_r1: NDArrays = [
        array([[1.5, 3.0, 4.5], [6.0, 7.5, 9.0]], dtype=float32),
        array([10.5, 12.0, 13.5, 15.0], dtype=float32),
    ]

    strategy = FedMom(
        initial_parameters=ndarrays_to_parameters(initial_weights),
        server_learning_rate=1.0,
        server_momentum=0.5,
    )

    # Execute Round 1
    actual_r1, _ = strategy.aggregate_fit(1, results_r1, failures)

    assert actual_r1
    for w_act, w_exp in zip(
        parameters_to_ndarrays(actual_r1), expected_r1, strict=True
    ):
        assert_almost_equal(w_act, w_exp)

    # Round 2: client returns 2.0 * w_avg1
    weights1_0 = array([[2, 4, 6], [8, 10, 12]], dtype=float32)
    weights1_1 = array([14, 16, 18, 20], dtype=float32)

    results_r2: list[tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters([weights1_0, weights1_1]),
                num_examples=1,
                metrics={},
            ),
        ),
    ]

    # Round 2 expectation:
    # w1 = 1.5 * w_avg1
    # g1 = w1 - w_avg2 = 1.5 * w_avg1 - 2.0 * w_avg1 = -0.5 * w_avg1
    # v2 = w1 - 1.0 * g1 = 1.5 * w_avg1 - (-0.5 * w_avg1) = 2.0 * w_avg1
    # w2 = v2 + 0.5 * (v2 - v1) = 2.0 * w_avg1 + 0.5 * (2.0 * w_avg1 - 1.0 * w_avg1) = 2.5 * w_avg1
    expected_r2: NDArrays = [
        array([[2.5, 5.0, 7.5], [10.0, 12.5, 15.0]], dtype=float32),
        array([17.5, 20.0, 22.5, 25.0], dtype=float32),
    ]

    # Execute Round 2
    actual_r2, _ = strategy.aggregate_fit(2, results_r2, failures)

    assert actual_r2
    for w_act, w_exp in zip(
        parameters_to_ndarrays(actual_r2), expected_r2, strict=True
    ):
        assert_almost_equal(w_act, w_exp)


def test_configure_fit_records_initial_parameters() -> None:
    """Test that configure_fit captures client-provided initial parameters."""
    strategy = FedMom(server_learning_rate=1.0, server_momentum=0.0)
    assert strategy.initial_parameters is None

    initial_weights: NDArrays = [array([1.0, 2.0, 3.0], dtype=float32)]
    initial_params = ndarrays_to_parameters(initial_weights)

    client_manager = MagicMock()
    client_manager.num_available.return_value = 2
    client_manager.sample.return_value = []

    strategy.configure_fit(1, initial_params, client_manager)

    assert strategy.initial_parameters is initial_params
