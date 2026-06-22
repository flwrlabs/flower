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
"""FedAvg tests."""


from unittest.mock import MagicMock

import numpy as np
from numpy.testing import assert_allclose

from flwr.common import (
    Code,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from .fedavg import FedAvg


def test_fedavg_num_fit_clients_20_available() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = FedAvg()
    expected = 20

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=20)

    # Assert
    assert expected == actual


def test_fedavg_num_fit_clients_19_available() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = FedAvg()
    expected = 19

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=19)

    # Assert
    assert expected == actual


def test_fedavg_num_fit_clients_10_available() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = FedAvg()
    expected = 10

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=10)

    # Assert
    assert expected == actual


def test_fedavg_num_fit_clients_minimum() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = FedAvg()
    expected = 9

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=9)

    # Assert
    assert expected == actual


def test_fedavg_num_evaluation_clients_40_available() -> None:
    """Test num_evaluation_clients function."""
    # Prepare
    strategy = FedAvg(fraction_evaluate=0.05)
    expected = 2

    # Execute
    actual, _ = strategy.num_evaluation_clients(num_available_clients=40)

    # Assert
    assert expected == actual


def test_fedavg_num_evaluation_clients_39_available() -> None:
    """Test num_evaluation_clients function."""
    # Prepare
    strategy = FedAvg(fraction_evaluate=0.05)
    expected = 2

    # Execute
    actual, _ = strategy.num_evaluation_clients(num_available_clients=39)

    # Assert
    assert expected == actual


def test_fedavg_num_evaluation_clients_20_available() -> None:
    """Test num_evaluation_clients function."""
    # Prepare
    strategy = FedAvg(fraction_evaluate=0.05)
    expected = 2

    # Execute
    actual, _ = strategy.num_evaluation_clients(num_available_clients=20)

    # Assert
    assert expected == actual


def test_fedavg_num_evaluation_clients_minimum() -> None:
    """Test num_evaluation_clients function."""
    # Prepare
    strategy = FedAvg(fraction_evaluate=0.05)
    expected = 2

    # Execute
    actual, _ = strategy.num_evaluation_clients(num_available_clients=19)

    # Assert
    assert expected == actual


def test_inplace_aggregate_fit_equivalence() -> None:
    """Test aggregate_fit equivalence between FedAvg and its inplace version."""
    # Prepare
    weights0_0 = np.random.randn(100, 64)
    weights0_1 = np.random.randn(314, 628, 3)
    weights1_0 = np.random.randn(100, 64)
    weights1_1 = np.random.randn(314, 628, 3)

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
                num_examples=5,
                metrics={},
            ),
        ),
    ]
    failures: list[tuple[ClientProxy, FitRes] | BaseException] = []

    fedavg_reference = FedAvg(inplace=False)
    fedavg_inplace = FedAvg()

    # Execute
    reference, _ = fedavg_reference.aggregate_fit(1, results, failures)
    assert reference
    inplace, _ = fedavg_inplace.aggregate_fit(1, results, failures)
    assert inplace

    # Convert to NumPy to check similarity
    reference_np = parameters_to_ndarrays(reference)
    inplace_np = parameters_to_ndarrays(inplace)

    # Assert
    for ref, inp in zip(reference_np, inplace_np, strict=True):
        assert_allclose(ref, inp)
