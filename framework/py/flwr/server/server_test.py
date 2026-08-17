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
"""Flower server tests."""


import numpy as np
import pytest
from unittest.mock import Mock

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    ReconnectIns,
    Scalar,
    Status,
    ndarray_to_bytes,
)
from flwr.common.fl_event import (
    FL_ROUND_COMPLETED,
    FL_ROUND_EVALUATE_COMPLETED,
    FL_ROUND_EVALUATE_STARTED,
    FL_ROUND_FAILED,
    FL_ROUND_FIT_COMPLETED,
    FL_ROUND_FIT_FAILED,
    FL_ROUND_FIT_STARTED,
    FL_ROUND_STARTED,
    FL_RUN_COMPLETED,
    FL_RUN_FAILED,
    FL_RUN_STARTED,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy import FedAvg

from .client_proxy import ClientProxy
from .server import Server, evaluate_clients, fit_clients


class SuccessClient(ClientProxy):
    """Test class."""

    def get_properties(
        self, ins: GetPropertiesIns, timeout: float | None, group_id: int | None
    ) -> GetPropertiesRes:
        """Raise an error because this method is not expected to be called."""
        raise NotImplementedError()

    def get_parameters(
        self, ins: GetParametersIns, timeout: float | None, group_id: int | None
    ) -> GetParametersRes:
        """Raise a error because this method is not expected to be called."""
        raise NotImplementedError()

    def fit(self, ins: FitIns, timeout: float | None, group_id: int | None) -> FitRes:
        """Simulate fit by returning a success FitRes with simple set of weights."""
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        arr_serialized = ndarray_to_bytes(arr)
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=[arr_serialized], tensor_type=""),
            num_examples=1,
            metrics={},
        )

    def evaluate(
        self, ins: EvaluateIns, timeout: float | None, group_id: int | None
    ) -> EvaluateRes:
        """Simulate evaluate by returning a success EvaluateRes with loss
        1.0."""
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=1.0,
            num_examples=1,
            metrics={},
        )

    def reconnect(
        self, ins: ReconnectIns, timeout: float | None, group_id: int | None
    ) -> DisconnectRes:
        """Simulate reconnect by returning a DisconnectRes with UNKNOWN reason."""
        return DisconnectRes(reason="UNKNOWN")


class FailingClient(ClientProxy):
    """Test class."""

    def get_properties(
        self, ins: GetPropertiesIns, timeout: float | None, group_id: int | None
    ) -> GetPropertiesRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()

    def get_parameters(
        self, ins: GetParametersIns, timeout: float | None, group_id: int | None
    ) -> GetParametersRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()

    def fit(self, ins: FitIns, timeout: float | None, group_id: int | None) -> FitRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()

    def evaluate(
        self, ins: EvaluateIns, timeout: float | None, group_id: int | None
    ) -> EvaluateRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()

    def reconnect(
        self, ins: ReconnectIns, timeout: float | None, group_id: int | None
    ) -> DisconnectRes:
        """Raise a NotImplementedError to simulate failure in the client."""
        raise NotImplementedError()


class EventClient(SuccessClient):
    """Client that supports parameter initialization for event tests."""

    def get_parameters(
        self, ins: GetParametersIns, timeout: float | None, group_id: int | None
    ) -> GetParametersRes:
        """Return empty parameters."""
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
        )


class FailingFitClient(EventClient):
    """Client that fails during fit for event tests."""

    def fit(self, ins: FitIns, timeout: float | None, group_id: int | None) -> FitRes:
        """Raise an exception to simulate a fit failure."""
        raise RuntimeError("fit failed")


class FailingAggregateStrategy(FedAvg):
    """Strategy that raises during aggregation for event tests."""

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Raise an exception to simulate an aggregation failure."""
        raise RuntimeError("aggregate failed")


def test_fit_emits_lifecycle_events() -> None:
    """Test that ``Server.fit`` emits the expected lifecycle events."""
    # Prepare
    client_manager = SimpleClientManager()
    client_manager.register(EventClient("1"))
    callback = Mock()
    server = Server(
        client_manager=client_manager,
        strategy=FedAvg(
            min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1
        ),
        event_callback=callback,
    )

    # Execute
    history, _ = server.fit(num_rounds=1, timeout=None)

    # Assert
    assert history is not None
    events = [call.args[0].event for call in callback.call_args_list]
    assert events == [
        FL_RUN_STARTED,
        FL_ROUND_STARTED,
        FL_ROUND_FIT_STARTED,
        FL_ROUND_FIT_COMPLETED,
        FL_ROUND_EVALUATE_STARTED,
        FL_ROUND_EVALUATE_COMPLETED,
        FL_ROUND_COMPLETED,
        FL_RUN_COMPLETED,
    ]


def test_fit_emits_failed_events_on_exception() -> None:
    """Test that ``Server.fit`` emits failed events when training raises."""
    # Prepare
    client_manager = SimpleClientManager()
    client_manager.register(EventClient("1"))
    callback = Mock()
    server = Server(
        client_manager=client_manager,
        strategy=FailingAggregateStrategy(
            min_fit_clients=1, min_evaluate_clients=1, min_available_clients=1
        ),
        event_callback=callback,
    )

    # Execute and assert
    with pytest.raises(RuntimeError, match="aggregate failed"):
        server.fit(num_rounds=1, timeout=None)

    events = [call.args[0].event for call in callback.call_args_list]
    assert events == [
        FL_RUN_STARTED,
        FL_ROUND_STARTED,
        FL_ROUND_FIT_STARTED,
        FL_ROUND_FIT_FAILED,
        FL_ROUND_FAILED,
        FL_RUN_FAILED,
    ]


def test_fit_clients() -> None:
    """Test fit_clients."""
    # Prepare
    clients: list[ClientProxy] = [
        FailingClient("0"),
        SuccessClient("1"),
    ]
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    arr_serialized = ndarray_to_bytes(arr)
    ins: FitIns = FitIns(Parameters(tensors=[arr_serialized], tensor_type=""), {})
    client_instructions = [(c, ins) for c in clients]

    # Execute
    results, failures = fit_clients(client_instructions, None, None, 0)

    # Assert
    assert len(results) == 1
    assert len(failures) == 1
    assert results[0][1].num_examples == 1


def test_eval_clients() -> None:
    """Test eval_clients."""
    # Prepare
    clients: list[ClientProxy] = [
        FailingClient("0"),
        SuccessClient("1"),
    ]
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    arr_serialized = ndarray_to_bytes(arr)
    ins: EvaluateIns = EvaluateIns(
        Parameters(tensors=[arr_serialized], tensor_type=""),
        {},
    )
    client_instructions = [(c, ins) for c in clients]

    # Execute
    results, failures = evaluate_clients(
        client_instructions=client_instructions,
        max_workers=None,
        timeout=None,
        group_id=0,
    )

    # Assert
    assert len(results) == 1
    assert len(failures) == 1
    assert results[0][1].loss == 1.0
    assert results[0][1].num_examples == 1


def test_set_max_workers() -> None:
    """Test eval_clients."""
    # Prepare
    server = Server(client_manager=SimpleClientManager())

    # Execute
    server.set_max_workers(42)

    # Assert
    assert server.max_workers == 42
