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
"""Tests for ClientManager."""


from flwr.common import (
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    ReconnectIns,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy


class TestClientProxy(ClientProxy):
    """Minimal ClientProxy test double."""

    def get_properties(
        self,
        ins: GetPropertiesIns,
        timeout: float | None,
        group_id: int | None,
    ) -> GetPropertiesRes:
        """Not used in this test module."""
        raise NotImplementedError

    def get_parameters(
        self,
        ins: GetParametersIns,
        timeout: float | None,
        group_id: int | None,
    ) -> GetParametersRes:
        """Not used in this test module."""
        raise NotImplementedError

    def fit(
        self,
        ins: FitIns,
        timeout: float | None,
        group_id: int | None,
    ) -> FitRes:
        """Not used in this test module."""
        raise NotImplementedError

    def evaluate(
        self,
        ins: EvaluateIns,
        timeout: float | None,
        group_id: int | None,
    ) -> EvaluateRes:
        """Not used in this test module."""
        raise NotImplementedError

    def reconnect(
        self,
        ins: ReconnectIns,
        timeout: float | None,
        group_id: int | None,
    ) -> DisconnectRes:
        """Not used in this test module."""
        raise NotImplementedError


def test_simple_client_manager_register() -> None:
    """Tests if the register method works correctly."""
    # Prepare
    cid = "1"
    client = TestClientProxy(cid=cid)
    client_manager = SimpleClientManager()

    # Execute
    first = client_manager.register(client)
    second = client_manager.register(client)

    # Assert
    assert first
    assert not second
    assert len(client_manager) == 1


def test_simple_client_manager_unregister() -> None:
    """Tests if the unregister method works correctly."""
    # Prepare
    cid = "1"
    client = TestClientProxy(cid=cid)
    client_manager = SimpleClientManager()
    client_manager.register(client)

    # Execute
    client_manager.unregister(client)

    # Assert
    assert len(client_manager) == 0
