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
"""Tests for networked Flower client implementation."""


import unittest
from unittest.mock import MagicMock

import numpy as np

import flwr
from flwr.common import Config, GetParametersIns
from flwr.proto.transport_pb2 import (  # pylint: disable=E0611
    ClientMessage,
    Parameters,
    Scalar,
)
from flwr.server.superlink.fleet.grpc_bidi.grpc_bridge import ResWrapper
from flwr.server.superlink.fleet.grpc_bidi.grpc_client_proxy import GrpcClientProxy

MESSAGE_PARAMETERS = Parameters(tensors=[], tensor_type="np")
MESSAGE_FIT_RES = ClientMessage(
    fit_res=ClientMessage.FitRes(
        parameters=MESSAGE_PARAMETERS,
        num_examples=10,
    )
)
CLIENT_PROPERTIES = {"tensor_type": Scalar(string="numpy.ndarray")}
MESSAGE_PROPERTIES_RES = ClientMessage(
    get_properties_res=ClientMessage.GetPropertiesRes(properties=CLIENT_PROPERTIES)
)

RES_WRAPPER_FIT_RES = ResWrapper(client_message=MESSAGE_FIT_RES)
RES_WRAPPER_PROPERTIES_RES = ResWrapper(client_message=MESSAGE_PROPERTIES_RES)


class GrpcClientProxyTestCase(unittest.TestCase):
    """Tests for GrpcClientProxy."""

    def setUp(self) -> None:
        """Set up mocks for tests."""
        self.bridge_mock = MagicMock()
        # Set return_value for usually blocking get_client_message method
        self.bridge_mock.request.return_value = RES_WRAPPER_FIT_RES
        # Set return_value for get_properties
        self.bridge_mock_get_proprieties = MagicMock()
        self.bridge_mock_get_proprieties.request.return_value = (
            RES_WRAPPER_PROPERTIES_RES
        )

    def test_get_parameters(self) -> None:
        """Test the get_parameters method of the client class.

        Note that this test is currently quite simple and should be improved.
        """
        # Prepare
        client = GrpcClientProxy(cid="1", bridge=self.bridge_mock)
        get_parameters_ins = GetParametersIns(config={})

        # Execute
        value: flwr.common.GetParametersRes = client.get_parameters(
            ins=get_parameters_ins, timeout=None, group_id=0
        )

        # Assert
        assert not value.parameters.tensors

    def test_fit(self) -> None:
        """Test the fit method of the client class.

        Note that this test is currently quite simple and should be improved.
        """
        # Prepare
        client = GrpcClientProxy(cid="1", bridge=self.bridge_mock)
        parameters = flwr.common.ndarrays_to_parameters([np.ones((2, 2))])
        ins: flwr.common.FitIns = flwr.common.FitIns(parameters, {})

        # Execute
        fit_res = client.fit(ins=ins, timeout=None, group_id=0)

        # Assert
        assert fit_res.parameters.tensor_type == "np"
        assert flwr.common.parameters_to_ndarrays(fit_res.parameters) == []
        assert fit_res.num_examples == 10

    def test_evaluate(self) -> None:
        """Test the evaluate method of the client class.

        Note that this test is currently quite simple and should be improved.
        """
        # Prepare
        client = GrpcClientProxy(cid="1", bridge=self.bridge_mock)
        parameters = flwr.common.Parameters(tensors=[], tensor_type="np")
        evaluate_ins: flwr.common.EvaluateIns = flwr.common.EvaluateIns(parameters, {})

        # Execute
        evaluate_res = client.evaluate(evaluate_ins, timeout=None, group_id=1)

        # Assert
        assert (0, 0.0) == (
            evaluate_res.num_examples,
            evaluate_res.loss,
        )

    def test_get_properties(self) -> None:
        """Test the get_properties method of the client class.

        Note that this test is currently quite simple and should be improved.
        """
        # Prepare
        client = GrpcClientProxy(cid="1", bridge=self.bridge_mock_get_proprieties)
        request_properties: Config = {"tensor_type": "str"}
        ins: flwr.common.GetPropertiesIns = flwr.common.GetPropertiesIns(
            config=request_properties
        )

        # Execute
        value: flwr.common.GetPropertiesRes = client.get_properties(
            ins, timeout=None, group_id=0
        )

        # Assert
        assert value.properties["tensor_type"] == "numpy.ndarray"
