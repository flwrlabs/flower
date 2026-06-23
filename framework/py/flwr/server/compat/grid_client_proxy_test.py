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
"""GridClientProxy tests."""


import unittest
import unittest.mock
from collections.abc import Callable, Iterable
from typing import Any, cast
from unittest.mock import Mock, patch

import numpy as np

import flwr
from flwr.app import Error, Message, RecordDict
from flwr.common import (
    Code,
    Config,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Properties,
    Status,
)
from flwr.compat.common import recorddict_compat as compat
from flwr.server.compat.grid_client_proxy import GridClientProxy

MESSAGE_PARAMETERS = Parameters(tensors=[b"abc"], tensor_type="np")

CLIENT_PROPERTIES = cast(Properties, {"tensor_type": "numpy.ndarray"})
CLIENT_STATUS = Status(code=Code.OK, message="OK")

ERROR_REPLY = Error(code=0, reason="mock error")

RUN_ID = 61016
NODE_ID = 1


class GridClientProxyTestCase(unittest.TestCase):
    """Tests for GridClientProxy."""

    def setUp(self) -> None:
        """Set up mocks for tests."""
        grid = Mock()
        grid.get_node_ids.return_value = [1]
        client = GridClientProxy(node_id=NODE_ID, grid=grid, run_id=61016)

        self.patcher = patch(
            "flwr.server.compat.grid_client_proxy.Message",
            side_effect=self._mock_message_init,
        )
        self.grid = grid
        self.client = client
        self.created_msg: Message | None = None
        self.called_times: int = 0
        self.patcher.start()

    def tearDown(self) -> None:
        """Tear down mocks."""
        self.patcher.stop()

    def test_get_properties(self) -> None:
        """Test positive case."""
        # Prepare
        res = GetPropertiesRes(status=CLIENT_STATUS, properties=CLIENT_PROPERTIES)
        self.grid.send_and_receive.side_effect = self._exec_send_and_receive(res)
        request_properties: Config = {"tensor_type": "str"}
        ins = GetPropertiesIns(config=request_properties)

        # Execute
        value = self.client.get_properties(ins, timeout=None, group_id=0)

        # Assert
        self._common_assertions(ins)
        self.assertEqual(value.properties["tensor_type"], "numpy.ndarray")

    def test_get_parameters(self) -> None:
        """Test positive case."""
        # Prepare
        res = GetParametersRes(
            status=CLIENT_STATUS,
            parameters=MESSAGE_PARAMETERS,
        )
        self.grid.send_and_receive.side_effect = self._exec_send_and_receive(res)
        ins = GetParametersIns(config={})

        # Execute
        value = self.client.get_parameters(ins, timeout=None, group_id=0)

        # Assert
        self._common_assertions(ins)
        self.assertEqual(value, res)

    def test_fit(self) -> None:
        """Test positive case."""
        # Prepare
        res = FitRes(
            status=CLIENT_STATUS,
            parameters=MESSAGE_PARAMETERS,
            num_examples=10,
            metrics={},
        )
        self.grid.send_and_receive.side_effect = self._exec_send_and_receive(res)
        parameters = flwr.common.ndarrays_to_parameters([np.ones((2, 2))])
        ins = FitIns(parameters, {})

        # Execute
        value = self.client.fit(ins=ins, timeout=None, group_id=0)

        # Assert
        self._common_assertions(ins)
        self.assertEqual(value, res)

    def test_evaluate(self) -> None:
        """Test positive case."""
        # Prepare
        res = EvaluateRes(
            status=CLIENT_STATUS,
            loss=0.0,
            num_examples=0,
            metrics={},
        )
        self.grid.send_and_receive.side_effect = self._exec_send_and_receive(res)
        parameters = Parameters(tensors=[b"random params%^&*F"], tensor_type="np")
        ins = EvaluateIns(parameters, {})

        # Execute
        value = self.client.evaluate(ins, timeout=None, group_id=0)

        # Assert
        self._common_assertions(ins)
        self.assertEqual(value, res)

    def test_get_properties_and_fail(self) -> None:
        """Test negative case."""
        # Prepare
        self.grid.send_and_receive.side_effect = self._exec_send_and_receive(
            None, error_reply=True
        )
        request_properties: Config = {"tensor_type": "str"}
        ins = GetPropertiesIns(config=request_properties)

        # Execute and assert
        self.assertRaises(
            ValueError, self.client.get_properties, ins, timeout=None, group_id=0
        )
        self._common_assertions(ins)

    def test_get_parameters_and_fail(self) -> None:
        """Test negative case."""
        # Prepare
        self.grid.send_and_receive.side_effect = self._exec_send_and_receive(
            None, error_reply=True
        )
        ins = GetParametersIns(config={})

        # Execute and assert
        self.assertRaises(
            ValueError, self.client.get_parameters, ins, timeout=None, group_id=0
        )
        self._common_assertions(ins)

    def test_fit_and_fail(self) -> None:
        """Test negative case."""
        # Prepare
        self.grid.send_and_receive.side_effect = self._exec_send_and_receive(
            None, error_reply=True
        )
        parameters = flwr.common.ndarrays_to_parameters([np.ones((2, 2))])
        ins = FitIns(parameters, {})

        # Execute and assert
        self.assertRaises(ValueError, self.client.fit, ins, timeout=None, group_id=0)
        self._common_assertions(ins)

    def test_evaluate_and_fail(self) -> None:
        """Test negative case."""
        # Prepare
        self.grid.send_and_receive.side_effect = self._exec_send_and_receive(
            None, error_reply=True
        )
        parameters = Parameters(tensors=[b"random params%^&*F"], tensor_type="np")
        ins = EvaluateIns(parameters, {})

        # Execute and assert
        self.assertRaises(
            ValueError, self.client.evaluate, ins, timeout=None, group_id=0
        )
        self._common_assertions(ins)

    def _mock_message_init(  # pylint: disable=R0913,too-many-positional-arguments
        self,
        content: RecordDict,
        dst_node_id: int,
        message_type: str,
        ttl: float | None = None,
        group_id: str | None = None,
    ) -> Message:
        """Create a new message.

        This is a method for the Mock object.
        """
        self.called_times += 1
        self.created_msg = Message(
            content, dst_node_id, message_type, ttl=ttl, group_id=group_id
        )
        return self.created_msg

    def _exec_send_and_receive(
        self,
        res: GetParametersRes | GetPropertiesRes | FitRes | EvaluateRes | None,
        error_reply: bool = False,
    ) -> Callable[[Iterable[Message]], Iterable[Message]]:
        """Get the generate_replies function that sets the return value of grid's
        send_and_receive when called."""

        def generate_replies(messages: Iterable[Message]) -> Iterable[Message]:
            msg = list(messages)[0]
            recorddict = None
            if error_reply:
                pass
            elif isinstance(res, GetParametersRes):
                recorddict = compat.getparametersres_to_recorddict(res, True)
            elif isinstance(res, GetPropertiesRes):
                recorddict = compat.getpropertiesres_to_recorddict(res)
            elif isinstance(res, FitRes):
                recorddict = compat.fitres_to_recorddict(res, True)
            elif isinstance(res, EvaluateRes):
                recorddict = compat.evaluateres_to_recorddict(res)

            if recorddict is not None:
                ret = Message(recorddict, reply_to=msg)
            else:
                ret = Message(ERROR_REPLY, reply_to=msg)

            # Reply messages given the push message
            return [ret]

        return generate_replies

    def _common_assertions(self, original_ins: Any) -> None:
        """Check common assertions."""
        # Check if the created message contains the orignal *Ins
        assert self.created_msg is not None
        actual_ins = {  # type: ignore
            GetPropertiesIns: compat.recorddict_to_getpropertiesins,
            GetParametersIns: compat.recorddict_to_getparametersins,
            FitIns: (lambda x: compat.recorddict_to_fitins(x, True)),
            EvaluateIns: (lambda x: compat.recorddict_to_evaluateins(x, True)),
        }[type(original_ins)](self.created_msg.content)
        self.assertEqual(self.called_times, 1)
        self.assertEqual(actual_ins, original_ins)

        # Check if send_and_receive is called once with expected args/kwargs.
        self.grid.send_and_receive.assert_called_once()
        try:
            self.grid.send_and_receive.assert_any_call([self.created_msg])
        except AssertionError:
            self.grid.send_and_receive.assert_any_call(messages=[self.created_msg])
