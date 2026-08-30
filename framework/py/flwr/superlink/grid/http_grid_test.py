# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for grid SDK."""


import threading
import time
import unittest
from contextlib import AbstractContextManager
from typing import Any
from unittest.mock import Mock, patch

import httpx
from parameterized import parameterized

from flwr.app import RecordDict
from flwr.app.error import Error
from flwr.app.message import Message
from flwr.common.constant import SUPERLINK_NODE_ID, ErrorCode
from flwr.common.serde import message_to_proto
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    GetNodesRequest,
    PullAppMessagesRequest,
    PushAppMessagesRequest,
)
from flwr.supercore.constant import PULL_MAX_TIME, PULL_MAX_TRIES_PER_OBJECT
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_tree,
)
from flwr.supercore.interceptors import (
    RuntimeTokenHttpInterceptor,
    RuntimeVersionHttpInterceptor,
)
from flwr.supercore.run import Run

from .http_grid import HttpGrid

original_wait = threading.Event.wait


class TestHttpGrid(unittest.TestCase):
    """Tests for `HttpGrid` class."""

    def setUp(self) -> None:
        """Initialize a mock Runtime HTTP client and Grid before each test."""
        self.mock_client = Mock()
        self.mock_run = Run.create_empty(61016)
        self.mock_run.fab_id = "mock/mock"
        self.mock_run.fab_version = "v1.0.0"
        self.mock_run.fab_hash = "9f86d08"
        self.grid = HttpGrid(token="test-token")
        self.grid._client = self.mock_client  # pylint: disable=protected-access
        self.grid.set_run(self.mock_run)

    def test_init_http_grid(self) -> None:
        """Test Runtime HTTP client initialization."""
        # Assert
        self.assertEqual(self.grid.run.run_id, 61016)
        self.assertEqual(self.grid.run.fab_id, "mock/mock")
        self.assertEqual(self.grid.run.fab_version, "v1.0.0")
        self.assertEqual(self.grid.run.fab_hash, "9f86d08")
        self.assertEqual(self.grid._addr, "127.0.0.1:8000")  # pylint: disable=W0212

    def test_get_nodes(self) -> None:
        """Test retrieval of nodes."""
        # Prepare
        mock_response = Mock()
        mock_response.nodes = [Mock(node_id=404), Mock(node_id=200)]
        self.mock_client.GetNodes.return_value = mock_response

        # Execute
        node_ids = self.grid.get_node_ids()
        args, kwargs = self.mock_client.GetNodes.call_args

        # Assert
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], GetNodesRequest)
        self.assertEqual(node_ids, [404, 200])

    def _prep_message(self, message: Message) -> Message:
        # We need to be able to specify the actual object IDs
        # in the mocked responses, due to this we need to set
        # elements in the metadata that would be normally be
        # set when pushing a message.
        # pylint: disable-next=W0212
        message.metadata._run_id = 61016  # type: ignore
        # pylint: disable-next=W0212
        message.metadata._src_node_id = SUPERLINK_NODE_ID  # type: ignore
        message.metadata.__dict__["_message_id"] = message.object_id
        return message

    def test_push_messages_valid(self) -> None:
        """Test pushing valid messages."""
        # Prepare
        msg1 = self._prep_message(Message(RecordDict(), 0, "query.A"))
        msg2 = self._prep_message(Message(RecordDict(), 0, "query.B"))

        msgs = [msg1, msg2]
        # The seconds ObjectIDs doesn't contain the object ID of the empty RecordDict
        # because it is the same as the one in msg1.
        mock_response = Mock(
            message_ids=[msg1.object_id, msg2.object_id],
            objects_to_push=[msg1.object_id, RecordDict().object_id, msg2.object_id],
            session_id="",
        )
        self.mock_client.PushMessages.return_value = mock_response
        self.mock_client.PushObject.return_value = Mock(stored=True)

        # Execute
        msg_ids = self.grid.push_messages(msgs)
        args, kwargs = self.mock_client.PushMessages.call_args

        # Assert
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], PushAppMessagesRequest)
        self.assertEqual(msg_ids, [msg1.object_id, msg2.object_id])
        for message in args[0].messages_list:
            self.assertEqual(message.metadata.run_id, 61016)

    def test_pull_messages_with_given_message_ids(self) -> None:
        """Test pulling messages with specific message IDs."""
        # Prepare: Create instruction messages
        ins1 = self._prep_message(Message(RecordDict(), 123, "query"))
        ins2 = self._prep_message(Message(RecordDict(), 456, "query"))

        # Prepare: Create a normal reply
        ok_msg = Message(RecordDict(), reply_to=ins1)
        ok_msg.metadata.__dict__["_message_id"] = ok_msg.object_id
        ok_msg_all_objs = get_all_nested_objects(ok_msg)

        # Prepare: Create an error reply
        err_msg = Message(Error(0), reply_to=ins2)
        err_msg.metadata.__dict__["_message_id"] = err_msg.object_id
        err_msg_all_objs = get_all_nested_objects(err_msg)

        # Prepare: Mock the objectStore
        obj_store = {k: v.deflate() for k, v in ok_msg_all_objs.items()}
        obj_store.update({k: v.deflate() for k, v in err_msg_all_objs.items()})

        # Prepare: Mock the response of PushMessages
        self.mock_client.PullMessages.return_value = Mock(
            messages_list=[message_to_proto(ok_msg), message_to_proto(err_msg)],
            message_object_trees=[
                get_object_tree(ok_msg),
                get_object_tree(err_msg),
            ],
        )
        # Prepare: Mock response of PullObject
        self.mock_client.PullObject.side_effect = lambda req: Mock(
            object_found=True,
            object_available=True,
            object_content=obj_store[req.object_id],
        )

        # Execute
        msgs = list(self.grid.pull_messages([ins1.object_id, ins2.object_id]))
        args, kwargs = self.mock_client.PullMessages.call_args

        # Assert
        self.assertEqual(len(args), 1)
        self.assertEqual(len(kwargs), 0)
        self.assertIsInstance(args[0], PullAppMessagesRequest)
        self.assertEqual(args[0].message_ids, [ins1.object_id, ins2.object_id])
        self.assertEqual(msgs[0].metadata, ok_msg.metadata)
        self.assertEqual(msgs[0].content, ok_msg.content)
        self.assertEqual(msgs[1].metadata, err_msg.metadata)
        self.assertEqual(msgs[1].error, err_msg.error)
        self.assertEqual(self.mock_client.PullObject.call_count, len(obj_store))

    def test_send_and_receive_messages_complete(self) -> None:
        """Test send and receive all messages successfully."""
        # Prepare: Create an instruction message and mock responses
        msg = self._prep_message(Message(RecordDict(), 0, "query"))
        self.mock_client.PushMessages.return_value = Mock(
            message_ids=[msg.object_id],
            objects_to_push=[msg.object_id, RecordDict().object_id],
            session_id="",
        )
        self.mock_client.PushObject.return_value = Mock(stored=True)

        # Prepare: create an error reply message and mock responses
        reply = Message(Error(0), reply_to=msg)
        reply.metadata.__dict__["_message_id"] = reply.object_id
        self.mock_client.PullMessages.return_value = Mock(
            messages_list=[message_to_proto(reply)],
            message_object_trees=[get_object_tree(reply)],
        )
        self.mock_client.PullObject.return_value = Mock(
            object_found=True, object_available=True, object_content=reply.deflate()
        )

        # Execute
        ret_msgs = list(self.grid.send_and_receive([msg]))

        # Assert
        self.assertEqual(len(ret_msgs), 1)
        self.assertEqual(ret_msgs[0].metadata, reply.metadata)
        self.assertEqual(ret_msgs[0].error, reply.error)

    def test_send_and_receive_messages_timeout(self) -> None:
        """Test send and receive messages but time out."""
        # Prepare
        msg = self._prep_message(Message(RecordDict(), 0, "query"))
        sleep_fn = time.sleep
        mock_response = Mock(
            message_ids=[msg.object_id],
            objects_to_push=[msg.object_id, RecordDict().object_id],
            session_id="",
        )
        self.mock_client.PushMessages.return_value = mock_response
        self.mock_client.PushObject.return_value = Mock(stored=True)
        mock_response = Mock(messages_list=[], message_object_trees=[])
        self.mock_client.PullMessages.return_value = mock_response

        # Execute
        with patch("time.sleep", side_effect=lambda t: sleep_fn(t * 0.01)):
            start_time = time.time()
            ret_msgs = list(self.grid.send_and_receive([msg], timeout=0.15))

        # Assert
        self.assertLess(time.time() - start_time, 0.2)
        self.assertEqual(len(ret_msgs), 0)

    def test_del_with_initialized_grid(self) -> None:
        """Test cleanup behavior when Grid is initialized."""
        # Execute
        self.grid.close()

        # Assert
        self.mock_client.close.assert_called_once()

    def test_del_with_uninitialized_grid(self) -> None:
        """Test cleanup behavior when Grid is not initialized."""
        # Prepare
        self.grid._client = None  # pylint: disable=protected-access

        # Execute
        self.grid.close()

        # Assert
        self.mock_client.close.assert_not_called()

    def test_set_run_rejects_non_run_instance(self) -> None:
        """Test `set_run` rejects invalid input types."""
        with self.assertRaises(TypeError):
            self.grid.set_run(61016)  # type: ignore[arg-type]

    @patch("flwr.superlink.grid.http_grid.RuntimeHttpClient.from_server_address")
    def test_connect_adds_client_interceptors(
        self,
        from_server_address: Mock,
    ) -> None:
        """`_connect` should pass HTTP interceptors to the Runtime client."""
        from_server_address.return_value = Mock()
        grid = HttpGrid(token="test-token")

        grid._connect()  # pylint: disable=protected-access

        kwargs = from_server_address.call_args.kwargs
        interceptors = kwargs["interceptors"]
        self.assertIsNotNone(interceptors)
        assert interceptors is not None
        self.assertEqual(len(interceptors), 2)
        self.assertIsInstance(interceptors[0], RuntimeVersionHttpInterceptor)
        self.assertIsInstance(interceptors[1], RuntimeTokenHttpInterceptor)
        # pylint: disable-next=protected-access
        self.assertEqual(interceptors[0]._metadata.component_name, "flwr-serverapp")

    def test_init_rejects_empty_token(
        self,
    ) -> None:
        """`HttpGrid` should reject empty token values."""
        with self.assertRaises(ValueError):
            HttpGrid(token="")

    def test_simple_retry_mechanism_get_nodes(self) -> None:
        """Test retry mechanism with the get_node_ids method."""
        # Prepare
        http_exc = httpx.ConnectError("Connection failed")
        mock_get_nodes = Mock()
        mock_get_nodes.side_effect = [
            http_exc,
            Mock(nodes=[Mock(node_id=404)]),
        ]
        # Make pylint happy
        # pylint: disable=protected-access
        self.grid._client = Mock(
            GetNodes=lambda *args, **kwargs: self.grid._retry_invoker.invoke(
                mock_get_nodes, *args, **kwargs
            )
        )
        # pylint: enable=protected-access

        # Execute
        with patch("time.sleep", side_effect=lambda _: None):
            node_ids = self.grid.get_node_ids()

        # Assert
        self.assertIn(404, node_ids)

    def test_missing_object_creates_message_with_error(self) -> None:
        """Test that a missing object creates a message with an error."""
        ins = self._prep_message(Message(RecordDict(), 123, "query"))
        reply = Message(RecordDict(), reply_to=ins)
        reply.metadata.__dict__["_message_id"] = reply.object_id
        self.mock_client.PullMessages.return_value = Mock(
            messages_list=[message_to_proto(reply)],
            message_object_trees=[get_object_tree(reply)],
        )
        self.mock_client.PullObject.return_value = Mock(
            object_found=False, object_available=False
        )

        messages = list(self.grid.pull_messages([ins.object_id]))

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].error.code, ErrorCode.MESSAGE_UNAVAILABLE)
        self.assertEqual(messages[0].metadata.reply_to_message_id, ins.object_id)
        self.mock_client.ConfirmMessageReceived.assert_not_called()

    @parameterized.expand(  # type: ignore
        [
            (
                patch.object(  # make test hit PULL_MAX_TRIES_PER_OBJECT
                    threading.Event,
                    "wait",
                    new=lambda self, timeout=None: original_wait(
                        self, timeout * 1e-9 if timeout is not None else None
                    ),
                ),
            ),
            (  # make test hit PULL_MAX_TIME
                patch("time.monotonic", side_effect=[0, PULL_MAX_TIME + 1]),
            ),
        ]
    )
    def test_timeout_pulling_object_creates_message_with_error(
        self, patcher: AbstractContextManager[Any]
    ) -> None:
        """Test that pulling an object with a timeout creates a message with an
        error."""
        # Prepare: Create instruction message
        ins1 = self._prep_message(Message(RecordDict(), 123, "query"))

        # Prepare: Create a normal reply
        ok_msg = Message(RecordDict(), reply_to=ins1)
        ok_msg.metadata.__dict__["_message_id"] = ok_msg.object_id

        # Prepare: Mock the response of PushMessages
        self.mock_client.PullMessages.return_value = Mock(
            messages_list=[message_to_proto(ok_msg)],
            message_object_trees=[
                get_object_tree(ok_msg),
            ],
        )

        num_objects = len(get_all_nested_objects(ok_msg))

        # Prepare: Mock the response of PullObject to simulate timeout
        response = Mock(object_found=True, object_available=False, object_content=None)
        self.mock_client.PullObject.return_value = response

        # Execute
        with patcher:
            # Depending on the patch context, this will either hit the timeout or
            # the limit of pulling attempts for a given object
            msgs = list(self.grid.pull_messages([ins1.object_id]))

        # Assert: `msgs` should contain a single error message
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].has_content(), False)
        self.assertEqual(msgs[0].error.code, ErrorCode.MESSAGE_UNAVAILABLE)
        # Assert that PullObject was called PULL_MAX_TRIES_PER_OBJECT times for each
        # object at most. Note that because the message contains multiple objects,
        # we account for this in the assertion.
        self.assertLessEqual(
            self.mock_client.PullObject.call_count,
            PULL_MAX_TRIES_PER_OBJECT * num_objects,
        )
