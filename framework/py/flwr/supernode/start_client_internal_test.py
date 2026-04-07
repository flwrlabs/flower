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
"""Tests for the Main Loop of Flower SuperNode."""


import unittest
from unittest.mock import Mock, patch

from flwr.common import ConfigRecord, Context, Message, RecordDict
from flwr.common.message import remove_content_from_message
from flwr.common.typing import Fab
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_tree,
)

from .start_client_internal import FAB_VERIFICATION_ERROR, _pull_and_store_message


class TestStartClientInternal(unittest.TestCase):  # pylint: disable=R0902
    """Test cases for the main loop."""

    def setUp(self) -> None:
        """Set up the test case."""
        self.mock_state = Mock()
        self.node_id = 111
        self.run_id = 110
        self.mock_state.get_node_id.return_value = self.node_id
        self.mock_object_store = Mock()
        self.mock_receive = Mock()
        self.mock_get_run = Mock()
        self.mock_get_fab = Mock()
        self.mock_push_object = Mock()
        self.mock_pull_object = Mock()
        self.mock_confirm_message_received = Mock()
        self.simple_store: dict[str, bytes] = {}

    def test_pull_and_store_message_no_message(self) -> None:
        """Test that no message is pulled when there are no messages."""
        # Prepare
        self.mock_receive.return_value = None

        # Execute
        res = _pull_and_store_message(
            state=self.mock_state,
            object_store=self.mock_object_store,
            node_config={},  # No need for this test
            receive=self.mock_receive,
            get_run=self.mock_get_run,
            get_fab=self.mock_get_fab,
            pull_object=self.mock_pull_object,
            confirm_message_received=self.mock_confirm_message_received,
            trusted_entities={},
        )

        # Assert
        assert res is None
        self.mock_receive.assert_called_once()
        self.mock_get_run.assert_not_called()
        self.mock_get_fab.assert_not_called()
        self.mock_pull_object.assert_not_called()
        self.mock_confirm_message_received.assert_not_called()

    def _prepare_for_pull_and_store_message(self) -> None:
        """Prepare mocks for pull_and_store_message."""
        # Prepare
        message = Message(
            content=RecordDict({"mock_cfg": ConfigRecord({"key": "value"})}),
            dst_node_id=self.node_id,
            message_type="query",
            group_id="test_group",
        )
        message.metadata.__dict__["_run_id"] = self.run_id
        message.metadata.__dict__["_message_id"] = message.object_id
        message_without_content = remove_content_from_message(message)
        self.mock_receive.return_value = (
            message_without_content,
            get_object_tree(message),
        )

        # Prepare: Mock pull object function
        store = {
            obj_id: obj.deflate()
            for obj_id, obj in get_all_nested_objects(message).items()
        }
        self.mock_pull_object.side_effect = lambda _, obj_id: store[obj_id]

        # Prepare: Mock object store
        self.mock_object_store.preregister.return_value = list(store.keys())
        self.simple_store = store

    def _assert_message_pulled_and_stored(self) -> None:
        """Assert that the message was pulled and stored correctly."""
        message_without_content = self.mock_receive.return_value[0]

        # Assert: the message should be pulled and stored
        self.mock_receive.assert_called_once()
        self.mock_confirm_message_received.assert_called_once()
        self.mock_state.get_run.assert_called_once()
        self.mock_state.store_message.assert_called_once_with(message_without_content)

        # Assert: All objects should be pulled and stored
        for obj_id, obj_content in self.simple_store.items():
            self.mock_pull_object.assert_any_call(self.run_id, obj_id)
            self.mock_object_store.put.assert_any_call(obj_id, obj_content)

    def test_pull_and_store_message_with_known_run_id(self) -> None:
        """Test that a message of a known run ID is pulled and stored."""
        # Prepare
        self._prepare_for_pull_and_store_message()
        self.mock_state.get_run.return_value = Mock()  # Mock non-None return

        # Execute
        res = _pull_and_store_message(
            state=self.mock_state,
            object_store=self.mock_object_store,
            node_config={},  # No need for this test
            receive=self.mock_receive,
            get_run=self.mock_get_run,
            get_fab=self.mock_get_fab,
            pull_object=self.mock_pull_object,
            confirm_message_received=self.mock_confirm_message_received,
            trusted_entities={},
        )

        # Assert
        assert res == self.run_id
        self._assert_message_pulled_and_stored()

        # Assert: All are not called if run_id is known
        self.mock_get_run.assert_not_called()
        self.mock_get_fab.assert_not_called()
        self.mock_state.store_fab.assert_not_called()
        self.mock_state.store_run.assert_not_called()
        self.mock_state.store_context.assert_not_called()

    def test_pull_and_store_message_with_unknown_run_id(self) -> None:
        """Test that a message of an unknown run ID is pulled and stored."""
        # Prepare
        self._prepare_for_pull_and_store_message()
        self.mock_state.get_run.return_value = None  # Mock None return

        # Prepare: Mock get_run and get_fab functions
        fab = Fab(
            hash_str="abc123",
            content=b"test_fab_content",
            verifications={"abc123": "abc123"},
        )
        mock_run = Mock(
            run_id=self.run_id,
            fab_hash=fab.hash_str,
            override_config={},
        )
        self.mock_get_run.return_value = mock_run
        self.mock_get_fab.return_value = fab

        # Execute
        mock_fused_run_config = {"mock_key": "mock_value"}
        with patch(
            "flwr.supernode.start_client_internal.get_fused_config_from_fab"
        ) as mock_get_fused_config:
            mock_get_fused_config.return_value = mock_fused_run_config
            res = _pull_and_store_message(
                state=self.mock_state,
                object_store=self.mock_object_store,
                node_config={},  # No need for this test
                receive=self.mock_receive,
                get_run=self.mock_get_run,
                get_fab=self.mock_get_fab,
                pull_object=self.mock_pull_object,
                confirm_message_received=self.mock_confirm_message_received,
                trusted_entities={},
            )

        # Assert
        assert res == self.run_id
        self._assert_message_pulled_and_stored()

        # Assert: the Run and FAB should be fetched and stored if run_id is unknown
        self.mock_get_run.assert_called_once_with(self.run_id)
        self.mock_get_fab.assert_called_once_with(fab.hash_str, self.run_id)
        self.mock_state.store_fab.assert_called_once_with(fab)
        self.mock_state.store_run.assert_called_once_with(mock_run)

        # Assert: the Context should be created and stored if run_id is unknown
        self.mock_state.store_context.assert_called_once()
        args, _ = self.mock_state.store_context.call_args
        ctxt = args[0]
        assert isinstance(ctxt, Context)
        assert ctxt.run_id == self.run_id
        assert ctxt.node_id == self.node_id
        assert ctxt.node_config == {}
        assert ctxt.run_config == mock_fused_run_config

    def test_pull_and_store_message_rejects_missing_verification_metadata(self) -> None:
        """Test that trusted-entity verification fails closed without metadata."""
        # Prepare
        self._prepare_for_pull_and_store_message()
        self.mock_state.get_run.return_value = None

        fab = Fab(
            hash_str="abc123",
            content=b"test_fab_content",
            verifications={},
        )
        mock_run = Mock(
            run_id=self.run_id,
            fab_hash=fab.hash_str,
            override_config={},
        )
        self.mock_get_run.return_value = mock_run
        self.mock_get_fab.return_value = fab

        with patch(
            "flwr.supernode.start_client_internal._verify_fab"
        ) as mock_verify_fab:
            res = _pull_and_store_message(
                state=self.mock_state,
                object_store=self.mock_object_store,
                node_config={},
                receive=self.mock_receive,
                get_run=self.mock_get_run,
                get_fab=self.mock_get_fab,
                pull_object=self.mock_pull_object,
                confirm_message_received=self.mock_confirm_message_received,
                trusted_entities={"trusted": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA"},
            )

        assert res == self.run_id
        mock_verify_fab.assert_not_called()
        self.mock_get_run.assert_called_once_with(self.run_id)
        self.mock_get_fab.assert_called_once_with(fab.hash_str, self.run_id)
        self.mock_state.store_context.assert_not_called()
        self.mock_state.store_run.assert_not_called()
        self.mock_confirm_message_received.assert_not_called()

        self.mock_state.store_message.assert_called_once()
        stored_message = self.mock_state.store_message.call_args.args[0]
        assert stored_message.has_error()
        assert stored_message.error == FAB_VERIFICATION_ERROR
        assert (
            stored_message.metadata.reply_to_message_id
            == self.mock_receive.return_value[0].metadata.message_id
        )

    def test_pull_and_store_message_rejects_unverified_fab(self) -> None:
        """Test that trusted-entity verification rejects invalid FAB signatures."""
        # Prepare
        self._prepare_for_pull_and_store_message()
        self.mock_state.get_run.return_value = None

        fab = Fab(
            hash_str="abc123",
            content=b"test_fab_content",
            verifications={
                "valid_license": "Valid",
                "trusted": '{"signature": "x", "signed_at": "y"}',
            },
        )
        mock_run = Mock(
            run_id=self.run_id,
            fab_hash=fab.hash_str,
            override_config={},
        )
        self.mock_get_run.return_value = mock_run
        self.mock_get_fab.return_value = fab

        with patch(
            "flwr.supernode.start_client_internal._verify_fab", return_value=False
        ) as mock_verify_fab:
            res = _pull_and_store_message(
                state=self.mock_state,
                object_store=self.mock_object_store,
                node_config={},
                receive=self.mock_receive,
                get_run=self.mock_get_run,
                get_fab=self.mock_get_fab,
                pull_object=self.mock_pull_object,
                confirm_message_received=self.mock_confirm_message_received,
                trusted_entities={"trusted": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA"},
            )

        assert res == self.run_id
        mock_verify_fab.assert_called_once_with(
            fab, {"trusted": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA"}
        )
        self.mock_state.store_context.assert_not_called()
        self.mock_state.store_run.assert_not_called()
        self.mock_confirm_message_received.assert_not_called()

        self.mock_state.store_message.assert_called_once()
        stored_message = self.mock_state.store_message.call_args.args[0]
        assert stored_message.has_error()
        assert stored_message.error == FAB_VERIFICATION_ERROR
