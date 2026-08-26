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

import pytest
from grpc import RpcError

from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.app.message import remove_content_from_message
from flwr.common.constant import TRANSPORT_TYPE_GRPC_RERE, SubStatus
from flwr.supercore.constant import TaskType
from flwr.supercore.fab import Fab
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_tree,
)
from flwr.supercore.inflatable.inflatable_utils import ObjectPushError

from .start_client_internal import (
    FAB_VERIFICATION_ERROR,
    _pull_and_store_message,
    _push_messages,
    start_client_internal,
)


class TestStartClientInternal(unittest.TestCase):  # pylint: disable=R0902
    """Test cases for the main loop."""

    def setUp(self) -> None:
        """Set up the test case."""
        self.mock_state = Mock()
        self.node_id = 111
        self.run_id = 110
        self.series_id = 112
        self.mock_state.get_node_id.return_value = self.node_id
        self.mock_state.get_run_series_context.return_value = None
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
        fab_hash = "abc123"
        self.mock_state.get_run.return_value = Mock(fab_hash=fab_hash)
        self.mock_state.create_task.return_value = 123

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
        self.mock_state.create_task.assert_called_once_with(
            task_type=TaskType.CLIENT_APP,
            run_id=self.run_id,
            fab_hash=fab_hash,
        )

        # Assert: All are not called if run_id is known
        self.mock_get_run.assert_not_called()
        self.mock_get_fab.assert_not_called()
        self.mock_state.store_fab.assert_not_called()
        self.mock_state.store_run.assert_not_called()
        self.mock_state.set_run_series_context.assert_not_called()

    def test_pull_and_store_message_returns_none_if_create_task_fails(self) -> None:
        """Test that message processing stops if task creation fails."""
        self._prepare_for_pull_and_store_message()
        fab_hash = "abc123"
        self.mock_state.get_run.return_value = Mock(fab_hash=fab_hash)
        self.mock_state.create_task.return_value = None

        res = _pull_and_store_message(
            state=self.mock_state,
            object_store=self.mock_object_store,
            node_config={},
            receive=self.mock_receive,
            get_run=self.mock_get_run,
            get_fab=self.mock_get_fab,
            pull_object=self.mock_pull_object,
            confirm_message_received=self.mock_confirm_message_received,
            trusted_entities={},
        )

        assert res is None
        self.mock_state.create_task.assert_called_once_with(
            task_type=TaskType.CLIENT_APP,
            run_id=self.run_id,
            fab_hash=fab_hash,
        )
        self.mock_object_store.preregister.assert_not_called()
        self.mock_state.store_message.assert_not_called()
        self.mock_confirm_message_received.assert_not_called()

    def test_pull_and_store_message_marks_task_failed_if_object_pull_fails(
        self,
    ) -> None:
        """Test that object-pull failures clean up the message and fail the task."""
        self._prepare_for_pull_and_store_message()
        fab_hash = "abc123"
        task_id = 123
        message_id = self.mock_receive.return_value[0].metadata.message_id
        self.mock_state.get_run.return_value = Mock(fab_hash=fab_hash)
        self.mock_state.create_task.return_value = task_id
        self.mock_pull_object.side_effect = RuntimeError("error")

        res = _pull_and_store_message(
            state=self.mock_state,
            object_store=self.mock_object_store,
            node_config={},
            receive=self.mock_receive,
            get_run=self.mock_get_run,
            get_fab=self.mock_get_fab,
            pull_object=self.mock_pull_object,
            confirm_message_received=self.mock_confirm_message_received,
            trusted_entities={},
        )

        assert res is None
        self.mock_state.create_task.assert_called_once_with(
            task_type=TaskType.CLIENT_APP,
            run_id=self.run_id,
            fab_hash=fab_hash,
        )
        self.mock_state.delete_messages.assert_called_once_with(
            message_ids=[message_id]
        )
        self.mock_object_store.delete.assert_called_once_with(message_id)
        self.mock_state.finish_task.assert_called_once_with(
            task_id,
            sub_status=SubStatus.FAILED,
            details="Pulling message objects failed: error",
        )
        self.mock_confirm_message_received.assert_not_called()

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
            series_id=self.series_id,
        )
        self.mock_get_run.return_value = mock_run
        self.mock_get_fab.return_value = fab
        shared_state = RecordDict({"shared": ConfigRecord({"value": "kept"})})
        self.mock_state.get_run_series_context.return_value = Context(
            run_id=109,
            node_id=self.node_id,
            node_config={},
            state=shared_state,
            run_config={},
            series_id=self.series_id,
        )

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
        self.mock_state.get_run_series_context.assert_called_once_with(self.series_id)

        # Assert: the Context should be created and stored if run_id is unknown
        self.mock_state.set_run_series_context.assert_called_once()
        args, _ = self.mock_state.set_run_series_context.call_args
        assert args[0] == self.series_id
        ctxt = args[1]
        assert isinstance(ctxt, Context)
        assert ctxt.run_id == self.run_id
        assert ctxt.node_id == self.node_id
        assert ctxt.node_config == {}
        assert ctxt.state is shared_state
        assert ctxt.run_config == mock_fused_run_config
        assert ctxt.series_id == self.series_id

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
        self.mock_state.set_run_series_context.assert_not_called()
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
        self.mock_state.set_run_series_context.assert_not_called()
        self.mock_state.store_run.assert_not_called()
        self.mock_confirm_message_received.assert_not_called()

        self.mock_state.store_message.assert_called_once()
        stored_message = self.mock_state.store_message.call_args.args[0]
        assert stored_message.has_error()
        assert stored_message.error == FAB_VERIFICATION_ERROR


class _StopAfterSuperExecLaunch(Exception):
    """Stop start_client_internal after command construction."""


def _run_until_connection_start(
    runtime_certificates: tuple[bytes, bytes, bytes] | None = None,
    runtime_root_certificates_path: str | None = None,
    runtime_api_address: str = "127.0.0.1:9094",
) -> Mock:
    """Run startup only far enough to inspect SuperExec HTTP wiring."""
    objectstore_factory = Mock()
    state_factory = Mock(objectstore_factory=objectstore_factory)
    with (
        patch("flwr.supernode.start_client_internal.register_signal_handlers"),
        patch("flwr.supernode.start_client_internal.subprocess.Popen") as popen,
        patch(
            "flwr.supernode.start_client_internal._init_connection",
            side_effect=_StopAfterSuperExecLaunch,
        ),
    ):
        # `_init_connection` starts the long-running SuperNode/SuperLink connection.
        # Raising there keeps this test focused on the setup performed before it.
        with pytest.raises(_StopAfterSuperExecLaunch):
            start_client_internal(
                state_factory=state_factory,
                server_address="127.0.0.1:9092",
                node_config={},
                root_certificates=None,
                insecure=True,
                transport=TRANSPORT_TYPE_GRPC_RERE,
                runtime_api_address=runtime_api_address,
                runtime_certificates=runtime_certificates,
                runtime_root_certificates_path=runtime_root_certificates_path,
            )

    return popen


def test_start_client_internal_launches_insecure_superexec_by_default() -> None:
    """Subprocess SuperExec should use insecure Runtime API when TLS is off."""
    # This verifies the default subprocess-isolation path: when SuperNode starts
    # Runtime API without server TLS means the spawned SuperExec uses plaintext too.
    popen = _run_until_connection_start()

    # No Runtime API server certificates means the local server is plaintext,
    # so the child SuperExec must connect with `--insecure`.
    command = popen.call_args.args[0]
    assert command[:2] == ["flower-superexec", "--insecure"]
    assert "--root-certificates" not in command
    assert command[command.index("--appio-api-address") + 1] == "127.0.0.1:9094"


def test_start_client_internal_launches_secure_superexec_with_root_certificates() -> (
    None
):
    """Subprocess SuperExec should trust the secure Runtime API server CA."""
    # This verifies the TLS subprocess-isolation path: when SuperNode starts
    # Runtime API with server TLS means the spawned SuperExec receives trust roots.
    certificates = (b"ca", b"cert", b"key")

    popen = _run_until_connection_start(
        runtime_certificates=certificates,
        runtime_root_certificates_path="/tmp/ca.pem",
    )

    # When the Runtime API starts with TLS, SuperExec should verify that server
    # certificate with the same CA file instead of falling back to plaintext.
    command = popen.call_args.args[0]
    assert "--insecure" not in command
    assert command[:3] == ["flower-superexec", "--root-certificates", "/tmp/ca.pem"]


def test_start_client_internal_launches_superexec_with_runtime_http_address() -> None:
    """Subprocess SuperExec should use the Runtime HTTP API address."""
    popen = _run_until_connection_start(runtime_api_address="localhost:54321")

    command = popen.call_args.args[0]
    assert command[command.index("--appio-api-address") + 1] == "localhost:54321"


def _make_reply_message() -> Message:
    """Create a reply message suitable for testing the sender path."""
    instruction = Message(content=RecordDict(), dst_node_id=1, message_type="query")
    reply = Message(content=RecordDict(), reply_to=instruction)
    reply.metadata.__dict__["_run_id"] = 1
    reply.metadata.__dict__["_message_id"] = reply.object_id
    return reply


def test_push_messages_deletes_local_state_only_after_success() -> None:
    """A successful message and object push removes the local reply."""
    state = Mock()
    object_store = Mock()
    message = _make_reply_message()
    state.get_messages.return_value = [message]
    state.get_message_processing_duration.return_value = 0.1
    send = Mock(return_value=({"object-id"}, "session-id"))
    push_object = Mock()

    with patch(
        "flwr.supernode.start_client_internal.push_object_contents_from_iterable"
    ):
        _push_messages(state, object_store, send, push_object)

    state.delete_messages.assert_called_once_with(
        message_ids=[message.metadata.message_id, message.metadata.reply_to_message_id]
    )
    object_store.delete.assert_called_once_with(message.metadata.message_id)


def test_push_messages_retries_after_object_upload_failure() -> None:
    """A failed object upload retries PushMessages with a new session."""
    state = Mock()
    object_store = Mock()
    message = _make_reply_message()
    state.get_messages.return_value = [message]
    state.get_message_processing_duration.return_value = 0.1
    send = Mock(
        side_effect=[
            ({"object-id"}, "old-session"),
            ({"object-id"}, "new-session"),
        ]
    )
    push_object = Mock()

    with (
        patch("flwr.supernode.start_client_internal.time.sleep"),
        patch(
            "flwr.supernode.start_client_internal.push_object_contents_from_iterable",
            side_effect=[ObjectPushError("object-id", 1, "old-session"), None],
        ),
    ):
        _push_messages(state, object_store, send, push_object)

    assert send.call_count == 2
    state.delete_messages.assert_called_once()
    object_store.delete.assert_called_once_with(message.metadata.message_id)


def test_push_messages_retries_after_push_messages_failure() -> None:
    """A failed PushMessages call retries the complete message push."""
    state = Mock()
    object_store = Mock()
    message = _make_reply_message()
    state.get_messages.return_value = [message]
    state.get_message_processing_duration.return_value = 0.1
    send = Mock(
        side_effect=[RpcError(), ({"object-id"}, "session-id")],
    )

    with (
        patch("flwr.supernode.start_client_internal.time.sleep"),
        patch(
            "flwr.supernode.start_client_internal.push_object_contents_from_iterable"
        ),
    ):
        _push_messages(state, object_store, send, Mock())

    assert send.call_count == 2
    state.delete_messages.assert_called_once()
    object_store.delete.assert_called_once_with(message.metadata.message_id)


def test_push_messages_retains_local_state_after_non_retryable_failure() -> None:
    """A failed push does not delete a reply that still needs delivery."""
    state = Mock()
    object_store = Mock()
    message = _make_reply_message()
    state.get_messages.return_value = [message]
    state.get_message_processing_duration.return_value = 0.1
    send = Mock(side_effect=ValueError("push failed"))

    _push_messages(state, object_store, send, Mock())

    state.delete_messages.assert_not_called()
    object_store.delete.assert_not_called()
