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
"""Fleet API message handler tests."""


from unittest.mock import MagicMock, patch

from flwr.app import Metadata, RecordDict
from flwr.app.message import make_message
from flwr.common.capability import capability_binding, participant_id_from_public_key
from flwr.common.constant import Status
from flwr.common.serde import message_to_proto
from flwr.proto.fleet_pb2 import (  # pylint: disable=E0611
    PullMessagesRequest,
    PushMessagesRequest,
)
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.proto.node_pb2 import Node, NodeInfo  # pylint: disable=E0611
from flwr.proto.run_pb2 import GetRunRequest  # pylint: disable=E0611
from flwr.supercore.date import now
from flwr.supercore.run import Run

from .message_handler import get_run, pull_messages, push_messages


def test_get_run_routes_capability_by_registered_public_key() -> None:
    """Return only the capability for the requesting SuperNode."""
    public_key = b"canonical-public-key"
    participant_id = participant_id_from_public_key(public_key)
    run = Run.create_empty(123)
    run.federation_id = "@account/federation"
    run.fab_hash = "a" * 64
    run.status.status = Status.RUNNING
    run.capability_packages = {
        participant_id: b"raw-secret-package",
        "flwr-p384-spki-pem-sha256:" + "b" * 64: b"other",
    }
    state = MagicMock()
    state.get_run_info.return_value = [run]
    state.federation_manager.has_node.return_value = True
    state.get_node_info.return_value = [NodeInfo(public_key=public_key)]

    with patch(
        "flwr.server.superlink.fleet.message_handler.message_handler.log"
    ) as mock_log:
        response = get_run(GetRunRequest(node=Node(node_id=7), run_id=123), state)

    assert response.run.capability_required
    assert response.run.capability_package == b"raw-secret-package"
    assert response.run.capability_binding == capability_binding(
        run.federation_id, run.fab_hash
    )
    rendered = " ".join(str(call.args) for call in mock_log.call_args_list)
    assert "[CAPABILITY]" in rendered
    assert "GetRun route" in rendered
    assert "123" in rendered and "7" in rendered
    assert participant_id[-64:-52] in rendered
    assert "match=%s%s" in rendered
    assert "[STORY]" in rendered
    assert "Capability %s: node_id=%s participant=%s" in rendered
    assert "selected" in rendered
    assert "raw-secret-package" not in rendered


def test_get_run_marks_missing_participant_capability_as_required() -> None:
    """Let SuperNode fail closed when no package matches its registered key."""
    run = Run.create_empty(123)
    run.status.status = Status.RUNNING
    run.capability_packages = {"flwr-p384-spki-pem-sha256:" + "b" * 64: b"other"}
    state = MagicMock()
    state.get_run_info.return_value = [run]
    state.federation_manager.has_node.return_value = True
    state.get_node_info.return_value = [NodeInfo(public_key=b"unmatched-key")]

    with patch(
        "flwr.server.superlink.fleet.message_handler.message_handler.log"
    ) as mock_log:
        response = get_run(GetRunRequest(node=Node(node_id=7), run_id=123), state)

    assert response.run.capability_required
    assert response.run.capability_package == b""
    rendered = " ".join(str(call.args) for call in mock_log.call_args_list)
    assert "missing" in rendered
    assert "fail_closed=true" in rendered
    assert "[STORY]" in rendered
    assert "Capability %s: node_id=%s participant=%s" in rendered


def test_pull_messages() -> None:
    """Test pull_messages."""
    # Prepare
    request = PullMessagesRequest(node=Node(node_id=1234))
    state = MagicMock()
    store = MagicMock()

    # Execute
    pull_messages(request=request, state=state, store=store)

    # Assert
    state.create_node.assert_not_called()
    state.delete_node.assert_not_called()
    state.store_message_ins.assert_not_called()
    state.get_message_ins.assert_called_once()
    state.store_message_res.assert_not_called()
    state.get_message_res.assert_not_called()
    state.store_traffic.assert_not_called()


def test_pull_messages_records_traffic_when_messages_found() -> None:
    """Test pull_messages records traffic when messages are successfully retrieved."""
    # Prepare
    msg = make_message(
        content=RecordDict(),
        metadata=Metadata(
            run_id=234,
            message_id="msg-234",
            group_id="",
            src_node_id=0,
            dst_node_id=1234,
            reply_to_message_id="",
            created_at=now().timestamp(),
            ttl=123,
            message_type="query",
        ),
    )
    request = PullMessagesRequest(node=Node(node_id=2345))
    state = MagicMock()
    state.get_message_ins.return_value = [msg]
    store = MagicMock()
    store.get_object_tree.return_value = {}

    # Execute
    pull_messages(request=request, state=state, store=store)

    # Assert
    state.get_message_ins.assert_called_once()
    store.get_object_tree.assert_called_once_with("msg-234")
    state.store_traffic.assert_called_once()
    # Verify store_traffic was called with run_id=123, bytes_sent > 0, bytes_recv=0
    call_args = state.store_traffic.call_args
    assert call_args[0][0] == 234  # run_id
    assert call_args[1]["bytes_sent"] > 0
    assert call_args[1]["bytes_recv"] > 0


def test_push_messages() -> None:
    """Test push_messages."""
    # Prepare
    msg = make_message(
        content=RecordDict(),
        metadata=Metadata(
            run_id=123,
            message_id="",
            group_id="",
            src_node_id=0,
            dst_node_id=0,
            reply_to_message_id="",
            created_at=now().timestamp(),
            ttl=123,
            message_type="query",
        ),
    )

    object_tree = ObjectTree(object_id="object-id")
    request = PushMessagesRequest(
        messages_list=[message_to_proto(msg)],
        message_object_trees=[object_tree],
    )
    state = MagicMock()
    state.start_session.return_value = "session-id"
    state.store_message_and_object_tree.return_value = (True, ["object-id"])

    # Execute
    response = push_messages(request=request, state=state)

    # Assert
    state.create_node.assert_not_called()
    state.delete_node.assert_not_called()
    state.store_message_ins.assert_not_called()
    state.get_message_ins.assert_not_called()
    state.store_message_res.assert_not_called()
    state.start_session.assert_called_once_with(123)
    state.store_message_and_object_tree.assert_called_once()
    assert state.store_message_and_object_tree.call_args.args[1] == object_tree
    assert state.store_message_and_object_tree.call_args.args[2] == "session-id"
    assert response.session_id == "session-id"
    state.get_message_res.assert_not_called()
    state.store_traffic.assert_called_once()
