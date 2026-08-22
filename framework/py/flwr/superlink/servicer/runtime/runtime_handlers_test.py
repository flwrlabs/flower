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
"""SuperLink Runtime API handler tests."""

# pylint: disable=too-many-lines


import hashlib
import os
import tempfile
import threading
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

from parameterized import parameterized

from flwr.app import ConfigRecord, Context, Error, Message, RecordDict
from flwr.common.constant import NOOP_FLWR_AID, SUPERLINK_NODE_ID, Status, SubStatus
from flwr.common.serde import context_to_proto, message_from_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartAutomationRequest,
    StartAutomationResponse,
    StartRunRequest,
)
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    ConfirmMessageReceivedRequest,
    ConfirmMessageReceivedResponse,
    ObjectTree,
    PullObjectRequest,
    PushObjectRequest,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.proto.runtime_pb2 import (  # pylint: disable=E0611
    ClaimTaskRequest,
    CreateTaskRequest,
    GetConnectorRequest,
    GetConnectorResponse,
    GetNodesRequest,
    GetNodesResponse,
    PullAppMessagesRequest,
    PullAppMessagesResponse,
    PullPendingTasksRequest,
    PullTaskInputRequest,
    PullTaskInputResponse,
    PushAppMessagesRequest,
    PushAppMessagesResponse,
    PushTaskOutputRequest,
    PushTaskOutputResponse,
)
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611
from flwr.server.superlink.linkstate.linkstate import LinkState
from flwr.server.superlink.linkstate.linkstate_factory import LinkStateFactory
from flwr.server.superlink.linkstate.linkstate_test import create_ins_message
from flwr.supercore.constant import (
    FLWR_IN_MEMORY_DB_NAME,
    NOOP_FEDERATION_ID,
    AutomationStatus,
    TaskType,
)
from flwr.supercore.date import now
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.fab import Fab
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_id,
    get_object_tree,
    iterate_object_tree,
)
from flwr.supercore.object_store import ObjectStoreFactory
from flwr.supercore.servicer.runtime import runtime_handlers as core_runtime_handlers
from flwr.superlink.federation import NoOpFederationManager
from flwr.superlink.servicer.runtime import runtime_handlers

# pylint: disable=broad-except,too-many-lines


def test_raise_if_false() -> None:
    """."""
    # Prepare
    validation_error = False
    detail = "test"

    try:
        # Execute
        runtime_handlers._raise_if(  # pylint: disable=protected-access
            validation_error=validation_error,
            request_name="DummyRequest",
            detail=detail,
        )

        # Assert
        assert True
    except ValueError as err:
        raise AssertionError() from err
    except Exception as err:
        raise AssertionError() from err


def test_raise_if_true() -> None:
    """."""
    # Prepare
    validation_error = True
    detail = "test"

    try:
        # Execute
        runtime_handlers._raise_if(  # pylint: disable=protected-access
            validation_error=validation_error,
            request_name="DummyRequest",
            detail=detail,
        )

        # Assert
        raise AssertionError()
    except ValueError as err:
        assert str(err) == "Malformed DummyRequest: test"
    except Exception as err:
        raise AssertionError() from err


def _create_shared_runtime(
    tmpdir: str,
) -> tuple[int, int, LinkState, LinkState]:
    database_path = os.path.join(tmpdir, "shared.db")

    objectstore_factory_0 = ObjectStoreFactory()
    objectstore_factory_1 = ObjectStoreFactory()
    state_factory_0 = LinkStateFactory(
        database_path, NoOpFederationManager(), objectstore_factory_0
    )
    state_factory_1 = LinkStateFactory(
        database_path, NoOpFederationManager(), objectstore_factory_1
    )
    state_0 = state_factory_0.state()
    fab_content = b"mock fab content"
    fab_hash = state_0.store_fab(
        Fab(hashlib.sha256(fab_content).hexdigest(), fab_content, {})
    )

    run_id = state_0.create_run(
        "",
        "",
        fab_hash,
        {},
        NOOP_FEDERATION_ID,
        None,
        "",
        TaskType.SERVER_APP,
    )
    run = state_0.get_run_info(run_ids=[run_id])[0]
    state_0.set_run_series_context(
        run.series_id,
        Context(
            run_id,
            SUPERLINK_NODE_ID,
            {},
            RecordDict(),
            {},
            series_id=run.series_id,
        ),
    )
    assert run.primary_task_id is not None
    task_id = run.primary_task_id
    return run_id, task_id, state_0, state_factory_1.state()


def _activate_in_parallel(
    state_0: LinkState, state_1: LinkState, task: Task
) -> list[bool | None]:
    timeout = 5.0
    barrier = threading.Barrier(3)
    results: list[bool | None] = [None, None]
    exceptions: list[Exception] = []

    def activate_task(idx: int, state: LinkState) -> None:
        try:
            barrier.wait(timeout=timeout)
            runtime_handlers.pull_task_input(PullTaskInputRequest(), state, task)
            results[idx] = True
        except FlowerError as err:
            assert err.code == ApiErrorCode.RUNTIME_TASK_START_FAILED
            results[idx] = False
        except Exception as ex:  # pylint: disable=broad-exception-caught
            exceptions.append(ex)

    threads = [
        threading.Thread(target=activate_task, args=(0, state_0)),
        threading.Thread(target=activate_task, args=(1, state_1)),
    ]
    for thread in threads:
        thread.start()
    try:
        barrier.wait(timeout=timeout)
    except threading.BrokenBarrierError as ex:
        exceptions.append(ex)
    for thread in threads:
        thread.join(timeout=timeout)

    alive_threads = [thread for thread in threads if thread.is_alive()]
    if alive_threads:
        raise AssertionError(
            f"Concurrent PullTaskInput test timed out; {len(alive_threads)} "
            f"thread(s) still alive after {timeout} seconds."
        )
    if exceptions:
        raise exceptions[0]
    return results


class TestGetConnector(unittest.TestCase):
    """Test connector credential authorization with mocked state."""

    def setUp(self) -> None:
        """Create mocked state."""
        self.state = Mock(spec=LinkState)

    def test_returns_authenticated_task_credentials(self) -> None:
        """GetConnector should return the run owner's matching credentials."""
        task = Mock(type=TaskType.CONNECTOR, connector_ref="notion", run_id=123)
        self.state.get_run_info.return_value = [Mock(flwr_aid="account-a")]
        self.state.get_connector.return_value = Mock(
            connector_ref="notion",
            credentials_json='{"token":"secret"}',
            config_json='{"workspace":"primary"}',
        )

        response = runtime_handlers.get_connector(
            GetConnectorRequest(), self.state, task
        )

        self.assertEqual(
            response,
            GetConnectorResponse(
                connector_ref="notion",
                credentials_json='{"token":"secret"}',
                config_json='{"workspace":"primary"}',
            ),
        )
        self.state.get_connector.assert_called_once_with(
            flwr_aid="account-a",
            connector_ref="notion",
        )

    @parameterized.expand(  # type: ignore
        [
            ("wrong_task_type", TaskType.AGENT_APP, "notion"),
            ("missing_ref", TaskType.CONNECTOR, ""),
        ]
    )
    def test_rejects_wrong_task_identity(
        self,
        _name: str,
        task_type: str,
        connector_ref: str,
    ) -> None:
        """GetConnector should reject tasks without a connector identity."""
        task = Mock(type=task_type, connector_ref=connector_ref, run_id=123)
        with self.assertRaises(FlowerError) as error:
            runtime_handlers.get_connector(GetConnectorRequest(), self.state, task)

        self.assertEqual(
            error.exception.code,
            ApiErrorCode.RUNTIME_CONNECTOR_CREDENTIALS_NOT_AVAILABLE,
        )
        self.state.get_connector.assert_not_called()

    def test_hides_other_account_credentials(self) -> None:
        """GetConnector should not fall back to another account's credentials."""
        task = Mock(type=TaskType.CONNECTOR, connector_ref="notion", run_id=123)
        self.state.get_run_info.return_value = [Mock(flwr_aid="account-b")]
        self.state.get_connector.return_value = None

        with self.assertRaises(FlowerError) as error:
            runtime_handlers.get_connector(GetConnectorRequest(), self.state, task)

        self.state.get_connector.assert_called_once_with(
            flwr_aid="account-b",
            connector_ref="notion",
        )
        self.assertEqual(error.exception.code, ApiErrorCode.CONNECTOR_NOT_FOUND)


class TestSuperLinkRuntimeHandlers(unittest.TestCase):  # pylint: disable=R0902, R0904
    """SuperLink Runtime API handler tests."""

    def setUp(self) -> None:
        """Initialize state shared by handler tests."""
        objectstore_factory = ObjectStoreFactory()
        state_factory = LinkStateFactory(
            FLWR_IN_MEMORY_DB_NAME, NoOpFederationManager(), objectstore_factory
        )
        self.objectstore_factory = objectstore_factory
        self.state_factory = state_factory
        self.state = state_factory.state()
        self.store = objectstore_factory.store()
        self.node_pk = b"fake public key"
        self.node_id = self.state.create_node(
            "mock_owner", "fake_name", self.node_pk, 30
        )
        self.state.acknowledge_node_heartbeat(self.node_id, 1e3)

        self._auth_run_id = self.state.create_run(
            "", "", "", {}, NOOP_FEDERATION_ID, None, "", TaskType.SERVER_APP
        )
        auth_task_id = self._primary_task_id(self._auth_run_id)
        assert self.state.claim_task(auth_task_id) is not None
        assert self.state.activate_task(auth_task_id)
        self._auth_task = self.state.get_tasks(task_ids=[auth_task_id])[0]

    def _primary_task_id(self, run_id: int) -> int:
        run = self.state.get_run_info(run_ids=[run_id])[0]
        assert run.primary_task_id is not None
        return run.primary_task_id

    def _transition_run_status(self, run_id: int, num_transitions: int) -> None:
        task_id = self._primary_task_id(run_id)
        if num_transitions > 0:
            assert self.state.claim_task(task_id) is not None
        if num_transitions > 1:
            assert self.state.activate_task(task_id)
        if num_transitions > 2:
            assert self.state.finish_task(task_id, "", "")

    def test_pull_pending_tasks_processes_due_automations(self) -> None:
        """A SuperExec poll should create and return a due automation's task."""
        series_id = self.state.get_run_info(run_ids=[self._auth_run_id])[0].series_id
        automation = self.state.store_automation(
            federation_id=NOOP_FEDERATION_ID,
            flwr_aid=NOOP_FLWR_AID,
            start_run_request=StartRunRequest(
                app_spec="@flwr/demo",
                federation=NOOP_FEDERATION_ID,
                series_id=series_id,
            ),
            series_id=series_id,
            next_run_at=datetime.now(tz=UTC).isoformat(),
            max_runs=1,
        )

        with (
            patch(
                "flwr.superlink.servicer.control.control_handlers._get_remote_fab",
                return_value=(b"fab", {}, None),
            ),
            patch(
                "flwr.superlink.servicer.control.control_handlers.get_fab_config",
                return_value={"tool": {"flwr": {"app": {}}}},
            ),
            patch(
                "flwr.superlink.servicer.control.control_handlers"
                ".get_metadata_from_config",
                return_value=("flwr/demo", "0.1.0"),
            ),
        ):
            response = runtime_handlers.pull_pending_tasks(
                PullPendingTasksRequest(), self.state
            )

        self.assertEqual(len(response.tasks), 1)
        run = self.state.get_run_info(run_ids=[response.tasks[0].run_id])[0]
        self.assertEqual(run.series_id, automation.series_id)
        completed = self.state.list_automations(
            automation_ids=[automation.automation_id],
            statuses=[AutomationStatus.COMPLETED],
            order_by="updated_at",
        )
        self.assertEqual(len(completed), 1)
        active = self.state.list_automations(
            automation_ids=[automation.automation_id],
            statuses=[AutomationStatus.ACTIVE],
            order_by="updated_at",
        )
        self.assertEqual(active, [])

    def _create_dummy_run(self, running: bool = True, *, fab_hash: str = "") -> int:
        run_id = self.state.create_run(
            "",
            "",
            fab_hash,
            {},
            NOOP_FEDERATION_ID,
            None,
            "",
            TaskType.SERVER_APP,
        )
        if running:
            self._transition_run_status(run_id, 2)
        return run_id

    def test_create_task_uses_authenticated_run_id(self) -> None:
        """CreateTask should create tasks for the authenticated run."""
        response = core_runtime_handlers.create_task(
            CreateTaskRequest(type=TaskType.MODEL, model_ref="models/abc"),
            self.state,
            self._auth_task,
        )

        assert response.HasField("task_id")
        task = self.state.get_tasks(task_ids=[response.task_id])[0]
        assert task.run_id == self._auth_run_id
        assert task.type == TaskType.MODEL
        assert task.model_ref == "models/abc"

    def test_start_automation_enriches_connector_refs(self) -> None:
        """Enrich connector refs and delegate automation creation."""
        # Prepare
        request = StartAutomationRequest(
            start_run_request=StartRunRequest(connector_refs=["untrusted"])
        )
        expected = StartAutomationResponse(automation_id=1)

        # Execute
        with (
            patch(
                "flwr.superlink.servicer.runtime.runtime_handlers."
                "start_control_automation",
                return_value=expected,
            ) as start_automation_mock,
            patch.object(
                self.state, "get_run_connector_refs", return_value=["calendar"]
            ),
        ):
            response = runtime_handlers.start_automation(
                request,
                self.state,
                Task(run_id=self._auth_run_id, type=TaskType.SERVER_APP),
            )

        # Assert
        assert response is expected
        assert list(request.start_run_request.connector_refs) == ["calendar"]
        assert start_automation_mock.call_args.args[0] is request

    def test_start_automation_rejects_clientapp_task(self) -> None:
        """ClientApp tasks cannot create automations through the Runtime API."""
        # Execute
        with self.assertRaises(FlowerError) as error:
            runtime_handlers.start_automation(
                StartAutomationRequest(),
                self.state,
                Task(run_id=self._auth_run_id, type=TaskType.CLIENT_APP),
            )

        # Assert
        self.assertEqual(
            error.exception.code,
            ApiErrorCode.RUNTIME_AUTOMATION_CREATION_NOT_ALLOWED,
        )

    def test_push_task_output_stores_simulation_runtime(self) -> None:
        """PushTaskOutput should persist Simulation Runtime usage."""
        # Prepare
        request = PushTaskOutputRequest(
            sub_status="completed",
            details="",
            clientapp_runtime=7.89,
        )

        # Execute
        response = runtime_handlers.push_task_output(
            request, self.state, self._auth_task
        )

        # Assert
        assert isinstance(response, PushTaskOutputResponse)
        run = self.state.get_run_info(run_ids=[self._auth_run_id])[0]
        assert run.clientapp_runtime == 7.89

    def test_push_task_output_stores_run_series_context(self) -> None:
        """PushTaskOutput should persist context in the authenticated run series."""
        # Prepare
        run = self.state.get_run_info(run_ids=[self._auth_run_id])[0]
        request_context = Context(
            run_id=123,
            node_id=SUPERLINK_NODE_ID,
            node_config={"key": "value"},
            state=RecordDict(),
            run_config={"test": "test"},
            series_id=456,
        )
        request = PushTaskOutputRequest(
            sub_status="completed",
            details="",
            context=context_to_proto(request_context),
        )

        # Execute
        response = runtime_handlers.push_task_output(
            request, self.state, self._auth_task
        )

        # Assert
        assert isinstance(response, PushTaskOutputResponse)
        stored_context = self.state.get_run_series_context(run.series_id)
        assert stored_context is not None
        assert stored_context == request_context

    def test_get_node(self) -> None:
        """Test `GetNode` success."""
        # Prepare
        request = GetNodesRequest()

        # Execute
        response = runtime_handlers.get_nodes(request, self.state, self._auth_task)

        # Assert
        assert isinstance(response, GetNodesResponse)

    def test_push_messages_keeps_shared_upload_hint_after_rejection(self) -> None:
        """PushMessages should keep accepted-message upload hints."""
        # Prepare
        run_id = self._auth_run_id
        message_1 = create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=self.node_id, run_id=run_id
        )
        message_2 = create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=self.node_id, run_id=run_id
        )
        shared_child_id = hashlib.sha256(b"shared-child").hexdigest()
        object_tree_1 = ObjectTree(
            object_id=message_1.metadata.message_id,
            children=[ObjectTree(object_id=shared_child_id)],
        )
        object_tree_2 = ObjectTree(
            object_id=message_2.metadata.message_id,
            children=[ObjectTree(object_id=shared_child_id)],
        )
        request = PushAppMessagesRequest(
            messages_list=[message_1, message_2],
            message_object_trees=[object_tree_1, object_tree_2],
        )
        original_store_message_ins = self.state.store_message_ins
        primary_task_id = self._primary_task_id(run_id)
        call_count = 0

        def store_message_ins_and_finish_run(message: Message) -> str | None:
            nonlocal call_count
            call_count += 1
            message_id = original_store_message_ins(message)
            if call_count == 1:
                assert self.state.finish_task(primary_task_id, SubStatus.COMPLETED, "")
            return message_id

        # Execute
        with patch.object(
            self.state,
            "store_message_ins",
            side_effect=store_message_ins_and_finish_run,
        ):
            response = runtime_handlers.push_messages(
                request, self.state, self._auth_task
            )

        # Assert
        assert isinstance(response, PushAppMessagesResponse)
        assert list(response.message_ids) == [message_1.metadata.message_id, ""]
        assert response.session_id
        assert set(response.objects_to_push) == {
            message_1.metadata.message_id,
            shared_child_id,
        }

    @parameterized.expand(
        [
            # The normal case:
            # The message is recognized by both `LinkState` and `ObjectStore`
            (True,),
            # The failure case:
            # The message is found in `LinkState` but not in `ObjectStore`
            (False,),
        ]
    )  # type: ignore
    def test_pull_messages(self, register_in_store: bool) -> None:
        """Test `PullMessages` success if objects are registered in ObjectStore."""
        # Prepare
        run_id = self._auth_run_id

        # Push Messages and reply
        message_ins = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=self.node_id, run_id=run_id
            )
        )
        # pylint: disable-next=W0212
        message_ins.metadata._message_id = message_ins.object_id  # type: ignore
        msg_id = self.state.store_message_ins(message=message_ins)
        msg_ = self.state.get_message_ins(node_id=self.node_id, limit=1)[0]

        reply_msg = Message(RecordDict(), reply_to=msg_)
        # pylint: disable-next=W0212
        reply_msg.metadata._message_id = reply_msg.object_id  # type: ignore
        self.state.store_message_res(message=reply_msg)

        # Register response in ObjectStore (so pulling message request can be completed)
        obj_ids_registered: list[str] = []
        if register_in_store:
            obj_ids_registered = self.store.preregister(
                run_id, get_object_tree(reply_msg)
            )

        request = PullAppMessagesRequest(message_ids=[str(msg_id)])

        # Execute
        response = runtime_handlers.pull_messages(request, self.state, self._auth_task)

        # Assert
        assert isinstance(response, PullAppMessagesResponse)

        if register_in_store:
            object_tree = response.message_object_trees[0]
            object_ids_in_response = [
                tree.object_id for tree in iterate_object_tree(object_tree)
            ]
            # Assert expected object_ids
            assert set(obj_ids_registered) == set(object_ids_in_response)
            # Assert the root node of the object tree is the message
            assert reply_msg.object_id == object_tree.object_id
        else:
            assert len(response.messages_list) == 0
            assert len(response.message_object_trees) == 0
            # Ins message was deleted
            assert self.state.num_message_ins() == 0

    @parameterized.expand(
        [
            # Reply with Message
            (RecordDict(), None),
            # Reply with Error
            (None, Error(code=0)),
        ]
    )  # type: ignore
    def test_successful_pull_messages_deletes_messages_in_linkstate(
        self, content: RecordDict | None, error: Error | None
    ) -> None:
        """Test `PullMessages` deletes messages from LinkState."""
        # Prepare
        run_id = self._auth_run_id

        # Push Messages and reply
        message_ins = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=self.node_id, run_id=run_id
            )
        )
        # pylint: disable-next=W0212
        message_ins.metadata._message_id = message_ins.object_id  # type: ignore

        msg_id = self.state.store_message_ins(message=message_ins)
        msg_ = self.state.get_message_ins(node_id=self.node_id, limit=1)[0]

        if content is not None:
            reply_msg = Message(content, reply_to=msg_)
        else:
            assert error is not None
            reply_msg = Message(error, reply_to=msg_)

        # pylint: disable-next=W0212
        reply_msg.metadata._message_id = reply_msg.object_id  # type: ignore

        self.state.store_message_res(message=reply_msg)
        # Register response in ObjectStore (so pulling message request can be completed)
        self.store.preregister(run_id, get_object_tree(reply_msg))
        request = PullAppMessagesRequest(message_ids=[str(msg_id)])

        # Execute
        response = runtime_handlers.pull_messages(request, self.state, self._auth_task)

        # Assert
        assert isinstance(response, PullAppMessagesResponse)
        assert self.state.num_message_ins() == 0
        assert self.state.num_message_res() == 0

    def test_pull_message_from_expired_message_error(self) -> None:
        """Test that the servicer correctly handles the registration in the ObjectStore
        of an Error message created by the LinkState due to an expired TTL."""
        # Prepare
        run_id = self._auth_run_id

        # Push Messages and reply
        message_ins = message_from_proto(
            create_ins_message(
                src_node_id=SUPERLINK_NODE_ID, dst_node_id=self.node_id, run_id=run_id
            )
        )
        message_ins.metadata.ttl = 1  # set short TTL for testing
        msg_id = self.state.store_message_ins(message=message_ins)

        # Simulate situation where the message has expired in the LinkState
        # This will trigger the creation of an Error message
        future_dt = now() + timedelta(seconds=message_ins.metadata.ttl + 0.1)
        with patch("datetime.datetime") as mock_dt:
            mock_dt.now.return_value = future_dt  # over TTL limit

            # Execute
            request = PullAppMessagesRequest(message_ids=[str(msg_id)])
            response = runtime_handlers.pull_messages(
                request, self.state, self._auth_task
            )

            # Assert
            assert isinstance(response, PullAppMessagesResponse)

            # Assert that objects to pull points to a message carrying an error
            msg_res = message_from_proto(response.messages_list[0])
            assert msg_res.has_error()
            object_tree = response.message_object_trees[0]
            object_ids_in_response = [
                tree.object_id for tree in iterate_object_tree(object_tree)
            ]
            # expected a single object id (that of the error message)
            assert list(object_ids_in_response) == [msg_res.object_id]

    def test_push_object_successful(self) -> None:
        """Test `PushObject`."""
        # Prepare
        run_id = self._auth_run_id
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Pre-register object
        session_id = self.state.start_session(run_id)
        self.state.preregister_object_tree(get_object_tree(obj), session_id)

        # Execute
        req = PushObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID),
            run_id=run_id,
            object_id=obj.object_id,
            object_content=obj_b,
            session_id=session_id,
        )
        res = runtime_handlers.push_object(req, self.state, self._auth_task)

        # Empty response
        assert res.stored

    def test_push_object_fails(self) -> None:
        """Test `PushObject` in unsupported scenarios."""
        run_id = self._auth_run_id

        # Node ID isn't recognized
        req = PushObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(FlowerError) as error:
            runtime_handlers.push_object(req, self.state, self._auth_task)
        assert error.exception.code == ApiErrorCode.RUNTIME_UNEXPECTED_NODE_ID

        # Prepare
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()
        session_id = self.state.start_session(run_id)

        # Push valid object but it hasn't been pre-registered
        req = PushObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID),
            run_id=run_id,
            object_id=obj.object_id,
            object_content=obj_b,
            session_id=session_id,
        )
        res = runtime_handlers.push_object(req, self.state, self._auth_task)

        # Assert: object not inserted
        assert not res.stored

        # Push valid object but its hash doesnt match the one passed in the request
        # Preregister under a different object-id
        fake_object_id = get_object_id(b"1234")
        self.state.preregister_object_tree(
            ObjectTree(object_id=fake_object_id), session_id
        )

        # Execute
        req = PushObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID),
            run_id=run_id,
            object_id=fake_object_id,
            object_content=obj_b,
            session_id=session_id,
        )
        res = runtime_handlers.push_object(req, self.state, self._auth_task)

        # Assert: object not inserted
        assert not res.stored

    def test_pull_object_successful(self) -> None:
        """Test `PullObject` functionality."""
        # Prepare
        run_id = self._create_dummy_run()
        obj = ConfigRecord({"a": 123, "b": [4, 5, 6]})
        obj_b = obj.deflate()

        # Preregister object
        self.store.preregister(run_id, get_object_tree(obj))

        # Pull
        req = PullObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID), run_id=run_id, object_id=obj.object_id
        )
        res = runtime_handlers.pull_object(req, self.state, self._auth_task)

        # Assert object content is b"" (it was never pushed)
        assert res.object_found
        assert not res.object_available
        assert res.object_content == b""

        # Put object in store, then check it can be pulled
        self.store.put(object_id=obj.object_id, object_content=obj_b)
        req = PullObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID), run_id=run_id, object_id=obj.object_id
        )
        res = runtime_handlers.pull_object(req, self.state, self._auth_task)

        # Assert, identical object pulled
        assert res.object_found
        assert res.object_available
        assert obj_b == res.object_content

    def test_pull_object_fails(self) -> None:
        """Test `PullObject` in unsuported scenarios."""
        run_id = self._create_dummy_run(running=False)

        # Run is running but node ID isn't recognized
        self._transition_run_status(run_id, 2)
        req = PullObjectRequest(node=Node(node_id=123), run_id=run_id)
        with self.assertRaises(FlowerError) as error:
            runtime_handlers.pull_object(req, self.state, self._auth_task)
        assert error.exception.code == ApiErrorCode.RUNTIME_UNEXPECTED_NODE_ID

        # Attempt pulling object that doesn't exist
        req = PullObjectRequest(
            node=Node(node_id=SUPERLINK_NODE_ID), run_id=run_id, object_id="1234"
        )
        res = runtime_handlers.pull_object(req, self.state, self._auth_task)
        # Empty response
        assert not res.object_found

    def test_confirm_message_received_successful(self) -> None:
        """Test `ConfirmMessageReceived` success."""
        # Prepare
        run_id = self._create_dummy_run()
        proto = create_ins_message(
            src_node_id=SUPERLINK_NODE_ID, dst_node_id=self.node_id, run_id=run_id
        )
        message_ins = message_from_proto(proto)
        message_res = Message(
            RecordDict({"cfg": ConfigRecord({"key": "value"})}), reply_to=message_ins
        )

        # Prepare: Save reply message in ObjectStore
        all_objects = get_all_nested_objects(message_res)
        self.store.preregister(run_id, get_object_tree(message_res))
        for obj_id, obj in all_objects.items():
            self.store.put(object_id=obj_id, object_content=obj.deflate())

        # Assert: All objects are stored in the ObjectStore
        assert len(self.store) == len(all_objects)

        # Execute: Confirm message received
        request = ConfirmMessageReceivedRequest(
            node=Node(node_id=self.node_id),
            run_id=run_id,
            message_object_id=message_res.object_id,
        )
        response = runtime_handlers.confirm_message_received(
            request, self.state, self._auth_task
        )

        # Assert
        assert isinstance(response, ConfirmMessageReceivedResponse)

        # Assert: Message is removed from LinkState
        assert len(self.store) == 0

    def test_run_status_transitions(self) -> None:
        """Test `PullTaskInput` activates a claimed task and marks the run running."""
        # Prepare: Create a run with FAB
        fab_content = b"mock fab content"
        fab_hash = self.state.store_fab(
            Fab(hashlib.sha256(fab_content).hexdigest(), fab_content, {})
        )
        run_id = self._create_dummy_run(running=False, fab_hash=fab_hash)
        task_id = self._primary_task_id(run_id)
        # Claim task to transition the run to STARTING.
        claim_response = core_runtime_handlers.claim_task(
            ClaimTaskRequest(task_id=task_id), self.state
        )
        assert claim_response.HasField("token")

        # Set run series context as if it was persisted by an earlier run.
        run = self.state.get_run_info(run_ids=[run_id])[0]
        context = Context(
            123,
            SUPERLINK_NODE_ID,
            {},
            RecordDict(),
            {},
            series_id=run.series_id,
        )
        self.state.set_run_series_context(run.series_id, context)

        run_status = self.state.get_run_status({run_id})[run_id]
        assert run_status.status == Status.STARTING

        # Execute: Pull task input
        request = PullTaskInputRequest()
        task = self.state.get_tasks(task_ids=[task_id])[0]
        response = runtime_handlers.pull_task_input(request, self.state, task)

        # Assert: Response is successful and run status is now RUNNING
        assert isinstance(response, PullTaskInputResponse)
        assert response.context.run_id == run_id
        assert response.context.series_id == run.series_id
        run_status = self.state.get_run_status({run_id})[run_id]
        assert run_status.status == Status.RUNNING


def test_ha_pull_task_input_claim_is_unique_across_replicas() -> None:
    """Ensure only one replica can claim STARTING -> RUNNING via PullTaskInput."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _, task_id, state_0, state_1 = _create_shared_runtime(tmpdir)
        assert state_0.claim_task(task_id) is not None
        task = state_0.get_tasks(task_ids=[task_id])[0]

        results = _activate_in_parallel(state_0, state_1, task)

        assert results.count(True) == 1
        assert results.count(False) == 1
        task_status = state_0.get_tasks(task_ids=[task_id])[0].status
        assert task_status.status == Status.RUNNING
