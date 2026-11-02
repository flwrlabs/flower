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
"""SQLAlchemy-based implementation of the link state."""

# pylint: disable=too-many-lines

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from logging import ERROR
from typing import Any, Literal, cast

from sqlalchemy import MetaData, bindparam, case, insert, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import aliased

from flwr.app import Message
from flwr.app.user_config import UserConfig
from flwr.common.constant import (
    HEARTBEAT_PATIENCE,
    NODE_ID_NUM_BYTES,
    RUN_ID_NUM_BYTES,
    SUPERLINK_NODE_ID,
    TASK_ID_NUM_BYTES,
    Status,
    SubStatus,
)
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskEvent  # pylint: disable=E0611
from flwr.supercore import log
from flwr.supercore.constant import NodeStatus, TaskType
from flwr.supercore.corestate.sql_corestate import SqlCoreState
from flwr.supercore.corestate.utils import timestamp_to_iso, validate_task_event_data
from flwr.supercore.date import now
from flwr.supercore.object_store.object_store import ObjectStore
from flwr.supercore.run import Run, RunStatus
from flwr.supercore.state.schema.corestate_models import Message as MessageModel
from flwr.supercore.state.schema.corestate_models import Task as TaskModel
from flwr.supercore.state.schema.corestate_models import TaskEvent as TaskEventModel
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.state.schema.linkstate_models import Node as NodeModel
from flwr.supercore.state.schema.linkstate_models import Run as RunModel
from flwr.supercore.state.schema.linkstate_tables import create_linkstate_metadata
from flwr.supercore.utils import (
    int64_to_uint64,
    simulation_config_from_json,
    simulation_config_to_json,
    uint64_to_int64,
)
from flwr.superlink.federation import FederationManager

from .linkstate import LinkState
from .utils import create_user_prompt_message, generate_rand_int_from_bytes


class SqlLinkState(LinkState, SqlCoreState):  # pylint: disable=R0904
    """SQLAlchemy-based LinkState implementation."""

    def __init__(
        self,
        database_path: str,
        federation_manager: FederationManager,
        object_store: ObjectStore,
    ) -> None:
        super().__init__(database_path, object_store)
        federation_manager.linkstate = self
        self._federation_manager = federation_manager

    def get_metadata(self) -> MetaData:
        """Return combined SQLAlchemy MetaData for LinkState and CoreState tables."""
        # Start with linkstate tables
        metadata = create_linkstate_metadata()

        # Add corestate tables (for example fab)
        corestate_metadata = create_corestate_metadata()
        for table in corestate_metadata.tables.values():
            table.to_metadata(metadata)

        return metadata

    @property
    def federation_manager(self) -> FederationManager:
        """Return the FederationManager instance."""
        return self._federation_manager

    def store_message_and_object_tree(
        self, message: Message, object_tree: ObjectTree, session_id: str
    ) -> tuple[bool, list[str]]:
        """Store a Message and preregister its ObjectTree."""
        with self.session() as session:
            if not self.store_messages([message])[0]:
                return False, []

            # Retries must not re-upload the body of an already cleared message.
            if session.scalar(
                select(MessageModel.cleared).where(
                    MessageModel.message_id == message.metadata.message_id
                )
            ):
                return True, []

            missing_objects = self.preregister_object_tree(object_tree, session_id)
            return True, missing_objects

    def stop_run(self, run_id: int) -> bool:
        """Stop a run and clean up run-scoped messages and objects."""
        # Check if the run exists
        runs = self.get_run_info(run_ids=[run_id])
        if not runs:
            return False

        # Stop the run's primary task, which will cascade to stop all its tasks
        primary_task_id = cast(int, runs[0].primary_task_id)
        if not self.finish_task(primary_task_id, SubStatus.STOPPED, ""):
            return False

        self.cleanup_run(run_id)
        return True

    def create_node(  # pylint: disable=too-many-arguments
        self,
        owner_aid: str,
        owner_name: str,
        public_key: bytes,
        heartbeat_interval: float,
        *,
        location: str | None = None,
        name: str | None = None,
    ) -> int:
        """Create, store in the link state, and return `node_id`."""
        # Sample a random uint64 as node_id
        uint64_node_id = generate_rand_int_from_bytes(
            NODE_ID_NUM_BYTES, exclude={SUPERLINK_NODE_ID, 0}
        )

        # Convert the uint64 value to sint64 for SQLite
        sint64_node_id = uint64_to_int64(uint64_node_id)

        # Mark the node online until now().timestamp() + heartbeat_interval
        try:
            with self.session() as session:
                session.execute(
                    insert(NodeModel).values(
                        node_id=sint64_node_id,
                        owner_aid=owner_aid,
                        owner_name=owner_name,
                        status=NodeStatus.REGISTERED,
                        registered_at=now().isoformat(),
                        last_activated_at=None,
                        last_deactivated_at=None,
                        unregistered_at=None,
                        online_until=None,  # initialized with offline status
                        heartbeat_interval=heartbeat_interval,
                        public_key=public_key,
                        location=location,
                        name=name,
                    )
                )
        except IntegrityError as e:
            # Check the underlying DB exception to distinguish constraint types.
            # - SQLite: str(e.orig) is e.g. "UNIQUE constraint failed: node.public_key"
            # - psycopg3: e.orig.diag.constraint_name contains the constraint name
            orig = e.orig
            constraint = getattr(getattr(orig, "diag", None), "constraint_name", None)
            is_pk_conflict = (
                "public_key" in constraint if constraint else "public_key" in str(orig)
            )
            if is_pk_conflict:
                raise ValueError("Public key already in use.") from None
            # Must be node ID conflict, almost impossible unless system is compromised
            log(ERROR, "Unexpected node registration failure.")
            return 0

        # Note: we need to return the uint64 value of the node_id
        return uint64_node_id

    def delete_node(self, owner_aid: str, node_id: int) -> None:
        """Delete a node."""
        sint64_node_id = uint64_to_int64(node_id)
        current = now()
        stmt = (
            update(NodeModel)
            .where(
                NodeModel.node_id == sint64_node_id,
                NodeModel.status != NodeStatus.UNREGISTERED,
                NodeModel.owner_aid == owner_aid,
            )
            .values(
                status=NodeStatus.UNREGISTERED,
                unregistered_at=current.isoformat(),
                online_until=case(
                    (NodeModel.online_until > current.timestamp(), current.timestamp()),
                    else_=NodeModel.online_until,
                ),
            )
            .returning(NodeModel.node_id)
        )
        with self.session() as session:
            updated_node_id = session.scalar(stmt)
        if updated_node_id is None:
            raise ValueError(
                f"Node {node_id} already deleted, not found or unauthorized "
                "deletion attempt."
            )

    def activate_node(self, node_id: int, heartbeat_interval: float) -> bool:
        """Activate the node with the specified `node_id`."""
        self._check_and_tag_offline_nodes([node_id])

        # Only activate if the node is currently registered or offline
        current_dt = now()
        sint64_node_id = uint64_to_int64(node_id)
        stmt = (
            update(NodeModel)
            .where(
                NodeModel.node_id == sint64_node_id,
                NodeModel.status.in_([NodeStatus.REGISTERED, NodeStatus.OFFLINE]),
            )
            .values(
                status=NodeStatus.ONLINE,
                last_activated_at=current_dt.isoformat(),
                online_until=current_dt.timestamp()
                + HEARTBEAT_PATIENCE * heartbeat_interval,
                heartbeat_interval=heartbeat_interval,
            )
            .returning(NodeModel.node_id)
        )
        with self.session() as session:
            return session.scalar(stmt) is not None

    def deactivate_node(self, node_id: int) -> bool:
        """Deactivate the node with the specified `node_id`."""
        self._check_and_tag_offline_nodes([node_id])

        # Only deactivate if the node is currently online
        current_dt = now()
        stmt = (
            update(NodeModel)
            .where(
                NodeModel.node_id == uint64_to_int64(node_id),
                NodeModel.status == NodeStatus.ONLINE,
            )
            .values(
                status=NodeStatus.OFFLINE,
                last_deactivated_at=current_dt.isoformat(),
                online_until=current_dt.timestamp(),
            )
            .returning(NodeModel.node_id)
        )
        with self.session() as session:
            return session.scalar(stmt) is not None

    def get_nodes(self, run_id: int) -> set[int]:
        """Retrieve all currently stored node IDs as a set.

        Constraints
        -----------
        If the provided `run_id` does not exist or has no matching nodes,
        an empty `Set` MUST be returned.
        """
        with self.session() as session:
            # Convert the uint64 value to sint64 for SQLite
            sint64_run_id = uint64_to_int64(run_id)

            # Validate run ID
            federation_id = session.scalar(
                select(RunModel.federation_id).where(RunModel.run_id == sint64_run_id)
            )
            if federation_id is None:
                return set()

            # Retrieve all online nodes
            node_ids = {
                node.node_id
                for node in self.get_node_info(statuses=[NodeStatus.ONLINE])
            }
        # Filter node IDs by federation
        return self.federation_manager.filter_nodes(node_ids, federation_id)

    def _check_and_tag_offline_nodes(self, node_ids: list[int] | None = None) -> None:
        """Check and tag offline nodes."""
        current_time = now().timestamp()
        stmt = select(NodeModel.node_id, NodeModel.online_until).where(
            NodeModel.online_until <= current_time,
            NodeModel.status == NodeStatus.ONLINE,
        )
        if node_ids is not None:
            if not node_ids:
                return
            sint64_node_ids = [uint64_to_int64(nid) for nid in node_ids]
            stmt = stmt.where(NodeModel.node_id.in_(sint64_node_ids))

        # Select candidate node_ids first so `last_deactivated_at` can preserve the
        # expiry time without relying on database-specific epoch formatting functions
        with self.session() as session:
            rows = session.execute(stmt).all()
            if not rows:
                return

            # Use one executemany UPDATE while keeping the state and expiry checks
            # in the statement to avoid overwriting a concurrent heartbeat.
            update_stmt = (
                update(NodeModel)
                .execution_options(dml_strategy="core_only")
                .where(
                    NodeModel.node_id == bindparam("offline_node_id"),
                    NodeModel.status == NodeStatus.ONLINE,
                    NodeModel.online_until <= bindparam("offline_current_time"),
                )
                .values(
                    status=NodeStatus.OFFLINE,
                    last_deactivated_at=bindparam("offline_deactivated_at"),
                )
            )
            update_params = [
                {
                    "offline_node_id": node_id,
                    "offline_current_time": current_time,
                    "offline_deactivated_at": datetime.fromtimestamp(
                        cast(float, online_until), tz=UTC
                    ).isoformat(),
                }
                for node_id, online_until in rows
            ]
            session.execute(update_stmt, update_params)

    def get_node_info(  # pylint: disable=too-many-locals
        self,
        *,
        node_ids: Sequence[int] | None = None,
        owner_aids: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
    ) -> Sequence[NodeInfo]:
        """Retrieve information about nodes based on the specified filters."""
        if node_ids is not None and len(node_ids) == 0:
            return []
        if owner_aids is not None and len(owner_aids) == 0:
            return []
        if statuses is not None and len(statuses) == 0:
            return []

        with self.session() as session:
            self._check_and_tag_offline_nodes()

            stmt = select(NodeModel).execution_options(populate_existing=True)
            if node_ids is not None:
                sint64_node_ids = [uint64_to_int64(node_id) for node_id in node_ids]
                stmt = stmt.where(NodeModel.node_id.in_(sint64_node_ids))
            if owner_aids is not None:
                stmt = stmt.where(NodeModel.owner_aid.in_(owner_aids))
            if statuses is not None:
                stmt = stmt.where(NodeModel.status.in_(statuses))

            return [
                _node_info_from_model(model) for model in session.scalars(stmt).all()
            ]

    def get_node_id_by_public_key(self, public_key: bytes) -> int | None:
        """Get `node_id` for the specified `public_key` if it exists and is not
        deleted."""
        stmt = select(NodeModel.node_id).where(
            NodeModel.public_key == public_key,
            NodeModel.status != NodeStatus.UNREGISTERED,
        )
        with self.session() as session:
            node_id = session.scalar(stmt)
        if node_id is None:
            return None

        # Convert sint64 node_id to uint64
        return int64_to_uint64(node_id)

    def create_run(  # pylint: disable=R0913, R0914, R0917
        self,
        fab_id: str | None,
        fab_version: str | None,
        fab_hash: str | None,
        override_config: UserConfig,
        federation_id: str,
        federation_config: SimulationConfig | None,
        flwr_aid: str | None,
        primary_task_type: str,
        series_id: int | None = None,
        series_description: str | None = None,
        connector_ids: Sequence[int] = (),
        initial_task_event: TaskEvent | None = None,
        user_prompt: str | None = None,
    ) -> int:
        """Create a new run."""
        if initial_task_event is not None:
            try:
                validate_task_event_data(initial_task_event.data)
            except ValueError:
                return 0
        # Convert federation_config to JSON string for storage
        fed_config_json = None
        if federation_config:
            fed_config_json = json.dumps(simulation_config_to_json(federation_config))

        override_config_json = json.dumps(override_config)
        run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)
        task_id = generate_rand_int_from_bytes(TASK_ID_NUM_BYTES)

        with self.session() as session:
            existing_run_id = session.scalar(
                select(RunModel.run_id).where(
                    RunModel.run_id == uint64_to_int64(run_id)
                )
            )
            if existing_run_id is None:
                current = now()
                resolved_series_id = self.store_run_in_series(
                    run_id=run_id,
                    federation_id=federation_id,
                    is_agent=primary_task_type == TaskType.AGENT_APP,
                    series_id=series_id,
                    description=series_description,
                )
                if resolved_series_id is None:
                    log(ERROR, "Unexpected run series membership failure.")
                    return 0
                self._refresh_run_series_context(
                    run_id=run_id,
                    series_id=resolved_series_id,
                )
                session.execute(
                    insert(RunModel).values(
                        run_id=uint64_to_int64(run_id),
                        fab_id=fab_id or "",
                        fab_version=fab_version or "",
                        fab_hash=fab_hash or "",
                        override_config=override_config_json,
                        federation_id=federation_id,
                        primary_task_id=uint64_to_int64(task_id),
                        federation_config=fed_config_json,
                        usage_reported_at="",
                        series_id=uint64_to_int64(resolved_series_id),
                        flwr_aid=flwr_aid or "",
                        bytes_sent=0,
                        bytes_recv=0,
                        clientapp_runtime=0.0,
                    )
                )
                session.execute(
                    insert(TaskModel).values(
                        task_id=uint64_to_int64(task_id),
                        type=primary_task_type,
                        run_id=uint64_to_int64(run_id),
                        fab_hash=fab_hash,
                        model_ref=None,
                        connector_ref=None,
                        token=None,
                        active_until=None,
                        pending_at=current,
                        starting_at=None,
                        running_at=None,
                        finished_at=None,
                        sub_status="",
                        details="",
                    )
                )
                if primary_task_type == TaskType.AGENT_APP and user_prompt is not None:
                    message = create_user_prompt_message(run_id, user_prompt)
                    self.store_messages([message])
                    self._store_generated_message(message)
                if initial_task_event is not None:
                    initial_task_event.run_id = run_id
                    initial_task_event.task_id = task_id
                    session.execute(
                        insert(TaskEventModel).values(
                            timestamp=current,
                            run_id=uint64_to_int64(run_id),
                            task_id=uint64_to_int64(task_id),
                            event=initial_task_event.event,
                            data=initial_task_event.data,
                        )
                    )
                self.bind_connectors_to_run(
                    run_id=run_id,
                    connector_ids=connector_ids,
                )
                return run_id

        log(ERROR, "Unexpected run creation failure.")
        return 0

    def get_run_info(  # pylint: disable=too-many-arguments, too-many-branches
        self,
        *,
        run_ids: Sequence[int] | None = None,
        statuses: Sequence[str] | None = None,
        flwr_aids: Sequence[str] | None = None,
        federation_ids: Sequence[str] | None = None,
        order_by: Literal["pending_at"] | None = None,
        ascending: bool = True,
        limit: int | None = None,
        skip: int = 0,
    ) -> Sequence[Run]:
        """Retrieve information about runs based on the specified filters."""
        self._cleanup_expired_task_tokens()
        stmt = (
            select(RunModel, TaskModel)
            .join(TaskModel, TaskModel.task_id == RunModel.primary_task_id)
            .execution_options(populate_existing=True)
        )

        # Filter by run_ids
        if run_ids is not None:
            if not run_ids:
                return []
            sint64_run_ids = [uint64_to_int64(run_id) for run_id in run_ids]
            stmt = stmt.where(RunModel.run_id.in_(sint64_run_ids))

        # Filter by statuses
        if statuses is not None:
            if not statuses:
                return []
            status_conditions = [
                _primary_task_status_filter(status)
                for status in (
                    Status.PENDING,
                    Status.STARTING,
                    Status.RUNNING,
                    Status.FINISHED,
                )
                if status in statuses
            ]
            if not status_conditions:
                return []
            stmt = stmt.where(or_(*status_conditions))

        # Filter by Flower Account IDs
        if flwr_aids is not None:
            if not flwr_aids:
                return []
            stmt = stmt.where(RunModel.flwr_aid.in_(flwr_aids))

        # Filter by federation IDs
        if federation_ids is not None:
            if not federation_ids:
                return []
            stmt = stmt.where(RunModel.federation_id.in_(federation_ids))

        if order_by is not None:
            order_column = TaskModel.pending_at
            stmt = stmt.order_by(
                order_column.asc() if ascending else order_column.desc(),
                RunModel.run_id.asc() if ascending else RunModel.run_id.desc(),
            )
        if skip:
            stmt = stmt.offset(skip)
        if limit is not None:
            stmt = stmt.limit(limit)

        with self.session() as session:
            rows = session.execute(stmt).all()
            return [
                _run_from_models(run_model, task_model)
                for run_model, task_model in rows
            ]

    def get_run_status(self, run_ids: set[int]) -> dict[int, RunStatus]:
        """Retrieve the statuses for the specified runs."""
        self._cleanup_expired_task_tokens()
        if not run_ids:
            return {}

        sint64_run_ids = [uint64_to_int64(rid) for rid in run_ids]
        stmt = (
            select(RunModel.run_id, TaskModel)
            .join(TaskModel, TaskModel.task_id == RunModel.primary_task_id)
            .where(RunModel.run_id.in_(sint64_run_ids))
            .execution_options(populate_existing=True)
        )
        with self.session() as session:
            rows = session.execute(stmt).all()
            return {
                int64_to_uint64(cast(int, run_id)): _run_status_from_task_model(task)
                for run_id, task in rows
            }

    def get_federation_config(self, run_id: int) -> SimulationConfig | None:
        """Get the resolved federation configuration for the specified `run_id`."""
        sint64_run_id = uint64_to_int64(run_id)
        with self.session() as session:
            row = session.execute(
                select(RunModel.federation_config).where(
                    RunModel.run_id == sint64_run_id
                )
            ).one_or_none()
        if row is None:
            log(ERROR, "`run_id` invalid for fetching resolved federation config")
            return None
        fed_config_json = row[0]
        if fed_config_json is None:
            return None

        return simulation_config_from_json(json.loads(fed_config_json))

    def _finish_run_tasks(
        self, run_primary_pairs: list[tuple[int, int]], sub_status: str, details: str
    ) -> None:
        """Finish all unfinished tasks of the run for the given run/primary-task pairs.

        The IDs must be sint64 DB values. Each task's ``finished_at`` is copied from
        its run's primary task.
        """
        if not run_primary_pairs:
            return

        sint_run_ids = [pair[0] for pair in run_primary_pairs]
        sint_task_ids = [pair[1] for pair in run_primary_pairs]
        primary_task = aliased(TaskModel)
        primary_finished_at = (
            select(primary_task.finished_at)
            .where(
                primary_task.task_id.in_(sint_task_ids),
                primary_task.run_id == TaskModel.run_id,
            )
            .correlate(TaskModel)
            .scalar_subquery()
        )
        stmt = (
            update(TaskModel)
            .where(
                TaskModel.run_id.in_(sint_run_ids),
                TaskModel.finished_at.is_(None),
            )
            .values(
                finished_at=primary_finished_at,
                sub_status=sub_status,
                details=details,
                active_until=None,
                token=None,
            )
        )
        with self.session() as session:
            session.execute(stmt)

    def finish_task(self, task_id: int, sub_status: str, details: str) -> bool:
        """Move an unfinished task to finished."""
        result = super().finish_task(task_id, sub_status, details)
        if result:
            sint64_task_id = uint64_to_int64(task_id)
            # Check whether this task is referenced as a run's primary task
            with self.session() as session:
                run_id = session.scalar(
                    select(RunModel.run_id).where(
                        RunModel.primary_task_id == sint64_task_id
                    )
                )
            if run_id is not None:
                # Stop all tasks of the run when the run is stopped
                if sub_status == SubStatus.STOPPED:
                    finish_sub_status = SubStatus.STOPPED
                    finish_details = "Task stopped because the run was stopped"
                # Otherwise, fail all tasks of the run
                else:
                    finish_sub_status = SubStatus.FAILED
                    finish_details = "Task failed because the run finished"
                self._finish_run_tasks(
                    [(run_id, sint64_task_id)],
                    sub_status=finish_sub_status,
                    details=finish_details,
                )
                self.federation_manager.report_run_usage()
        return result

    def _on_task_tokens_expired(self, tasks: list[Task]) -> None:
        """Fail unfinished tasks for runs whose primary task expired and report usage.

        When an expired task is the primary task of a run, this hook marks all
        unfinished tasks in that run as finished with FAILED status, removes any
        associated task tokens, and reports run usage.
        """
        if not tasks:
            return

        # Check if any of the expired tasks is referenced as a run's primary task.
        task_ids = [uint64_to_int64(task.task_id) for task in tasks]
        with self.session() as session:
            rows = session.execute(
                select(RunModel.run_id, RunModel.primary_task_id).where(
                    RunModel.primary_task_id.in_(task_ids)
                )
            ).all()
        if not rows:
            return

        # Fail any remaining tasks for expired runs
        self._finish_run_tasks(
            [(cast(int, run_id), primary_task_id) for run_id, primary_task_id in rows],
            sub_status=SubStatus.FAILED,
            details="Task failed because the run expired",
        )

        # Report usage for the run
        self.federation_manager.report_run_usage()

    def acknowledge_node_heartbeat(
        self, node_id: int, heartbeat_interval: float
    ) -> bool:
        """Acknowledge a heartbeat received from a node, serving as a heartbeat.

        A node is considered online as long as it sends heartbeats within
        the tolerated interval: HEARTBEAT_PATIENCE × heartbeat_interval.
        HEARTBEAT_PATIENCE = N allows for N-1 missed heartbeat before
        the node is marked as offline.
        """
        sint64_node_id = uint64_to_int64(node_id)

        current_dt = now()
        stmt = (
            update(NodeModel)
            .where(
                NodeModel.node_id == sint64_node_id,
                NodeModel.status != NodeStatus.UNREGISTERED,
            )
            .values(
                online_until=current_dt.timestamp()
                + HEARTBEAT_PATIENCE * heartbeat_interval,
                heartbeat_interval=heartbeat_interval,
                last_activated_at=case(
                    (
                        NodeModel.status != NodeStatus.ONLINE,
                        current_dt.isoformat(),
                    ),
                    else_=NodeModel.last_activated_at,
                ),
                status=NodeStatus.ONLINE,
            )
            .returning(NodeModel.node_id)
        )
        with self.session() as session:
            return session.scalar(stmt) is not None

    def store_traffic(self, run_id: int, *, bytes_sent: int, bytes_recv: int) -> None:
        """Store traffic data for the specified `run_id`."""
        # Validate non-negative values
        if bytes_sent < 0 or bytes_recv < 0:
            raise ValueError(
                f"Negative traffic values for run {run_id}: "
                f"bytes_sent={bytes_sent}, bytes_recv={bytes_recv}"
            )

        if bytes_sent == 0 and bytes_recv == 0:
            raise ValueError(
                f"Both bytes_sent and bytes_recv cannot be zero for run {run_id}"
            )

        sint64_run_id = uint64_to_int64(run_id)

        stmt = (
            update(RunModel)
            .where(RunModel.run_id == sint64_run_id)
            .values(
                bytes_sent=RunModel.bytes_sent + bytes_sent,
                bytes_recv=RunModel.bytes_recv + bytes_recv,
            )
            .returning(RunModel.run_id)
        )
        with self.session() as session:
            if session.scalar(stmt) is None:
                raise ValueError(f"Run {run_id} not found")

    def add_clientapp_runtime(self, run_id: int, runtime: float) -> None:
        """Add ClientApp runtime to the cumulative total for the specified `run_id`."""
        sint64_run_id = uint64_to_int64(run_id)
        stmt = (
            update(RunModel)
            .where(RunModel.run_id == sint64_run_id)
            .values(
                clientapp_runtime=RunModel.clientapp_runtime + runtime,
            )
            .returning(RunModel.run_id)
        )
        with self.session() as session:
            if session.scalar(stmt) is None:
                raise ValueError(f"Run {run_id} not found")


def _primary_task_status_filter(status: str) -> Any:
    """Return the ORM filter expression for a primary task status."""
    if status == Status.PENDING:
        return TaskModel.starting_at.is_(None) & TaskModel.finished_at.is_(None)
    if status == Status.STARTING:
        return (
            TaskModel.starting_at.is_not(None)
            & TaskModel.running_at.is_(None)
            & TaskModel.finished_at.is_(None)
        )
    if status == Status.RUNNING:
        return TaskModel.running_at.is_not(None) & TaskModel.finished_at.is_(None)
    if status == Status.FINISHED:
        return TaskModel.finished_at.is_not(None)
    raise ValueError(f"Unsupported task status {status!r}.")


def _run_status_from_task_model(task: TaskModel) -> RunStatus:
    """Determine a run status from its primary task model."""
    if task.pending_at:
        if task.finished_at:
            return RunStatus(
                status=Status.FINISHED,
                sub_status=task.sub_status,
                details=task.details,
            )
        if task.starting_at:
            if task.running_at:
                return RunStatus(status=Status.RUNNING, sub_status="", details="")
            return RunStatus(status=Status.STARTING, sub_status="", details="")
        return RunStatus(status=Status.PENDING, sub_status="", details="")
    task_id = int64_to_uint64(task.task_id)
    raise ValueError(f"The task {task_id} does not have a valid status.")


def _node_info_from_model(model: NodeModel) -> NodeInfo:
    """Convert a node model to a NodeInfo message."""
    return NodeInfo(
        node_id=int64_to_uint64(cast(int, model.node_id)),
        owner_aid=cast(str, model.owner_aid),
        owner_name=cast(str, model.owner_name),
        status=cast(str, model.status),
        registered_at=cast(str, model.registered_at),
        last_activated_at=model.last_activated_at,
        last_deactivated_at=model.last_deactivated_at,
        unregistered_at=model.unregistered_at,
        online_until=model.online_until,
        heartbeat_interval=cast(float, model.heartbeat_interval),
        public_key=cast(bytes, model.public_key),
        location=model.location,
        name=model.name,
    )


def _run_from_models(run: RunModel, task: TaskModel) -> Run:
    """Convert a run and its primary task models to a Run object."""
    return Run(
        run_id=int64_to_uint64(cast(int, run.run_id)),
        fab_id=cast(str, run.fab_id),
        fab_version=cast(str, run.fab_version),
        fab_hash=cast(str, run.fab_hash),
        override_config=json.loads(cast(str, run.override_config)),
        pending_at=timestamp_to_iso(task.pending_at),
        starting_at=timestamp_to_iso(task.starting_at),
        running_at=timestamp_to_iso(task.running_at),
        finished_at=timestamp_to_iso(task.finished_at),
        status=_run_status_from_task_model(task),
        flwr_aid=cast(str, run.flwr_aid),
        federation_id=cast(str, run.federation_id),
        primary_task_id=int64_to_uint64(run.primary_task_id),
        bytes_sent=cast(int, run.bytes_sent),
        bytes_recv=cast(int, run.bytes_recv),
        clientapp_runtime=cast(float, run.clientapp_runtime),
        primary_task_type=task.type,
        series_id=int64_to_uint64(run.series_id) if run.series_id else 0,
    )
