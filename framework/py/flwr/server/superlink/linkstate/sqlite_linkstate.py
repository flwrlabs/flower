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
"""SQLite based implemenation of the link state."""


# pylint: disable=too-many-lines

import json
import sqlite3
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from logging import ERROR, WARNING
from typing import Any, cast

from flwr.common import Context, Message, Metadata, log, now
from flwr.common.constant import (
    HEARTBEAT_PATIENCE,
    MESSAGE_TTL_TOLERANCE,
    NODE_ID_NUM_BYTES,
    RUN_FAILURE_DETAILS_NO_HEARTBEAT,
    RUN_ID_NUM_BYTES,
    SUPERLINK_NODE_ID,
    Status,
    SubStatus,
)
from flwr.common.message import make_message
from flwr.common.record import ConfigRecord, MetricRecord
from flwr.common.serde import recorddict_from_proto, recorddict_to_proto
from flwr.common.serde_utils import error_from_proto, error_to_proto
from flwr.common.typing import Run, RunStatus, UserConfig

# pylint: disable=E0611
from flwr.proto.error_pb2 import Error as ProtoError
from flwr.proto.node_pb2 import NodeInfo
from flwr.proto.recorddict_pb2 import RecordDict as ProtoRecordDict

# pylint: enable=E0611
from flwr.server.utils.validator import validate_message
from flwr.supercore.constant import NodeStatus
from flwr.supercore.corestate.sqlite_corestate import SqliteCoreState
from flwr.supercore.object_store.object_store import ObjectStore
from flwr.supercore.utils import int64_to_uint64, uint64_to_int64
from flwr.superlink.federation import FederationManager

from .linkstate import LinkState
from .utils import (
    check_node_availability_for_in_message,
    configrecord_from_bytes,
    configrecord_to_bytes,
    context_from_bytes,
    context_to_bytes,
    convert_sint64_values_in_dict_to_uint64,
    convert_uint64_values_in_dict_to_sint64,
    generate_rand_int_from_bytes,
    has_valid_sub_status,
    is_valid_transition,
    verify_found_message_replies,
    verify_message_ids,
)

SQL_CREATE_TABLE_NODE = """
CREATE TABLE IF NOT EXISTS node(
    node_id                 INTEGER UNIQUE,
    owner_aid               TEXT,
    owner_name              TEXT,
    status                  TEXT,
    registered_at           TEXT,
    last_activated_at       TEXT NULL,
    last_deactivated_at     TEXT NULL,
    unregistered_at         TEXT NULL,
    online_until            TIMESTAMP NULL,
    heartbeat_interval      REAL,
    public_key              BLOB UNIQUE
);
"""

SQL_CREATE_TABLE_PUBLIC_KEY = """
CREATE TABLE IF NOT EXISTS public_key(
    public_key      BLOB PRIMARY KEY
);
"""

SQL_CREATE_INDEX_ONLINE_UNTIL = """
CREATE INDEX IF NOT EXISTS idx_online_until ON node (online_until);
"""

SQL_CREATE_INDEX_OWNER_AID = """
CREATE INDEX IF NOT EXISTS idx_node_owner_aid ON node(owner_aid);
"""

SQL_CREATE_INDEX_NODE_STATUS = """
CREATE INDEX IF NOT EXISTS idx_node_status ON node(status);
"""

SQL_CREATE_TABLE_RUN = """
CREATE TABLE IF NOT EXISTS run(
    run_id                INTEGER UNIQUE,
    fab_id                TEXT,
    fab_version           TEXT,
    fab_hash              TEXT,
    override_config       TEXT,
    pending_at            TEXT,
    starting_at           TEXT,
    running_at            TEXT,
    finished_at           TEXT,
    sub_status            TEXT,
    details               TEXT,
    federation            TEXT,
    federation_options    BLOB,
    flwr_aid              TEXT,
    bytes_sent            INTEGER DEFAULT 0,
    bytes_recv            INTEGER DEFAULT 0,
    clientapp_runtime     REAL DEFAULT 0.0
);
"""

SQL_CREATE_TABLE_LOGS = """
CREATE TABLE IF NOT EXISTS logs (
    timestamp             REAL,
    run_id                INTEGER,
    node_id               INTEGER,
    log                   TEXT,
    PRIMARY KEY (timestamp, run_id, node_id),
    FOREIGN KEY (run_id) REFERENCES run(run_id)
);
"""

SQL_CREATE_TABLE_CONTEXT = """
CREATE TABLE IF NOT EXISTS context(
    run_id                INTEGER UNIQUE,
    context               BLOB,
    FOREIGN KEY(run_id) REFERENCES run(run_id)
);
"""

SQL_CREATE_TABLE_MESSAGE_INS = """
CREATE TABLE IF NOT EXISTS message_ins(
    message_id              TEXT UNIQUE,
    group_id                TEXT,
    run_id                  INTEGER,
    src_node_id             INTEGER,
    dst_node_id             INTEGER,
    reply_to_message_id     TEXT,
    created_at              REAL,
    delivered_at            TEXT,
    ttl                     REAL,
    message_type            TEXT,
    content                 BLOB NULL,
    error                   BLOB NULL,
    FOREIGN KEY(run_id) REFERENCES run(run_id)
);
"""


SQL_CREATE_TABLE_MESSAGE_RES = """
CREATE TABLE IF NOT EXISTS message_res(
    message_id              TEXT UNIQUE,
    group_id                TEXT,
    run_id                  INTEGER,
    src_node_id             INTEGER,
    dst_node_id             INTEGER,
    reply_to_message_id     TEXT,
    created_at              REAL,
    delivered_at            TEXT,
    ttl                     REAL,
    message_type            TEXT,
    content                 BLOB NULL,
    error                   BLOB NULL,
    FOREIGN KEY(run_id) REFERENCES run(run_id)
);
"""


class SqliteLinkState(LinkState, SqliteCoreState):  # pylint: disable=R0904
    """SQLite-based LinkState implementation."""

    def __init__(
        self,
        database_path: str,
        federation_manager: FederationManager,
        object_store: ObjectStore,
    ) -> None:
        super().__init__(database_path, object_store)
        federation_manager.linkstate = self
        self._federation_manager = federation_manager
        self._delivery_lock = threading.RLock()
        self._delivery_timings: dict[str, dict[str, float | None]] = {}

    def get_sql_statements(self) -> tuple[str, ...]:
        """Return SQL statements for LinkState tables."""
        return super().get_sql_statements() + (
            SQL_CREATE_TABLE_RUN,
            SQL_CREATE_TABLE_LOGS,
            SQL_CREATE_TABLE_CONTEXT,
            SQL_CREATE_TABLE_MESSAGE_INS,
            SQL_CREATE_TABLE_MESSAGE_RES,
            SQL_CREATE_TABLE_NODE,
            SQL_CREATE_TABLE_PUBLIC_KEY,
            SQL_CREATE_INDEX_ONLINE_UNTIL,
            SQL_CREATE_INDEX_OWNER_AID,
            SQL_CREATE_INDEX_NODE_STATUS,
        )

    @property
    def federation_manager(self) -> FederationManager:
        """Get the FederationManager instance."""
        return self._federation_manager

    def store_message_ins(self, message: Message) -> str | None:
        """Store one Message."""
        # Validate message
        errors = validate_message(message=message, is_reply_message=False)
        if any(errors):
            log(ERROR, errors)
            return None

        # Anchor reply enqueue in SuperLink clock for upstream delivery metrics.
        enqueued_at_ms = now().timestamp() * 1000.0
        message.metadata.created_at = enqueued_at_ms / 1000.0

        # Store Message
        data = (message_to_dict(message),)

        # Convert values from uint64 to sint64 for SQLite
        convert_uint64_values_in_dict_to_sint64(
            data[0], ["run_id", "src_node_id", "dst_node_id"]
        )

        # Validate source node ID
        if message.metadata.src_node_id != SUPERLINK_NODE_ID:
            log(
                ERROR,
                "Invalid source node ID for Message: %s",
                message.metadata.src_node_id,
            )
            return None

        with self.conn:
            # Validate run_id
            query = "SELECT federation FROM run WHERE run_id = ?;"
            rows = self.conn.execute(query, (data[0]["run_id"],)).fetchall()
            if not rows:
                log(ERROR, "Invalid run ID for Message: %s", message.metadata.run_id)
                return None
            federation: str = rows[0]["federation"]

            # Validate destination node ID
            query = "SELECT node_id FROM node WHERE node_id = ? AND status IN (?, ?);"
            rows = self.conn.execute(
                query, (data[0]["dst_node_id"], NodeStatus.ONLINE, NodeStatus.OFFLINE)
            ).fetchall()
            if not rows or not self.federation_manager.has_node(
                message.metadata.dst_node_id, federation
            ):
                log(
                    ERROR,
                    "Invalid destination node ID for Message: %s",
                    message.metadata.dst_node_id,
                )
                return None

            columns = ", ".join([f":{key}" for key in data[0]])
            query = f"INSERT INTO message_ins VALUES({columns});"

            # Only invalid run_id can trigger IntegrityError.
            # This may need to be changed in the future version
            # with more integrity checks.
            self.conn.execute(query, data[0])

        self.record_instruction_enqueued(
            message_id=message.metadata.message_id,
            enqueued_at_ms=now().timestamp() * 1000.0,
        )
        return message.metadata.message_id

    def _check_stored_messages(self, message_ids: set[str]) -> None:
        """Check and delete the message if it's invalid."""
        if not message_ids:
            return

        with self.conn:
            invalid_msg_ids: set[str] = set()
            current_time = now().timestamp()

            for msg_id in message_ids:
                # Check if message exists
                query = "SELECT * FROM message_ins WHERE message_id = ?;"
                message_row = self.conn.execute(query, (msg_id,)).fetchone()
                if not message_row:
                    continue

                # Check if the message has expired
                available_until = message_row["created_at"] + message_row["ttl"]
                if available_until <= current_time:
                    invalid_msg_ids.add(msg_id)
                    continue

                # Check if src_node_id and dst_node_id are in the federation
                # Get federation from run table
                run_id = message_row["run_id"]
                query = "SELECT federation FROM run WHERE run_id = ?;"
                run_row = self.conn.execute(query, (run_id,)).fetchone()
                if not run_row:  # This should not happen
                    invalid_msg_ids.add(msg_id)
                    continue
                federation = run_row["federation"]

                # Convert sint64 to uint64 for node IDs
                src_node_id = int64_to_uint64(message_row["src_node_id"])
                dst_node_id = int64_to_uint64(message_row["dst_node_id"])

                # Filter nodes to check if they're in the federation
                filtered = self.federation_manager.filter_nodes(
                    {src_node_id, dst_node_id}, federation
                )
                if len(filtered) != 2:  # Not both nodes are in the federation
                    invalid_msg_ids.add(msg_id)

            # Delete all invalid messages
            self.delete_messages(invalid_msg_ids)

    def get_message_ins(self, node_id: int, limit: int | None) -> list[Message]:
        """Get all Messages that have not been delivered yet."""
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        if node_id == SUPERLINK_NODE_ID:
            msg = f"`node_id` must be != {SUPERLINK_NODE_ID}"
            raise AssertionError(msg)

        data: dict[str, str | int] = {}

        # Convert the uint64 value to sint64 for SQLite
        data["node_id"] = uint64_to_int64(node_id)

        with self.conn:
            # Retrieve all Messages for node_id
            query = """
                SELECT message_id
                FROM message_ins
                WHERE   dst_node_id == :node_id
                AND   delivered_at = ""
                AND   (created_at + ttl) > CAST(strftime('%s', 'now') AS REAL)
            """

            if limit is not None:
                query += " LIMIT :limit"
                data["limit"] = limit

            query += ";"

            rows = self.conn.execute(query, data).fetchall()
            message_ids: set[str] = {row["message_id"] for row in rows}
            self._check_stored_messages(message_ids)

            # Mark retrieved Messages as delivered
            if rows:
                # Prepare query
                placeholders: str = ",".join(
                    [f":id_{i}" for i in range(len(message_ids))]
                )
                query = f"""
                    UPDATE message_ins
                    SET delivered_at = :delivered_at
                    WHERE message_id IN ({placeholders})
                    RETURNING *;
                """

                # Prepare data for query
                delivered_at = now().isoformat()
                data = {"delivered_at": delivered_at}
                for index, msg_id in enumerate(message_ids):
                    data[f"id_{index}"] = str(msg_id)

                # Run query
                rows = self.conn.execute(query, data).fetchall()

            for row in rows:
                # Convert values from sint64 to uint64
                convert_sint64_values_in_dict_to_uint64(
                    row, ["run_id", "src_node_id", "dst_node_id"]
                )

        result = [dict_to_message(row) for row in rows]

        return result

    def store_message_res(self, message: Message) -> str | None:
        """Store one Message."""
        # Validate message
        errors = validate_message(message=message, is_reply_message=True)
        if any(errors):
            log(ERROR, errors)
            return None

        res_metadata = message.metadata
        msg_ins_id = res_metadata.reply_to_message_id
        msg_ins = self.get_valid_message_ins(msg_ins_id)
        if msg_ins is None:
            log(
                ERROR,
                "Failed to store Message reply: "
                "The message it replies to with message_id %s does not exist or "
                "has expired, or was deleted because the target SuperNode was "
                "removed from the federation.",
                msg_ins_id,
            )
            return None

        # Ensure that the dst_node_id of the original message matches the src_node_id of
        # reply being processed.
        if (
            msg_ins
            and message
            and int64_to_uint64(msg_ins["dst_node_id"]) != res_metadata.src_node_id
        ):
            return None

        # Fail if the Message TTL exceeds the
        # expiration time of the Message it replies to.
        # Condition: ins_metadata.created_at + ins_metadata.ttl ≥
        #            res_metadata.created_at + res_metadata.ttl
        # A small tolerance is introduced to account
        # for floating-point precision issues.
        max_allowed_ttl = (
            msg_ins["created_at"] + msg_ins["ttl"] - res_metadata.created_at
        )
        if res_metadata.ttl and (
            res_metadata.ttl - max_allowed_ttl > MESSAGE_TTL_TOLERANCE
        ):
            log(
                WARNING,
                "Received Message with TTL %.2f exceeding the allowed maximum "
                "TTL %.2f.",
                res_metadata.ttl,
                max_allowed_ttl,
            )
            return None

        # Store Message
        data = (message_to_dict(message),)

        # Convert values from uint64 to sint64 for SQLite
        convert_uint64_values_in_dict_to_sint64(
            data[0], ["run_id", "src_node_id", "dst_node_id"]
        )

        columns = ", ".join([f":{key}" for key in data[0]])
        query = f"INSERT INTO message_res VALUES({columns});"

        # Only invalid run_id can trigger IntegrityError.
        # This may need to be changed in the future version with more integrity checks.
        try:
            self.query(query, data)
        except sqlite3.IntegrityError:
            log(ERROR, "`run` is invalid")
            return None

        self.record_reply_enqueued(msg_ins_id, enqueued_at_ms)
        return message.metadata.message_id

    def get_message_res(self, message_ids: set[str]) -> list[Message]:
        """Get reply Messages for the given Message IDs."""
        # pylint: disable-msg=too-many-locals
        ret: dict[str, Message] = {}

        with self.conn:
            # Verify Message IDs
            self._check_stored_messages(message_ids)
            current = now().timestamp()
            query = f"""
                SELECT *
                FROM message_ins
                WHERE message_id IN ({','.join(['?'] * len(message_ids))});
            """
            rows = self.conn.execute(
                query, tuple(str(message_id) for message_id in message_ids)
            ).fetchall()
            found_message_ins_dict: dict[str, Message] = {}
            for row in rows:
                convert_sint64_values_in_dict_to_uint64(
                    row, ["run_id", "src_node_id", "dst_node_id"]
                )
                found_message_ins_dict[row["message_id"]] = dict_to_message(row)

            ret = verify_message_ids(
                inquired_message_ids=message_ids,
                found_message_ins_dict=found_message_ins_dict,
                current_time=current,
            )

            # Check node availability
            dst_node_ids: set[int] = set()
            for message_id in message_ids:
                in_message = found_message_ins_dict[message_id]
                sint_node_id = uint64_to_int64(in_message.metadata.dst_node_id)
                dst_node_ids.add(sint_node_id)
            query = f"""
                SELECT node_id, online_until
                FROM node
                WHERE node_id IN ({','.join(['?'] * len(dst_node_ids))})
                AND status != ?
            """
            rows = self.conn.execute(
                query, tuple(dst_node_ids) + (NodeStatus.UNREGISTERED,)
            ).fetchall()
            tmp_ret_dict = check_node_availability_for_in_message(
                inquired_in_message_ids=message_ids,
                found_in_message_dict=found_message_ins_dict,
                node_id_to_online_until={
                    int64_to_uint64(row["node_id"]): row["online_until"] for row in rows
                },
                current_time=current,
            )
            ret.update(tmp_ret_dict)

            # Find all reply Messages
            query = f"""
                SELECT *
                FROM message_res
                WHERE reply_to_message_id IN ({','.join(['?'] * len(message_ids))})
                AND delivered_at = "";
            """
            rows = self.conn.execute(
                query, tuple(str(message_id) for message_id in message_ids)
            ).fetchall()
            for row in rows:
                convert_sint64_values_in_dict_to_uint64(
                    row, ["run_id", "src_node_id", "dst_node_id"]
                )
            tmp_ret_dict = verify_found_message_replies(
                inquired_message_ids=message_ids,
                found_message_ins_dict=found_message_ins_dict,
                found_message_res_list=[dict_to_message(row) for row in rows],
                current_time=current,
            )
            ret.update(tmp_ret_dict)

            # Mark existing reply Messages to be returned as delivered
            delivered_at_dt = now()
            delivered_at = delivered_at_dt.isoformat()
            delivered_at_ms = delivered_at_dt.timestamp() * 1000.0
            for message_res in ret.values():
                message_res.metadata.delivered_at = delivered_at
                ins_msg_id = message_res.metadata.reply_to_message_id
                self.record_serverapp_delivered(
                    run_id=message_res.metadata.run_id,
                    message_id=ins_msg_id,
                    delivered_at_ms=delivered_at_ms,
                )
                timings = self.get_delivery_timings(ins_msg_id)
                ins_enqueued_at_ms = timings.get("ins_enqueued_at_ms")
                clientapp_delivered_at_ms = timings.get("clientapp_delivered_at_ms")
                if (
                    ins_enqueued_at_ms is not None
                    and clientapp_delivered_at_ms is not None
                    and message_res.has_content()
                ):
                    downstream_ms = max(
                        float(clientapp_delivered_at_ms) - float(ins_enqueued_at_ms),
                        0.0,
                    )
                    metric_record = message_res.content.metric_records.get(
                        "_flwr_network_delivery"
                    )
                    if metric_record is None:
                        metric_record = MetricRecord()
                        message_res.content.metric_records[
                            "_flwr_network_delivery"
                        ] = metric_record
                    metric_record["downstream_ms"] = downstream_ms
            message_res_ids = [
                message_res.metadata.message_id for message_res in ret.values()
            ]
            query = f"""
                UPDATE message_res
                SET delivered_at = ?
                WHERE message_id IN ({','.join(['?'] * len(message_res_ids))});
            """
            data: list[Any] = [delivered_at] + message_res_ids
            self.conn.execute(query, data)

        return list(ret.values())

    def num_message_ins(self) -> int:
        """Calculate the number of instruction Messages in store.

        This includes delivered but not yet deleted.
        """
        query = "SELECT count(*) AS num FROM message_ins;"
        rows = self.query(query)
        result = rows[0]
        num = cast(int, result["num"])
        return num

    def num_message_res(self) -> int:
        """Calculate the number of reply Messages in store.

        This includes delivered but not yet deleted.
        """
        query = "SELECT count(*) AS num FROM message_res;"
        rows = self.query(query)
        result: dict[str, int] = rows[0]
        return result["num"]

    def delete_messages(self, message_ins_ids: set[str]) -> None:
        """Delete a Message and its reply based on provided Message IDs."""
        if not message_ins_ids:
            return
        if self.conn is None:
            raise AttributeError("LinkState not initialized")

        placeholders = ",".join(["?"] * len(message_ins_ids))
        data = tuple(str(message_id) for message_id in message_ins_ids)

        # Delete Message
        query_1 = f"""
            DELETE FROM message_ins
            WHERE message_id IN ({placeholders});
        """

        # Delete reply Message
        query_2 = f"""
            DELETE FROM message_res
            WHERE reply_to_message_id IN ({placeholders});
        """

        with self.conn:
            self.conn.execute(query_1, data)
            self.conn.execute(query_2, data)

    def get_message_ids_from_run_id(self, run_id: int) -> set[str]:
        """Get all instruction Message IDs for the given run_id."""
        if self.conn is None:
            raise AttributeError("LinkState not initialized")

        query = """
            SELECT message_id
            FROM message_ins
            WHERE run_id = :run_id;
        """

        sint64_run_id = uint64_to_int64(run_id)
        data = {"run_id": sint64_run_id}

        with self.conn:
            rows = self.conn.execute(query, data).fetchall()

        return {row["message_id"] for row in rows}

    def create_node(
        self,
        owner_aid: str,
        owner_name: str,
        public_key: bytes,
        heartbeat_interval: float,
    ) -> int:
        """Create, store in the link state, and return `node_id`."""
        # Sample a random uint64 as node_id
        uint64_node_id = generate_rand_int_from_bytes(
            NODE_ID_NUM_BYTES, exclude=[SUPERLINK_NODE_ID, 0]
        )

        # Convert the uint64 value to sint64 for SQLite
        sint64_node_id = uint64_to_int64(uint64_node_id)

        query = """
            INSERT INTO node
            (node_id, owner_aid, owner_name, status, registered_at, last_activated_at,
            last_deactivated_at, unregistered_at, online_until, heartbeat_interval,
            public_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        # Mark the node online until now().timestamp() + heartbeat_interval
        try:
            self.query(
                query,
                (
                    sint64_node_id,  # node_id
                    owner_aid,  # owner_aid
                    owner_name,  # owner_name
                    NodeStatus.REGISTERED,  # status
                    now().isoformat(),  # registered_at
                    None,  # last_activated_at
                    None,  # last_deactivated_at
                    None,  # unregistered_at
                    None,  # online_until, initialized with offline status
                    heartbeat_interval,  # heartbeat_interval
                    public_key,  # public_key
                ),
            )
        except sqlite3.IntegrityError as e:
            if "UNIQUE constraint failed: node.public_key" in str(e):
                raise ValueError("Public key already in use.") from None
            # Must be node ID conflict, almost impossible unless system is compromised
            log(ERROR, "Unexpected node registration failure.")
            return 0

        # Note: we need to return the uint64 value of the node_id
        return uint64_node_id

    def delete_node(self, owner_aid: str, node_id: int) -> None:
        """Delete a node."""
        sint64_node_id = uint64_to_int64(node_id)

        query = """
            UPDATE node
            SET status = ?, unregistered_at = ?,
            online_until = IIF(online_until > ?, ?, online_until)
            WHERE node_id = ? AND status != ? AND owner_aid = ?
            RETURNING node_id
        """
        current = now()
        params = (
            NodeStatus.UNREGISTERED,
            current.isoformat(),
            current.timestamp(),
            current.timestamp(),
            sint64_node_id,
            NodeStatus.UNREGISTERED,
            owner_aid,
        )

        rows = self.query(query, params)
        if not rows:
            raise ValueError(
                f"Node {node_id} already deleted, not found or unauthorized "
                "deletion attempt."
            )

    def activate_node(self, node_id: int, heartbeat_interval: float) -> bool:
        """Activate the node with the specified `node_id`."""
        with self.conn:
            self._check_and_tag_offline_nodes([node_id])

            # Only activate if the node is currently registered or offline
            current_dt = now()
            query = """
                UPDATE node
                SET status = ?,
                    last_activated_at = ?,
                    online_until = ?,
                    heartbeat_interval = ?
                WHERE node_id = ? AND status in (?, ?)
                RETURNING node_id
            """
            params = (
                NodeStatus.ONLINE,
                current_dt.isoformat(),
                current_dt.timestamp() + HEARTBEAT_PATIENCE * heartbeat_interval,
                heartbeat_interval,
                uint64_to_int64(node_id),
                NodeStatus.REGISTERED,
                NodeStatus.OFFLINE,
            )

            row = self.conn.execute(query, params).fetchone()
            return row is not None

    def deactivate_node(self, node_id: int) -> bool:
        """Deactivate the node with the specified `node_id`."""
        with self.conn:
            self._check_and_tag_offline_nodes([node_id])

            # Only deactivate if the node is currently online
            current_dt = now()
            query = """
                UPDATE node
                SET status = ?,
                    last_deactivated_at = ?,
                    online_until = ?
                WHERE node_id = ? AND status = ?
                RETURNING node_id
            """
            params = (
                NodeStatus.OFFLINE,
                current_dt.isoformat(),
                current_dt.timestamp(),
                uint64_to_int64(node_id),
                NodeStatus.ONLINE,
            )

            row = self.conn.execute(query, params).fetchone()
            return row is not None

    def get_nodes(self, run_id: int) -> set[int]:
        """Retrieve all currently stored node IDs as a set.

        Constraints
        -----------
        If the provided `run_id` does not exist or has no matching nodes,
        an empty `Set` MUST be returned.
        """
        if self.conn is None:
            raise AttributeError("LinkState not initialized")

        with self.conn:
            # Convert the uint64 value to sint64 for SQLite
            sint64_run_id = uint64_to_int64(run_id)

            # Validate run ID
            query = "SELECT federation FROM run WHERE run_id = ?"
            rows = self.conn.execute(query, (sint64_run_id,)).fetchall()
            if not rows:
                return set()
            federation: str = rows[0]["federation"]

            # Retrieve all online nodes
            node_ids = {
                node.node_id
                for node in self.get_node_info(statuses=[NodeStatus.ONLINE])
            }
        # Filter node IDs by federation
        return self.federation_manager.filter_nodes(node_ids, federation)

    def _check_and_tag_offline_nodes(self, node_ids: list[int] | None = None) -> None:
        """Check and tag offline nodes."""
        # strftime will convert POSIX timestamp to ISO format
        query = """
            UPDATE node SET status = ?,
            last_deactivated_at =
            strftime("%Y-%m-%dT%H:%M:%f+00:00", online_until, "unixepoch")
            WHERE online_until <= ? AND status == ?
        """
        params = [
            NodeStatus.OFFLINE,
            now().timestamp(),
            NodeStatus.ONLINE,
        ]
        if node_ids is not None:
            placeholders = ",".join(["?"] * len(node_ids))
            query += f" AND node_id IN ({placeholders})"
            params.extend(uint64_to_int64(node_id) for node_id in node_ids)
        self.conn.execute(query, params)

    def get_node_info(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        owner_aids: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
    ) -> Sequence[NodeInfo]:
        """Retrieve information about nodes based on the specified filters."""
        with self.conn:
            self._check_and_tag_offline_nodes()

            # Build the WHERE clause based on provided filters
            conditions = []
            params: list[Any] = []
            if node_ids is not None:
                sint64_node_ids = [uint64_to_int64(node_id) for node_id in node_ids]
                placeholders = ",".join(["?"] * len(sint64_node_ids))
                conditions.append(f"node_id IN ({placeholders})")
                params.extend(sint64_node_ids)
            if owner_aids is not None:
                placeholders = ",".join(["?"] * len(owner_aids))
                conditions.append(f"owner_aid IN ({placeholders})")
                params.extend(owner_aids)
            if statuses is not None:
                placeholders = ",".join(["?"] * len(statuses))
                conditions.append(f"status IN ({placeholders})")
                params.extend(statuses)

            # Construct the final query
            query = "SELECT * FROM node"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            rows = self.conn.execute(query, params).fetchall()

            result: list[NodeInfo] = []
            for row in rows:
                # Convert sint64 node_id to uint64
                row["node_id"] = int64_to_uint64(row["node_id"])
                result.append(NodeInfo(**row))

            return result

    def get_node_public_key(self, node_id: int) -> bytes:
        """Get `public_key` for the specified `node_id`."""
        # Convert the uint64 value to sint64 for SQLite
        sint64_node_id = uint64_to_int64(node_id)

        # Query the public key for the given node_id
        query = "SELECT public_key FROM node WHERE node_id = ? AND status != ?;"
        rows = self.query(query, (sint64_node_id, NodeStatus.UNREGISTERED))

        # If no result is found, return None
        if not rows:
            raise ValueError(f"Node ID {node_id} not found")

        # Return the public key
        return cast(bytes, rows[0]["public_key"])

    def get_node_id_by_public_key(self, public_key: bytes) -> int | None:
        """Get `node_id` for the specified `public_key` if it exists and is not
        deleted."""
        query = "SELECT node_id FROM node WHERE public_key = ? AND status != ?;"
        rows = self.query(query, (public_key, NodeStatus.UNREGISTERED))

        # If no result is found, return None
        if not rows:
            return None

        # Convert sint64 node_id to uint64
        node_id = int64_to_uint64(rows[0]["node_id"])
        return node_id

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def create_run(
        self,
        fab_id: str | None,
        fab_version: str | None,
        fab_hash: str | None,
        override_config: UserConfig,
        federation: str,
        federation_options: ConfigRecord,
        flwr_aid: str | None,
    ) -> int:
        """Create a new run."""
        # Sample a random int64 as run_id
        uint64_run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)

        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(uint64_run_id)

        with self.conn:
            # Check conflicts
            query = "SELECT COUNT(*) FROM run WHERE run_id = ?;"
            # If sint64_run_id does not exist
            row = self.conn.execute(query, (sint64_run_id,)).fetchone()
            if row["COUNT(*)"] == 0:
                query = """
                    INSERT INTO run
                    (run_id, fab_id, fab_version,
                    fab_hash, override_config, federation, federation_options,
                    pending_at, starting_at, running_at, finished_at, sub_status,
                    details, flwr_aid, bytes_sent, bytes_recv, clientapp_runtime)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """
                override_config_json = json.dumps(override_config)
                data = [
                    sint64_run_id,  # run_id
                    fab_id,  # fab_id
                    fab_version,  # fab_version
                    fab_hash,  # fab_hash
                    override_config_json,  # override_config
                    federation,  # federation
                    configrecord_to_bytes(federation_options),  # federation_options
                    now().isoformat(),  # pending_at
                    "",  # starting_at
                    "",  # running_at
                    "",  # finished_at
                    "",  # sub_status
                    "",  # details
                    flwr_aid or "",  # flwr_aid
                    0,  # bytes_sent
                    0,  # bytes_recv
                    0,  # clientapp_runtime
                ]
                self.conn.execute(query, tuple(data))
                return uint64_run_id
        log(ERROR, "Unexpected run creation failure.")
        return 0

    def get_run_ids(self, flwr_aid: str | None) -> set[int]:
        """Retrieve all run IDs if `flwr_aid` is not specified.

        Otherwise, retrieve all run IDs for the specified `flwr_aid`.
        """
        if flwr_aid:
            rows = self.query(
                "SELECT run_id FROM run WHERE flwr_aid = ?;",
                (flwr_aid,),
            )
        else:
            rows = self.query("SELECT run_id FROM run;", ())
        return {int64_to_uint64(row["run_id"]) for row in rows}

    def get_run(self, run_id: int) -> Run | None:
        """Retrieve information about the run with the specified `run_id`."""
        # Clean up expired tokens; this will flag inactive runs as needed
        self._cleanup_expired_tokens()

        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(run_id)
        query = "SELECT * FROM run WHERE run_id = ?;"
        rows = self.query(query, (sint64_run_id,))
        if rows:
            row = rows[0]
            return Run(
                run_id=int64_to_uint64(row["run_id"]),
                fab_id=row["fab_id"],
                fab_version=row["fab_version"],
                fab_hash=row["fab_hash"],
                override_config=json.loads(row["override_config"]),
                pending_at=row["pending_at"],
                starting_at=row["starting_at"],
                running_at=row["running_at"],
                finished_at=row["finished_at"],
                status=RunStatus(
                    status=determine_run_status(row),
                    sub_status=row["sub_status"],
                    details=row["details"],
                ),
                flwr_aid=row["flwr_aid"],
                federation=row["federation"],
                bytes_sent=row["bytes_sent"],
                bytes_recv=row["bytes_recv"],
                clientapp_runtime=row["clientapp_runtime"],
            )
        log(ERROR, "`run_id` does not exist.")
        return None

    def get_run_status(self, run_ids: set[int]) -> dict[int, RunStatus]:
        """Retrieve the statuses for the specified runs."""
        # Clean up expired tokens; this will flag inactive runs as needed
        self._cleanup_expired_tokens()

        # Convert the uint64 value to sint64 for SQLite
        sint64_run_ids = (uint64_to_int64(run_id) for run_id in set(run_ids))
        query = f"SELECT * FROM run WHERE run_id IN ({','.join(['?'] * len(run_ids))});"
        rows = self.query(query, tuple(sint64_run_ids))

        return {
            # Restore uint64 run IDs
            int64_to_uint64(row["run_id"]): RunStatus(
                status=determine_run_status(row),
                sub_status=row["sub_status"],
                details=row["details"],
            )
            for row in rows
        }

    def update_run_status(self, run_id: int, new_status: RunStatus) -> bool:
        """Update the status of the run with the specified `run_id`."""
        # Clean up expired tokens; this will flag inactive runs as needed
        self._cleanup_expired_tokens()

        with self.conn:
            # Convert the uint64 value to sint64 for SQLite
            sint64_run_id = uint64_to_int64(run_id)
            query = "SELECT * FROM run WHERE run_id = ?;"
            rows = self.conn.execute(query, (sint64_run_id,)).fetchall()

            # Check if the run_id exists
            if not rows:
                log(ERROR, "`run_id` is invalid")
                return False

            # Check if the status transition is valid
            row = rows[0]
            current_status = RunStatus(
                status=determine_run_status(row),
                sub_status=row["sub_status"],
                details=row["details"],
            )
            if not is_valid_transition(current_status, new_status):
                log(
                    ERROR,
                    'Invalid status transition: from "%s" to "%s"',
                    current_status.status,
                    new_status.status,
                )
                return False

            # Check if the sub-status is valid
            if not has_valid_sub_status(current_status):
                log(
                    ERROR,
                    'Invalid sub-status "%s" for status "%s"',
                    current_status.sub_status,
                    current_status.status,
                )
                return False

            # Update the status
            query = """
                UPDATE run SET %s= ?, sub_status = ?, details = ? WHERE run_id = ?;
            """

            # Prepare data for query
            current = now()

            # Determine the timestamp field based on the new status
            timestamp_fld = ""
            if new_status.status == Status.STARTING:
                timestamp_fld = "starting_at"
            elif new_status.status == Status.RUNNING:
                timestamp_fld = "running_at"
            elif new_status.status == Status.FINISHED:
                timestamp_fld = "finished_at"

            data = (
                current.isoformat(),
                new_status.sub_status,
                new_status.details,
                uint64_to_int64(run_id),
            )
            self.conn.execute(query % timestamp_fld, data)
        return True

    def get_pending_run_id(self) -> int | None:
        """Get the `run_id` of a run with `Status.PENDING` status, if any."""
        pending_run_id = None

        # Fetch all runs with unset `starting_at` (i.e. they are in PENDING status)
        query = "SELECT * FROM run WHERE starting_at = '' LIMIT 1;"
        rows = self.query(query)
        if rows:
            pending_run_id = int64_to_uint64(rows[0]["run_id"])

        return pending_run_id

    def get_federation_options(self, run_id: int) -> ConfigRecord | None:
        """Retrieve the federation options for the specified `run_id`."""
        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(run_id)
        query = "SELECT federation_options FROM run WHERE run_id = ?;"
        rows = self.query(query, (sint64_run_id,))

        # Check if the run_id exists
        if not rows:
            log(ERROR, "`run_id` is invalid")
            return None

        row = rows[0]
        return configrecord_from_bytes(row["federation_options"])

    def acknowledge_node_heartbeat(
        self, node_id: int, heartbeat_interval: float
    ) -> bool:
        """Acknowledge a heartbeat received from a node, serving as a heartbeat.

        A node is considered online as long as it sends heartbeats within
        the tolerated interval: HEARTBEAT_PATIENCE × heartbeat_interval.
        HEARTBEAT_PATIENCE = N allows for N-1 missed heartbeat before
        the node is marked as offline.
        """
        if self.conn is None:
            raise AttributeError("LinkState not initialized")

        sint64_node_id = uint64_to_int64(node_id)

        with self.conn:
            # Check if node exists and not deleted
            query = "SELECT status FROM node WHERE node_id = ? AND status != ?"
            row = self.conn.execute(
                query, (sint64_node_id, NodeStatus.UNREGISTERED)
            ).fetchone()
            if row is None:
                return False

            # Construct query and params
            current_dt = now()
            query = "UPDATE node SET online_until = ?, heartbeat_interval = ?"
            params: list[Any] = [
                current_dt.timestamp() + HEARTBEAT_PATIENCE * heartbeat_interval,
                heartbeat_interval,
            ]

            # Set timestamp if the status changes
            if row["status"] != NodeStatus.ONLINE:
                query += ", status = ?, last_activated_at = ?"
                params += [NodeStatus.ONLINE, current_dt.isoformat()]

            # Execute the query, refreshing `online_until` and `heartbeat_interval`
            query += " WHERE node_id = ?"
            params += [sint64_node_id]
            self.conn.execute(query, params)
            return True

    def _on_tokens_expired(self, expired_records: list[tuple[int, float]]) -> None:
        """Transition runs with expired tokens to failed status.

        Parameters
        ----------
        expired_records : list[tuple[int, float]]
            List of tuples containing (run_id, active_until timestamp)
            for expired tokens.
        """
        if not expired_records:
            return

        with self.conn:
            query = """
                UPDATE run
                SET sub_status = ?, details = ?, finished_at = ?
                WHERE run_id = ?;
            """
            data = [
                (
                    SubStatus.FAILED,
                    RUN_FAILURE_DETAILS_NO_HEARTBEAT,
                    datetime.fromtimestamp(active_until, tz=timezone.utc).isoformat(),
                    uint64_to_int64(run_id),
                )
                for run_id, active_until in expired_records
            ]
            self.conn.executemany(query, data)

    def get_serverapp_context(self, run_id: int) -> Context | None:
        """Get the context for the specified `run_id`."""
        # Retrieve context if any
        query = "SELECT context FROM context WHERE run_id = ?;"
        rows = self.query(query, (uint64_to_int64(run_id),))
        context = context_from_bytes(rows[0]["context"]) if rows else None
        return context

    def set_serverapp_context(self, run_id: int, context: Context) -> None:
        """Set the context for the specified `run_id`."""
        # Convert context to bytes
        context_bytes = context_to_bytes(context)
        sint_run_id = uint64_to_int64(run_id)

        with self.conn:
            # Check if any existing Context assigned to the run_id
            query = "SELECT COUNT(*) FROM context WHERE run_id = ?;"
            row = self.conn.execute(query, (sint_run_id,)).fetchone()
            if row["COUNT(*)"] > 0:
                # Update context
                query = "UPDATE context SET context = ? WHERE run_id = ?;"
                self.conn.execute(query, (context_bytes, sint_run_id))
            else:
                try:
                    # Store context
                    query = "INSERT INTO context (run_id, context) VALUES (?, ?);"
                    self.conn.execute(query, (sint_run_id, context_bytes))
                except sqlite3.IntegrityError:
                    raise ValueError(f"Run {run_id} not found") from None

    def add_serverapp_log(self, run_id: int, log_message: str) -> None:
        """Add a log entry to the ServerApp logs for the specified `run_id`."""
        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(run_id)

        # Store log
        try:
            query = """
                INSERT INTO logs (timestamp, run_id, node_id, log) VALUES (?, ?, ?, ?);
            """
            self.query(query, (now().timestamp(), sint64_run_id, 0, log_message))
        except sqlite3.IntegrityError:
            raise ValueError(f"Run {run_id} not found") from None

    def get_serverapp_log(
        self, run_id: int, after_timestamp: float | None
    ) -> tuple[str, float]:
        """Get the ServerApp logs for the specified `run_id`."""
        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(run_id)

        with self.conn:
            # Check if the run_id exists
            query = "SELECT run_id FROM run WHERE run_id = ?;"
            rows = self.conn.execute(query, (sint64_run_id,)).fetchall()
            if not rows:
                raise ValueError(f"Run {run_id} not found")

            # Retrieve logs
            if after_timestamp is None:
                after_timestamp = 0.0
            query = """
                SELECT log, timestamp FROM logs
                WHERE run_id = ? AND node_id = ? AND timestamp > ?;
            """
            rows = self.conn.execute(
                query, (sint64_run_id, 0, after_timestamp)
            ).fetchall()
            rows.sort(key=lambda x: x["timestamp"])
            latest_timestamp = rows[-1]["timestamp"] if rows else 0.0
        return "".join(row["log"] for row in rows), latest_timestamp

    def get_valid_message_ins(self, message_id: str) -> dict[str, Any] | None:
        """Check if the Message exists and is valid (not expired).

        Return Message if valid.
        """
        with self.conn:
            self._check_stored_messages({message_id})
            query = """
                SELECT *
                FROM message_ins
                WHERE message_id = :message_id
            """
            data = {"message_id": message_id}
            rows: list[dict[str, Any]] = self.conn.execute(query, data).fetchall()
            if not rows:
                # Message does not exist
                return None

        return rows[0]

    def store_traffic(self, run_id: int, *, bytes_sent: int, bytes_recv: int) -> None:
        """Store traffic data for the specified `run_id`."""
        # Validate non-negative values
        if bytes_sent < 0 or bytes_recv < 0:
            raise ValueError(
                f"Negative traffic values for run {run_id}: "
                f"bytes_sent={bytes_sent}, bytes_recv={bytes_recv}"
            )

        # Transport can report a zero-delta sample; skip these updates.
        if bytes_sent == 0 and bytes_recv == 0:
            return

        sint64_run_id = uint64_to_int64(run_id)

        with self.conn:
            # Check if run exists, performing the update only if it does
            update_query = """
                UPDATE run
                SET bytes_sent = bytes_sent + ?,
                    bytes_recv = bytes_recv + ?
                WHERE run_id = ?
                RETURNING run_id;
            """
            rows = self.conn.execute(
                update_query, (bytes_sent, bytes_recv, sint64_run_id)
            ).fetchall()

            if not rows:
                raise ValueError(f"Run {run_id} not found")

    def add_clientapp_runtime(self, run_id: int, runtime: float) -> None:
        """Add ClientApp runtime to the cumulative total for the specified `run_id`."""
        sint64_run_id = uint64_to_int64(run_id)
        with self.conn:
            # Check if run exists, performing the update only if it does
            update_query = """
                UPDATE run
                SET clientapp_runtime = clientapp_runtime + ?
                WHERE run_id = ?
                RETURNING run_id;
            """
            rows = self.conn.execute(update_query, (runtime, sint64_run_id)).fetchall()

            if not rows:
                raise ValueError(f"Run {run_id} not found")

    def record_instruction_enqueued(self, message_id: str, enqueued_at_ms: float) -> None:
        """Record when a ServerApp instruction is enqueued at SuperLink."""
        with self._delivery_lock:
            timings = self._delivery_timings.setdefault(
                message_id,
                {
                    "ins_enqueued_at_ms": None,
                    "clientapp_delivered_at_ms": None,
                    "res_enqueued_at_ms": None,
                    "serverapp_delivered_at_ms": None,
                },
            )
            timings["ins_enqueued_at_ms"] = enqueued_at_ms

    def record_reply_enqueued(self, message_id: str, enqueued_at_ms: float) -> None:
        """Record when a SuperNode reply is enqueued at SuperLink."""
        with self._delivery_lock:
            timings = self._delivery_timings.setdefault(
                message_id,
                {
                    "ins_enqueued_at_ms": None,
                    "clientapp_delivered_at_ms": None,
                    "res_enqueued_at_ms": None,
                    "serverapp_delivered_at_ms": None,
                },
            )
            timings["res_enqueued_at_ms"] = enqueued_at_ms

    def record_clientapp_delivered(
        self, run_id: int, message_id: str, delivered_at_ms: float
    ) -> None:
        """Record when an instruction is delivered to ClientApp runtime."""
        del run_id  # Unused in SQLite implementation.
        with self._delivery_lock:
            timings = self._delivery_timings.setdefault(
                message_id,
                {
                    "ins_enqueued_at_ms": None,
                    "clientapp_delivered_at_ms": None,
                    "res_enqueued_at_ms": None,
                    "serverapp_delivered_at_ms": None,
                },
            )
            timings["clientapp_delivered_at_ms"] = delivered_at_ms

    def record_serverapp_delivered(
        self, run_id: int, message_id: str, delivered_at_ms: float
    ) -> None:
        """Record when a reply is delivered to ServerApp runtime."""
        del run_id  # Unused in SQLite implementation.
        with self._delivery_lock:
            timings = self._delivery_timings.setdefault(
                message_id,
                {
                    "ins_enqueued_at_ms": None,
                    "clientapp_delivered_at_ms": None,
                    "res_enqueued_at_ms": None,
                    "serverapp_delivered_at_ms": None,
                },
            )
            timings["serverapp_delivered_at_ms"] = delivered_at_ms

    def get_delivery_timings(self, message_id: str) -> dict[str, float | None]:
        """Get delivery timing anchors for a specific instruction message ID."""
        with self._delivery_lock:
            timings = self._delivery_timings.get(message_id)
            if timings is None:
                return {
                    "ins_enqueued_at_ms": None,
                    "clientapp_delivered_at_ms": None,
                    "res_enqueued_at_ms": None,
                    "serverapp_delivered_at_ms": None,
                }
            return dict(timings)


def message_to_dict(message: Message) -> dict[str, Any]:
    """Transform Message to dict."""
    result = {
        "message_id": message.metadata.message_id,
        "group_id": message.metadata.group_id,
        "run_id": message.metadata.run_id,
        "src_node_id": message.metadata.src_node_id,
        "dst_node_id": message.metadata.dst_node_id,
        "reply_to_message_id": message.metadata.reply_to_message_id,
        "created_at": message.metadata.created_at,
        "delivered_at": message.metadata.delivered_at,
        "ttl": message.metadata.ttl,
        "message_type": message.metadata.message_type,
        "content": None,
        "error": None,
    }

    if message.has_content():
        result["content"] = recorddict_to_proto(message.content).SerializeToString()
    else:
        result["error"] = error_to_proto(message.error).SerializeToString()

    return result


def dict_to_message(message_dict: dict[str, Any]) -> Message:
    """Transform dict to Message."""
    content, error = None, None
    if (b_content := message_dict.pop("content")) is not None:
        content = recorddict_from_proto(ProtoRecordDict.FromString(b_content))
    if (b_error := message_dict.pop("error")) is not None:
        error = error_from_proto(ProtoError.FromString(b_error))

    # Metadata constructor doesn't allow passing created_at. We set it later
    metadata = Metadata(
        **{k: v for k, v in message_dict.items() if k not in ["delivered_at"]}
    )
    msg = make_message(metadata=metadata, content=content, error=error)
    msg.metadata.delivered_at = message_dict["delivered_at"]
    return msg


def determine_run_status(row: dict[str, Any]) -> str:
    """Determine the status of the run based on timestamp fields."""
    if row["pending_at"]:
        if row["finished_at"]:
            return Status.FINISHED
        if row["starting_at"]:
            if row["running_at"]:
                return Status.RUNNING
            return Status.STARTING
        return Status.PENDING
    run_id = int64_to_uint64(row["run_id"])
    raise sqlite3.IntegrityError(f"The run {run_id} does not have a valid status.")
