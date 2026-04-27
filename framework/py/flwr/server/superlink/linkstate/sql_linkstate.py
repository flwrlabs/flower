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
from datetime import datetime, timezone
from logging import ERROR, WARNING
from typing import Any, Literal

from sqlalchemy import MetaData, text
from sqlalchemy.exc import IntegrityError

from flwr.app.user_config import UserConfig
from flwr.common import Context, Message, log, now
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
from flwr.common.typing import Run, RunStatus
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.server.utils.validator import validate_message
from flwr.supercore.constant import NodeStatus
from flwr.supercore.corestate.sql_corestate import SqlCoreState
from flwr.supercore.object_store.object_store import ObjectStore
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.state.schema.linkstate_tables import create_linkstate_metadata
from flwr.supercore.utils import (
    int64_to_uint64,
    simulation_config_from_json,
    simulation_config_to_json,
    uint64_to_int64,
)
from flwr.superlink.federation import FederationManager

from .linkstate import LinkState
from .utils import (
    check_node_availability_for_in_message,
    context_from_bytes,
    context_to_bytes,
    convert_sint64_values_in_dict_to_uint64,
    convert_uint64_values_in_dict_to_sint64,
    dict_to_message,
    generate_rand_int_from_bytes,
    message_to_dict,
    verify_found_message_replies,
    verify_message_ids,
)


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

        # Add corestate tables (for example token_store and fab)
        corestate_metadata = create_corestate_metadata()
        for table in corestate_metadata.tables.values():
            table.to_metadata(metadata)

        return metadata

    @property
    def federation_manager(self) -> FederationManager:
        """Return the FederationManager instance."""
        return self._federation_manager

    def store_message_ins(self, message: Message) -> str | None:
        """Store one Message."""
        # Validate message
        errors = validate_message(message=message, is_reply_message=False)
        if any(errors):
            log(ERROR, errors)
            return None

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

        with self.session():
            # Validate run_id
            query = "SELECT federation FROM run WHERE run_id = :run_id"
            rows = self.query(query, {"run_id": data[0]["run_id"]})
            if not rows:
                log(ERROR, "Invalid run ID for Message: %s", message.metadata.run_id)
                return None
            federation: str = rows[0]["federation"]

            # Validate destination node ID
            query = """SELECT node_id FROM node WHERE node_id = :node_id
                       AND status IN (:online, :offline)"""
            rows = self.query(
                query,
                {
                    "node_id": data[0]["dst_node_id"],
                    "online": NodeStatus.ONLINE,
                    "offline": NodeStatus.OFFLINE,
                },
            )
            if not rows or not self.federation_manager.has_node(
                message.metadata.dst_node_id, federation
            ):
                log(
                    ERROR,
                    "Invalid destination node ID for Message: %s",
                    message.metadata.dst_node_id,
                )
                return None

            # Insert message
            columns = ", ".join([f":{key}" for key in data[0]])
            query = f"INSERT INTO message_ins VALUES({columns})"

            # Only invalid run_id can trigger IntegrityError.
            # This may need to be changed in the future version
            # with more integrity checks.
            self.query(query, data[0])

        return message.metadata.message_id

    # pylint: disable-next=too-many-locals
    def _check_stored_messages(self, message_ids: set[str]) -> None:
        """Check and delete the message if it's invalid."""
        if not message_ids:
            return

        with self.session():
            # Batch fetch all messages in one query
            placeholders = ",".join([f":mid_{i}" for i in range(len(message_ids))])
            query = f"""
                SELECT * FROM message_ins
                WHERE message_id IN ({placeholders})
            """
            params = {f"mid_{i}": str(mid) for i, mid in enumerate(message_ids)}
            message_rows = self.query(query, params)

            if not message_rows:
                return

            # Build message lookup dict
            message_dict: dict[str, dict[str, Any]] = {
                row["message_id"]: row for row in message_rows
            }

            # Collect unique run_ids for batch federation lookup
            run_ids = {row["run_id"] for row in message_rows}
            placeholders = ",".join([f":rid_{i}" for i in range(len(run_ids))])
            query = f"""
                SELECT run_id, federation FROM run
                WHERE run_id IN ({placeholders})
            """
            params = {f"rid_{i}": rid for i, rid in enumerate(run_ids)}
            run_rows = self.query(query, params)

            # Build run_id to federation mapping
            run_id_to_federation: dict[int, str] = {
                row["run_id"]: row["federation"] for row in run_rows
            }

            invalid_msg_ids: set[str] = set()
            current_time = now().timestamp()

            # Check each message for validity
            for msg_id in message_ids:
                message_row = message_dict.get(msg_id)
                if not message_row:
                    continue

                # Check if the message has expired
                available_until = message_row["created_at"] + message_row["ttl"]
                if available_until <= current_time:
                    invalid_msg_ids.add(msg_id)
                    continue

                # Check if run exists and get federation
                run_id = message_row["run_id"]
                federation = run_id_to_federation.get(run_id)
                if not federation:
                    invalid_msg_ids.add(msg_id)
                    continue

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

        with self.session():
            rows = self._claim_message_ins_rows(node_id, limit)
            message_ids: set[str] = {row["message_id"] for row in rows}
            self._check_stored_messages(message_ids)

            # _check_stored_messages can delete claimed Messages if they became invalid
            # (for example, node removed from federation), so re-read current rows.
            if message_ids:
                rows = self._load_message_ins_rows(message_ids)

            for row in rows:
                # Convert values from sint64 to uint64
                convert_sint64_values_in_dict_to_uint64(
                    row, ["run_id", "src_node_id", "dst_node_id"]
                )

        result = [dict_to_message(dict(row)) for row in rows]

        return result

    def _claim_message_ins_rows(
        self, node_id: int, limit: int | None
    ) -> list[dict[str, Any]]:
        """Atomically claim eligible instruction Messages for a node."""
        current_time = now()
        params: dict[str, str | int | float] = {
            # Convert the uint64 value to sint64 for SQLite
            "node_id": uint64_to_int64(node_id),
            "current": current_time.timestamp(),
            "delivered_at": current_time.isoformat(),
        }
        common_condition = """
            dst_node_id = :node_id
            AND delivered_at = ''
            AND (created_at + ttl) > :current
        """
        condition = common_condition
        if limit is not None:
            condition = f"""
                message_id IN (
                    SELECT message_id
                    FROM message_ins
                    WHERE {common_condition}
                    ORDER BY rowid
                    LIMIT :limit
                )
                AND delivered_at = ''
            """
            params["limit"] = limit

        query = f"""
            UPDATE message_ins
            SET delivered_at = :delivered_at
            WHERE {condition}
            RETURNING *
        """
        return self.query(query, params)

    def _load_message_ins_rows(self, message_ids: set[str]) -> list[dict[str, Any]]:
        """Load instruction Messages by IDs."""
        placeholders = ",".join([f":mid_{i}" for i in range(len(message_ids))])
        query = f"""
            SELECT *
            FROM message_ins
            WHERE message_id IN ({placeholders})
            ORDER BY rowid
        """
        params = {f"mid_{i}": msg_id for i, msg_id in enumerate(message_ids)}
        return self.query(query, params)

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

        # Ensure that the dst_node_id of the original message matches the src_node_id
        # of reply being processed.
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
        msg_dict = message_to_dict(message)

        # Convert values from uint64 to sint64 for SQLite
        convert_uint64_values_in_dict_to_sint64(
            msg_dict, ["run_id", "src_node_id", "dst_node_id"]
        )

        columns = ", ".join([f":{key}" for key in msg_dict])
        query = f"INSERT INTO message_res VALUES({columns})"

        try:
            self.query(query, msg_dict)
        except IntegrityError:
            log(ERROR, "`run` is invalid")
            return None

        return message.metadata.message_id

    def get_message_res(self, message_ids: set[str]) -> list[Message]:
        """Get reply Messages for the given Message IDs."""
        # pylint: disable=too-many-locals
        ret: dict[str, Message] = {}

        with self.session():
            # Verify Message IDs
            self._check_stored_messages(message_ids)
            current = now().timestamp()
            placeholders = ",".join([f":mid_{i}" for i in range(len(message_ids))])
            query = f"""
                SELECT *
                FROM message_ins
                WHERE message_id IN ({placeholders})
            """
            params = {f"mid_{i}": str(mid) for i, mid in enumerate(message_ids)}
            rows = self.query(query, params)
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
            placeholders = ",".join([f":nid_{i}" for i in range(len(dst_node_ids))])
            query = f"""
                SELECT node_id, online_until
                FROM node
                WHERE node_id IN ({placeholders})
                AND status != :unregistered
            """
            node_params: dict[str, int | str] = {
                f"nid_{i}": nid for i, nid in enumerate(dst_node_ids)
            }
            node_params["unregistered"] = NodeStatus.UNREGISTERED
            rows = self.query(query, node_params)
            tmp_ret_dict = check_node_availability_for_in_message(
                inquired_in_message_ids=message_ids,
                found_in_message_dict=found_message_ins_dict,
                node_id_to_online_until={
                    int64_to_uint64(row["node_id"]): row["online_until"] for row in rows
                },
                current_time=current,
            )
            ret.update(tmp_ret_dict)

            # Atomically claim all eligible reply Messages
            placeholders = ",".join([f":mid_{i}" for i in range(len(message_ids))])
            delivered_at = now().isoformat()
            query = f"""
                UPDATE message_res
                SET delivered_at = :delivered_at
                WHERE reply_to_message_id IN ({placeholders})
                AND delivered_at = ''
                RETURNING *
            """
            params = {"delivered_at": delivered_at}
            params.update({f"mid_{i}": str(mid) for i, mid in enumerate(message_ids)})
            rows = self.query(query, params)
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

        return list(ret.values())

    def num_message_ins(self) -> int:
        """Calculate the number of instruction Messages in store.

        This includes delivered but not yet deleted.
        """
        query = "SELECT count(*) AS num FROM message_ins"
        rows = self.query(query, {})
        return int(rows[0]["num"])

    def num_message_res(self) -> int:
        """Calculate the number of reply Messages in store.

        This includes delivered but not yet deleted.
        """
        query = "SELECT count(*) AS num FROM message_res"
        rows = self.query(query)
        return int(rows[0]["num"])

    def delete_messages(self, message_ins_ids: set[str]) -> None:
        """Delete a Message and its reply based on provided Message IDs."""
        if not message_ins_ids:
            return

        placeholders = ",".join([f":mid_{i}" for i in range(len(message_ins_ids))])
        params = {f"mid_{i}": str(mid) for i, mid in enumerate(message_ins_ids)}

        # Delete Message
        query_1 = f"""
            DELETE FROM message_ins
            WHERE message_id IN ({placeholders})
        """

        # Delete reply Message
        query_2 = f"""
            DELETE FROM message_res
            WHERE reply_to_message_id IN ({placeholders})
        """

        with self.session():
            self.query(query_1, params)
            self.query(query_2, params)

    def get_message_ids_from_run_id(self, run_id: int) -> set[str]:
        """Get all instruction Message IDs for the given run_id."""
        query = """
            SELECT message_id
            FROM message_ins
            WHERE run_id = :run_id
        """
        sint64_run_id = uint64_to_int64(run_id)
        params = {"run_id": sint64_run_id}

        with self.session():
            rows = self.query(query, params)

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
            NODE_ID_NUM_BYTES, exclude={SUPERLINK_NODE_ID, 0}
        )

        # Convert the uint64 value to sint64 for SQLite
        sint64_node_id = uint64_to_int64(uint64_node_id)

        query = """
            INSERT INTO node
            (node_id, owner_aid, owner_name, status, registered_at, last_activated_at,
            last_deactivated_at, unregistered_at, online_until, heartbeat_interval,
            public_key)
            VALUES (:node_id, :owner_aid, :owner_name, :status, :registered_at,
            :last_activated_at, :last_deactivated_at, :unregistered_at, :online_until,
            :heartbeat_interval, :public_key)
        """

        # Mark the node online until now().timestamp() + heartbeat_interval
        try:
            self.query(
                query,
                {
                    "node_id": sint64_node_id,
                    "owner_aid": owner_aid,
                    "owner_name": owner_name,
                    "status": NodeStatus.REGISTERED,
                    "registered_at": now().isoformat(),
                    "last_activated_at": None,
                    "last_deactivated_at": None,
                    "unregistered_at": None,
                    "online_until": None,  # initialized with offline status
                    "heartbeat_interval": heartbeat_interval,
                    "public_key": public_key,
                },
            )
        except IntegrityError as e:
            if "node.public_key" in str(e):
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
            SET status = :unregistered, unregistered_at = :unregistered_at,
            online_until = IIF(online_until > :current, :current, online_until)
            WHERE node_id = :node_id AND status != :unregistered
            AND owner_aid = :owner_aid
            RETURNING node_id
        """
        current = now()
        params = {
            "unregistered": NodeStatus.UNREGISTERED,
            "unregistered_at": current.isoformat(),
            "current": current.timestamp(),
            "node_id": sint64_node_id,
            "owner_aid": owner_aid,
        }

        rows = self.query(query, params)
        if not rows:
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
        query = """
            UPDATE node
            SET status = :online,
                last_activated_at = :current,
                online_until = :online_until,
                heartbeat_interval = :heartbeat_interval
            WHERE node_id = :node_id AND status IN (:registered, :offline)
            RETURNING node_id
        """
        params = {
            "online": NodeStatus.ONLINE,
            "current": current_dt.isoformat(),
            "online_until": current_dt.timestamp()
            + HEARTBEAT_PATIENCE * heartbeat_interval,
            "heartbeat_interval": heartbeat_interval,
            "node_id": sint64_node_id,
            "registered": NodeStatus.REGISTERED,
            "offline": NodeStatus.OFFLINE,
        }

        rows = self.query(query, params)
        return len(rows) > 0

    def deactivate_node(self, node_id: int) -> bool:
        """Deactivate the node with the specified `node_id`."""
        self._check_and_tag_offline_nodes([node_id])

        # Only deactivate if the node is currently online
        current_dt = now()
        query = """
            UPDATE node
            SET status = :offline,
                last_deactivated_at = :current_iso,
                online_until = :current_ts
            WHERE node_id = :node_id AND status = :online
            RETURNING node_id
        """
        params = {
            "offline": NodeStatus.OFFLINE,
            "current_iso": current_dt.isoformat(),
            "current_ts": current_dt.timestamp(),
            "node_id": uint64_to_int64(node_id),
            "online": NodeStatus.ONLINE,
        }

        rows = self.query(query, params)
        return len(rows) > 0

    def get_nodes(self, run_id: int) -> set[int]:
        """Retrieve all currently stored node IDs as a set.

        Constraints
        -----------
        If the provided `run_id` does not exist or has no matching nodes,
        an empty `Set` MUST be returned.
        """
        with self.session():
            # Convert the uint64 value to sint64 for SQLite
            sint64_run_id = uint64_to_int64(run_id)

            # Validate run ID
            query = "SELECT federation FROM run WHERE run_id = :run_id"
            rows = self.query(query, {"run_id": sint64_run_id})
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
            UPDATE node SET status = :offline,
            last_deactivated_at =
            strftime('%Y-%m-%dT%H:%M:%f+00:00', online_until, 'unixepoch')
            WHERE online_until <= :current_time AND status = :online
        """
        params: dict[str, Any] = {
            "offline": NodeStatus.OFFLINE,
            "current_time": now().timestamp(),
            "online": NodeStatus.ONLINE,
        }
        if node_ids is not None:
            placeholders = ",".join([f":nid_{i}" for i in range(len(node_ids))])
            query += f" AND node_id IN ({placeholders})"
            params.update(
                {f"nid_{i}": uint64_to_int64(nid) for i, nid in enumerate(node_ids)}
            )
        self.query(query, params)

    def get_node_info(  # pylint: disable=too-many-locals
        self,
        *,
        node_ids: Sequence[int] | None = None,
        owner_aids: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
    ) -> Sequence[NodeInfo]:
        """Retrieve information about nodes based on the specified filters."""
        with self.session():
            self._check_and_tag_offline_nodes()

            # Build the WHERE clause based on provided filters
            conditions = []
            params: dict[str, Any] = {}
            if node_ids is not None:
                sint64_node_ids = [uint64_to_int64(node_id) for node_id in node_ids]
                placeholders = ",".join(
                    [f":nid_{i}" for i in range(len(sint64_node_ids))]
                )
                conditions.append(f"node_id IN ({placeholders})")
                for i, nid in enumerate(sint64_node_ids):
                    params[f"nid_{i}"] = nid
            if owner_aids is not None:
                placeholders = ",".join([f":aid_{i}" for i in range(len(owner_aids))])
                conditions.append(f"owner_aid IN ({placeholders})")
                for i, aid in enumerate(owner_aids):
                    params[f"aid_{i}"] = aid
            if statuses is not None:
                placeholders = ",".join([f":st_{i}" for i in range(len(statuses))])
                conditions.append(f"status IN ({placeholders})")
                for i, status in enumerate(statuses):
                    params[f"st_{i}"] = status

            # Construct the final query
            query = "SELECT * FROM node"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            rows = self.query(query, params)

            result: list[NodeInfo] = []
            for row in rows:
                # Convert sint64 node_id to uint64
                row["node_id"] = int64_to_uint64(row["node_id"])
                result.append(NodeInfo(**row))

            return result

    def get_node_id_by_public_key(self, public_key: bytes) -> int | None:
        """Get `node_id` for the specified `public_key` if it exists and is not
        deleted."""
        query = """SELECT node_id FROM node
                   WHERE public_key = :public_key AND status != :unregistered;"""
        rows = self.query(
            query, {"public_key": public_key, "unregistered": NodeStatus.UNREGISTERED}
        )

        # If no result is found, return None
        if not rows:
            return None

        # Convert sint64 node_id to uint64
        node_id = int64_to_uint64(rows[0]["node_id"])
        return node_id

    def create_run(  # pylint: disable=R0913, R0914, R0917
        self,
        fab_id: str | None,
        fab_version: str | None,
        fab_hash: str | None,
        override_config: UserConfig,
        federation: str,
        federation_config: SimulationConfig | None,
        flwr_aid: str | None,
        run_type: str,
    ) -> int:
        """Create a new run."""
        # Sample a random int64 as run_id
        uint64_run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)

        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(uint64_run_id)

        # Convert federation_config to JSON string for storage
        fed_config_json = None
        if federation_config:
            fed_config_json = json.dumps(simulation_config_to_json(federation_config))

        with self.session():
            # Check conflicts
            query = "SELECT COUNT(*) as cnt FROM run WHERE run_id = :run_id"
            rows = self.query(query, {"run_id": sint64_run_id})
            if rows[0]["cnt"] == 0:
                query = """
                    INSERT INTO run
                    (run_id, fab_id, fab_version, fab_hash, override_config, federation,
                    federation_config, run_type, pending_at, starting_at, running_at,
                    finished_at, usage_reported_at, sub_status, details, flwr_aid,
                    bytes_sent, bytes_recv, clientapp_runtime)
                    VALUES (:run_id, :fab_id, :fab_version, :fab_hash, :override_config,
                    :federation, :federation_config, :run_type, :pending_at,
                    :starting_at, :running_at, :finished_at, :usage_reported_at,
                    :sub_status, :details, :flwr_aid, :bytes_sent, :bytes_recv,
                    :clientapp_runtime)
                """
                override_config_json = json.dumps(override_config)
                params = {
                    "run_id": sint64_run_id,
                    "fab_id": fab_id or "",
                    "fab_version": fab_version or "",
                    "fab_hash": fab_hash or "",
                    "override_config": override_config_json,
                    "federation": federation,
                    "federation_config": fed_config_json,
                    "run_type": run_type,
                    "pending_at": now().isoformat(),
                    "starting_at": "",
                    "running_at": "",
                    "finished_at": "",
                    "sub_status": "",
                    "details": "",
                    "flwr_aid": flwr_aid or "",
                    "bytes_sent": 0,
                    "bytes_recv": 0,
                    "clientapp_runtime": 0.0,
                    "usage_reported_at": "",
                }
                self.query(query, params)
                return uint64_run_id
        log(ERROR, "Unexpected run creation failure.")
        return 0

    def get_run_info(  # pylint: disable=too-many-arguments, too-many-branches
        self,
        *,
        run_ids: Sequence[int] | None = None,
        statuses: Sequence[str] | None = None,
        flwr_aids: Sequence[str] | None = None,
        federations: Sequence[str] | None = None,
        order_by: Literal["pending_at"] | None = None,
        ascending: bool = True,
        limit: int | None = None,
    ) -> Sequence[Run]:
        """Retrieve information about runs based on the specified filters."""
        # Clean up expired tokens; this will flag inactive runs as needed
        self._cleanup_expired_tokens()

        # Build dynamic SQL filters:
        # - OR within each individual filter
        # - AND across different filters
        conditions = []
        params: dict[str, Any] = {}

        # Filter by run_ids
        if run_ids is not None:
            if not run_ids:
                return []
            sint64_run_ids = [uint64_to_int64(run_id) for run_id in run_ids]
            placeholders = ",".join([f":rid_{i}" for i in range(len(sint64_run_ids))])
            conditions.append(f"run_id IN ({placeholders})")
            params.update(
                {f"rid_{i}": run_id for i, run_id in enumerate(sint64_run_ids)}
            )

        # Filter by statuses
        # (maybe we should add a derived `status` column in the DB to simplify this?)
        if statuses is not None:
            if not statuses:
                return []
            status_conditions = []
            # Map logical run status to persisted timestamp state.
            if Status.PENDING in statuses:
                status_conditions.append("starting_at = '' AND finished_at = ''")
            if Status.STARTING in statuses:
                status_conditions.append(
                    "starting_at != '' AND running_at = '' AND finished_at = ''"
                )
            if Status.RUNNING in statuses:
                status_conditions.append("running_at != '' AND finished_at = ''")
            if Status.FINISHED in statuses:
                status_conditions.append("finished_at != ''")
            if not status_conditions:
                return []
            # SQL precedence: AND > OR, so A AND B OR C AND D == (A AND B) OR (C AND D)
            conditions.append(f"({' OR '.join(status_conditions)})")

        # Filter by Flower Account IDs
        if flwr_aids is not None:
            if not flwr_aids:
                return []
            placeholders = ",".join([f":aid_{i}" for i in range(len(flwr_aids))])
            conditions.append(f"flwr_aid IN ({placeholders})")
            params.update({f"aid_{i}": aid for i, aid in enumerate(flwr_aids)})

        # Filter by federations
        if federations is not None:
            if not federations:
                return []
            placeholders = ",".join([f":fed_{i}" for i in range(len(federations))])
            conditions.append(f"federation IN ({placeholders})")
            params.update({f"fed_{i}": fed for i, fed in enumerate(federations)})

        # Construct the final query
        query = "SELECT * FROM run"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        if order_by is not None:
            # `order_by` is a constrained Literal; safe to interpolate here.
            # ISO format(YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00) can be sorted
            # lexicographically to achieve correct chronological order.
            query += f" ORDER BY {order_by} {'ASC' if ascending else 'DESC'}"
        if limit is not None:
            query += " LIMIT :limit"
            params["limit"] = limit

        rows = self.query(query, params)
        # Convert DB rows into domain-level `Run` objects.
        return [
            Run(
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
                run_type=row["run_type"],
            )
            for row in rows
        ]

    def get_run_status(self, run_ids: set[int]) -> dict[int, RunStatus]:
        """Retrieve the statuses for the specified runs."""
        # Clean up expired tokens; this will flag inactive runs as needed
        self._cleanup_expired_tokens()

        # Convert the uint64 value to sint64 for SQLite
        placeholders = ",".join([f":rid_{i}" for i in range(len(run_ids))])
        query = f"SELECT * FROM run WHERE run_id IN ({placeholders})"
        params = {f"rid_{i}": uint64_to_int64(rid) for i, rid in enumerate(run_ids)}
        rows = self.query(query, params)

        return {
            # Restore uint64 run IDs
            int64_to_uint64(row["run_id"]): RunStatus(
                status=determine_run_status(row),
                sub_status=row["sub_status"],
                details=row["details"],
            )
            for row in rows
        }

    def get_federation_config(self, run_id: int) -> SimulationConfig | None:
        """Get the resolved federation configuration for the specified `run_id`."""
        query = "SELECT federation_config FROM run WHERE run_id = :run_id"
        sint64_run_id = uint64_to_int64(run_id)
        rows = self.query(query, {"run_id": sint64_run_id})
        if not rows:
            log(ERROR, "`run_id` invalid for fetching resolved federation config")
            return None

        fed_config_json = rows[0]["federation_config"]
        if fed_config_json is None:
            return None

        return simulation_config_from_json(json.loads(fed_config_json))

    def update_run_status(self, run_id: int, new_status: RunStatus) -> bool:
        """Update the status of the run with the specified `run_id`."""
        # Clean up expired tokens; this will flag inactive runs as needed
        self._cleanup_expired_tokens()

        # Determine the timestamp field and conditions based on the new status
        ts_fld = ""
        ts_con = ""
        if new_status.status == Status.STARTING:
            ts_fld = "starting_at"
            # Condition: current status is PENDING
            ts_con = "starting_at = '' AND finished_at = ''"
        elif new_status.status == Status.RUNNING:
            ts_fld = "running_at"
            # Condition: current status is STARTING
            ts_con = "starting_at != '' AND running_at = '' AND finished_at = ''"
        elif new_status.status == Status.FINISHED:
            ts_fld = "finished_at"
            # Condition: current status is not FINISHED
            ts_con = "finished_at = ''"
            if new_status.sub_status == SubStatus.COMPLETED:
                # For COMPLETED runs, ensure they are currently RUNNING
                ts_con += " AND running_at != ''"
        else:
            return False  # Cannot update to PENDING

        # Prepare the query and parameters
        query = f"""
            UPDATE run SET {ts_fld} = :timestamp,
            sub_status = :sub_status, details = :details
            WHERE run_id = :run_id AND {ts_con}
            RETURNING run_id
        """
        params = {
            "timestamp": now().isoformat(),
            "sub_status": new_status.sub_status,
            "details": new_status.details,
            "run_id": uint64_to_int64(run_id),
        }

        # Update the status
        rows = self.query(query, params)
        # Report usage if the run is marked as finished after the update
        if rows and new_status.status == Status.FINISHED:
            self.federation_manager.report_run_usage()
        return len(rows) > 0

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

        # Check if the node exists and is not unregistered
        query = """
            SELECT status FROM node WHERE node_id = :node_id AND status != :unregistered
        """
        rows = self.query(
            query, {"node_id": sint64_node_id, "unregistered": NodeStatus.UNREGISTERED}
        )
        if not rows:
            return False

        # Construct query and params
        current_dt = now()
        query = (
            "UPDATE node SET online_until = :online_until, "
            "heartbeat_interval = :heartbeat_interval"
        )
        params: dict[str, Any] = {
            "online_until": current_dt.timestamp()
            + HEARTBEAT_PATIENCE * heartbeat_interval,
            "heartbeat_interval": heartbeat_interval,
        }

        # Set timestamp if the status changes
        if rows[0]["status"] != NodeStatus.ONLINE:
            query += ", status = :online, last_activated_at = :last_activated_at"
            params["online"] = NodeStatus.ONLINE
            params["last_activated_at"] = current_dt.isoformat()

        # Execute the query, refreshing `online_until` and `heartbeat_interval`
        query += " WHERE node_id = :node_id"
        params["node_id"] = sint64_node_id
        self.query(query, params)
        return True

    def _on_tokens_expired(self, expired_records: list[tuple[int, float]]) -> None:
        """Handle cleanup of expired tokens.

        Override in subclasses to add custom cleanup logic.

        Parameters
        ----------
        expired_records : list[tuple[int, float]]
            List of tuples containing (run_id, active_until timestamp)
            for expired tokens.
        """
        if not expired_records:
            return

        with self.session() as session:
            query = """
                UPDATE run
                SET sub_status = :failed, details = :details, finished_at = :finished_at
                WHERE run_id = :run_id AND finished_at = ''
            """
            data = [
                {
                    "failed": SubStatus.FAILED,
                    "details": RUN_FAILURE_DETAILS_NO_HEARTBEAT,
                    "finished_at": datetime.fromtimestamp(
                        active_until, tz=timezone.utc
                    ).isoformat(),
                    "run_id": uint64_to_int64(run_id),
                }
                for run_id, active_until in expired_records
            ]
            result = session.execute(text(query), data)
            updated_rows = result.rowcount or 0  # type: ignore[attr-defined]

        # Report usage for runs that have been marked as failed due to expired tokens
        if updated_rows > 0:
            self.federation_manager.report_run_usage()

    def get_serverapp_context(self, run_id: int) -> Context | None:
        """Get the context for the specified `run_id`."""
        # Retrieve context if any
        query = "SELECT context FROM context WHERE run_id = :run_id"
        rows = self.query(query, {"run_id": uint64_to_int64(run_id)})
        context = context_from_bytes(rows[0]["context"]) if rows else None
        return context

    def set_serverapp_context(self, run_id: int, context: Context) -> None:
        """Set the context for the specified `run_id`."""
        # Convert context to bytes
        context_bytes = context_to_bytes(context)
        sint_run_id = uint64_to_int64(run_id)

        with self.session():
            # Check if any existing Context assigned to the run_id
            query = "SELECT COUNT(*) as count FROM context WHERE run_id = :run_id"
            row = self.query(query, {"run_id": sint_run_id})[0]
            if row["count"] > 0:
                # Update context
                query = """
                    UPDATE context
                    SET context = :context_bytes WHERE run_id = :run_id
                """
                self.query(
                    query, {"context_bytes": context_bytes, "run_id": sint_run_id}
                )
            else:
                try:
                    # Store context
                    query = (
                        "INSERT INTO context (run_id, context) "
                        "VALUES (:run_id, :context_bytes)"
                    )
                    self.query(
                        query, {"run_id": sint_run_id, "context_bytes": context_bytes}
                    )
                except IntegrityError:
                    raise ValueError(f"Run {run_id} not found") from None

    def add_serverapp_log(self, run_id: int, log_message: str) -> None:
        """Add a log entry to the ServerApp logs for the specified `run_id`."""
        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(run_id)

        # Store log
        try:
            query = """
                INSERT INTO logs (timestamp, run_id, node_id, log)
                VALUES (:current_ts, :run_id, :node_id, :log)
            """
            self.query(
                query,
                {
                    "current_ts": now().timestamp(),
                    "run_id": sint64_run_id,
                    "node_id": 0,
                    "log": log_message,
                },
            )
        except IntegrityError:
            raise ValueError(f"Run {run_id} not found") from None

    def get_serverapp_log(
        self, run_id: int, after_timestamp: float | None
    ) -> tuple[str, float]:
        """Get the ServerApp logs for the specified `run_id`."""
        # Convert the uint64 value to sint64 for SQLite
        sint64_run_id = uint64_to_int64(run_id)

        with self.session():
            # Check if the run_id exists
            query = "SELECT run_id FROM run WHERE run_id = :run_id"
            rows = self.query(query, {"run_id": sint64_run_id})
            if not rows:
                raise ValueError(f"Run {run_id} not found")

            # Retrieve logs
            if after_timestamp is None:
                after_timestamp = 0.0
            query = """
                SELECT log, timestamp FROM logs
                WHERE run_id = :run_id AND node_id = :node_id
                AND timestamp > :after_timestamp
                ORDER BY timestamp
            """
            rows = self.query(
                query,
                {
                    "run_id": sint64_run_id,
                    "node_id": 0,
                    "after_timestamp": after_timestamp,
                },
            )
            latest_timestamp = rows[-1]["timestamp"] if rows else 0.0
        return "".join(row["log"] for row in rows), latest_timestamp

    def get_valid_message_ins(self, message_id: str) -> dict[str, Any] | None:
        """Check if the Message exists and is valid (not expired).

        Return Message if valid.
        """
        with self.session():
            self._check_stored_messages({message_id})
            query = """
                SELECT *
                FROM message_ins
                WHERE message_id = :message_id
            """
            rows = self.query(query, {"message_id": message_id})
            if not rows:
                # Message does not exist
                return None

        return dict(rows[0])

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

        with self.session():
            # Check if run exists, performing the update only if it does
            update_query = """
                UPDATE run
                SET bytes_sent = bytes_sent + :bytes_sent,
                    bytes_recv = bytes_recv + :bytes_recv
                WHERE run_id = :run_id
                RETURNING run_id
            """
            rows = self.query(
                update_query,
                {
                    "bytes_sent": bytes_sent,
                    "bytes_recv": bytes_recv,
                    "run_id": sint64_run_id,
                },
            )

            if not rows:
                raise ValueError(f"Run {run_id} not found")

    def add_clientapp_runtime(self, run_id: int, runtime: float) -> None:
        """Add ClientApp runtime to the cumulative total for the specified `run_id`."""
        sint64_run_id = uint64_to_int64(run_id)
        with self.session():
            # Check if run exists, performing the update only if it does
            update_query = """
                UPDATE run
                SET clientapp_runtime = clientapp_runtime + :runtime
                WHERE run_id = :run_id
                RETURNING run_id
            """
            rows = self.query(
                update_query, {"runtime": runtime, "run_id": sint64_run_id}
            )

            if not rows:
                raise ValueError(f"Run {run_id} not found")


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
    raise ValueError(f"The run {run_id} does not have a valid status.")
