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
"""In-memory LinkState implementation."""


import threading
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import ERROR, WARNING
from typing import Literal

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
from flwr.server.superlink.linkstate.linkstate import LinkState
from flwr.server.utils import validate_message
from flwr.supercore.constant import NodeStatus
from flwr.supercore.corestate.in_memory_corestate import InMemoryCoreState
from flwr.supercore.object_store.object_store import ObjectStore
from flwr.superlink.federation import FederationManager

from .utils import (
    check_node_availability_for_in_message,
    generate_rand_int_from_bytes,
    has_valid_sub_status,
    is_valid_transition,
    verify_found_message_replies,
    verify_message_ids,
)


@dataclass
class RunRecord:  # pylint: disable=R0902
    """The record of a specific run, including its status and timestamps."""

    run: Run
    federation_config: SimulationConfig | None = None
    logs: list[tuple[float, str]] = field(default_factory=list)
    usage_reported_at: str = ""
    log_lock: threading.Lock = field(default_factory=threading.Lock)
    lock: threading.RLock = field(default_factory=threading.RLock)


class InMemoryLinkState(LinkState, InMemoryCoreState):  # pylint: disable=R0902,R0904
    """In-memory LinkState implementation."""

    def __init__(
        self, federation_manager: FederationManager, object_store: ObjectStore
    ) -> None:
        super().__init__(object_store)

        # Map node_id to NodeInfo
        self.nodes: dict[int, NodeInfo] = {}
        self.node_public_key_to_node_id: dict[bytes, int] = {}
        self.owner_to_node_ids: dict[str, set[int]] = {}  # Quick lookup

        # Map run_id to RunRecord
        self.run_ids: dict[int, RunRecord] = {}
        self.contexts: dict[int, Context] = {}
        self.message_ins_store: dict[str, Message] = {}
        self.message_res_store: dict[str, Message] = {}
        self.message_ins_id_to_message_res_id: dict[str, str] = {}

        # Map flwr_aid to run_ids for O(1) reverse index lookup
        self.flwr_aid_to_run_ids: dict[str, set[int]] = defaultdict(set)

        self.node_public_keys: set[bytes] = set()

        self.lock = threading.RLock()
        federation_manager.linkstate = self
        self._federation_manager = federation_manager

    @property
    def federation_manager(self) -> FederationManager:
        """Get the FederationManager instance."""
        return self._federation_manager

    def store_message_ins(self, message: Message) -> str | None:
        """Store one Message."""
        # Validate message
        errors = validate_message(message, is_reply_message=False)
        if any(errors):
            log(ERROR, errors)
            return None
        # Validate run_id
        if message.metadata.run_id not in self.run_ids:
            log(ERROR, "Invalid run ID for Message: %s", message.metadata.run_id)
            return None
        federation = self.run_ids[message.metadata.run_id].run.federation
        # Validate source node ID
        if message.metadata.src_node_id != SUPERLINK_NODE_ID:
            log(
                ERROR,
                "Invalid source node ID for Message: %s",
                message.metadata.src_node_id,
            )
            return None
        # Validate destination node ID
        dst_node = self.nodes.get(message.metadata.dst_node_id)
        if (
            # Node must exist
            dst_node is None
            # Node must be online or offline
            or dst_node.status not in (NodeStatus.ONLINE, NodeStatus.OFFLINE)
            # Node must belong to the same federation
            or not self.federation_manager.has_node(dst_node.node_id, federation)
        ):
            log(
                ERROR,
                "Invalid destination node ID for Message: %s",
                message.metadata.dst_node_id,
            )
            return None

        message_id = message.metadata.message_id
        with self.lock:
            self.message_ins_store[message_id] = message

        # Return the new message_id
        return message_id

    def _check_stored_messages(self, message_ids: set[str]) -> None:
        """Check and delete the message if it's invalid."""
        with self.lock:
            invalid_msg_ids: set[str] = set()
            current = now().timestamp()
            for msg_id in message_ids:
                if not (message := self.message_ins_store.get(msg_id)):
                    continue

                # Check if the message has expired
                available_until = message.metadata.created_at + message.metadata.ttl
                if available_until <= current:
                    invalid_msg_ids.add(msg_id)
                    continue

                # Check if the destination node and the source node are still in the
                # same federation
                src_node_id = message.metadata.src_node_id
                dst_node_id = message.metadata.dst_node_id
                filtered = self.federation_manager.filter_nodes(
                    {src_node_id, dst_node_id},
                    self.run_ids[message.metadata.run_id].run.federation,
                )
                if len(filtered) != 2:  # Not both nodes are in the federation
                    invalid_msg_ids.add(msg_id)

            # Delete all invalid messages
            self.delete_messages(invalid_msg_ids)

    def get_message_ins(self, node_id: int, limit: int | None) -> list[Message]:
        """Get all Messages that have not been delivered yet."""
        if limit is not None and limit < 1:
            raise AssertionError("`limit` must be >= 1")

        # Find Message for node_id that were not delivered yet
        message_ins_list: list[Message] = []
        with self.lock:
            for msg_id in list(self.message_ins_store.keys()):
                self._check_stored_messages({msg_id})

                if (
                    (msg_ins := self.message_ins_store.get(msg_id))
                    and msg_ins.metadata.dst_node_id == node_id
                    and msg_ins.metadata.delivered_at == ""
                ):
                    message_ins_list.append(msg_ins)
                if limit and len(message_ins_list) == limit:
                    break

        # Mark all of them as delivered
        delivered_at = now().isoformat()
        for msg_ins in message_ins_list:
            msg_ins.metadata.delivered_at = delivered_at

        # Return list of messages
        return message_ins_list

    # pylint: disable=R0911
    def store_message_res(self, message: Message) -> str | None:
        """Store one Message."""
        # Validate message
        errors = validate_message(message, is_reply_message=True)
        if any(errors):
            log(ERROR, errors)
            return None

        res_metadata = message.metadata
        with self.lock:
            # Check if the Message it is replying to exists and is valid
            msg_ins_id = res_metadata.reply_to_message_id
            self._check_stored_messages({msg_ins_id})
            msg_ins = self.message_ins_store.get(msg_ins_id)

            # Ensure that dst_node_id of original Message matches the src_node_id of
            # reply Message.
            if (
                msg_ins
                and message
                and msg_ins.metadata.dst_node_id != res_metadata.src_node_id
            ):
                return None

            if msg_ins is None:
                log(
                    ERROR,
                    "Message with ID %s does not exist.",
                    msg_ins_id,
                )
                return None

            # Fail if the Message TTL exceeds the
            # expiration time of the Message it replies to.
            # Condition: ins_metadata.created_at + ins_metadata.ttl ≥
            #            res_metadata.created_at + res_metadata.ttl
            # A small tolerance is introduced to account
            # for floating-point precision issues.
            ins_metadata = msg_ins.metadata
            max_allowed_ttl = (
                ins_metadata.created_at + ins_metadata.ttl - res_metadata.created_at
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

        # Validate run_id
        if res_metadata.run_id != ins_metadata.run_id:
            log(ERROR, "`metadata.run_id` is invalid")
            return None

        message_id = message.metadata.message_id
        with self.lock:
            self.message_res_store[message_id] = message
            self.message_ins_id_to_message_res_id[msg_ins_id] = message_id

        # Return the new message_id
        return message_id

    def get_message_res(self, message_ids: set[str]) -> list[Message]:
        """Get reply Messages for the given Message IDs."""
        ret: dict[str, Message] = {}

        with self.lock:
            self._check_stored_messages(message_ids)
            current = now().timestamp()

            # Verify Message IDs
            ret = verify_message_ids(
                inquired_message_ids=message_ids,
                found_message_ins_dict=self.message_ins_store,
                current_time=current,
            )

            # Check node availability
            dst_node_ids = {
                self.message_ins_store[message_id].metadata.dst_node_id
                for message_id in message_ids
            }
            tmp_ret_dict = check_node_availability_for_in_message(
                inquired_in_message_ids=message_ids,
                found_in_message_dict=self.message_ins_store,
                node_id_to_online_until={
                    node_id: self.nodes[node_id].online_until
                    for node_id in dst_node_ids
                    if node_id in self.nodes
                    and self.nodes[node_id].status != NodeStatus.UNREGISTERED
                },
                current_time=current,
            )
            ret.update(tmp_ret_dict)

            # Find all reply Messages
            message_res_found: list[Message] = []
            for message_id in message_ids:
                # If Message exists and is not delivered, add it to the list
                if message_res_id := self.message_ins_id_to_message_res_id.get(
                    message_id
                ):
                    message_res = self.message_res_store[message_res_id]
                    if message_res.metadata.delivered_at == "":
                        message_res_found.append(message_res)
            tmp_ret_dict = verify_found_message_replies(
                inquired_message_ids=message_ids,
                found_message_ins_dict=self.message_ins_store,
                found_message_res_list=message_res_found,
                current_time=current,
            )
            ret.update(tmp_ret_dict)

            # Mark existing reply Messages to be returned as delivered
            delivered_at = now().isoformat()
            for message_res in message_res_found:
                message_res.metadata.delivered_at = delivered_at

        return list(ret.values())

    def delete_messages(self, message_ins_ids: set[str]) -> None:
        """Delete a Message and its reply based on provided Message IDs."""
        if not message_ins_ids:
            return

        with self.lock:
            for message_id in message_ins_ids:
                # Delete Messages
                if message_id in self.message_ins_store:
                    del self.message_ins_store[message_id]
                # Delete Message replies
                if message_id in self.message_ins_id_to_message_res_id:
                    message_res_id = self.message_ins_id_to_message_res_id.pop(
                        message_id
                    )
                    del self.message_res_store[message_res_id]

    def get_message_ids_from_run_id(self, run_id: int) -> set[str]:
        """Get all instruction Message IDs for the given run_id."""
        message_id_list: set[str] = set()
        with self.lock:
            for message_id, message in self.message_ins_store.items():
                if message.metadata.run_id == run_id:
                    message_id_list.add(message_id)

        return message_id_list

    def num_message_ins(self) -> int:
        """Calculate the number of instruction Messages in store.

        This includes delivered but not yet deleted.
        """
        return len(self.message_ins_store)

    def num_message_res(self) -> int:
        """Calculate the number of reply Messages in store.

        This includes delivered but not yet deleted.
        """
        return len(self.message_res_store)

    def create_node(
        self,
        owner_aid: str,
        owner_name: str,
        public_key: bytes,
        heartbeat_interval: float,
    ) -> int:
        """Create, store in the link state, and return `node_id`."""
        # Sample a random int64 as node_id
        node_id = generate_rand_int_from_bytes(
            NODE_ID_NUM_BYTES, exclude=[SUPERLINK_NODE_ID, 0]
        )

        with self.lock:
            if node_id in self.nodes:
                log(ERROR, "Unexpected node registration failure.")
                return 0
            if public_key in self.node_public_key_to_node_id:
                raise ValueError("Public key already in use")

            # The node is not activated upon creation
            self.nodes[node_id] = NodeInfo(
                node_id=node_id,
                owner_aid=owner_aid,
                owner_name=owner_name,
                status=NodeStatus.REGISTERED,
                registered_at=now().isoformat(),
                last_activated_at=None,
                last_deactivated_at=None,
                unregistered_at=None,
                online_until=None,
                heartbeat_interval=heartbeat_interval,
                public_key=public_key,
            )
            self.node_public_key_to_node_id[public_key] = node_id
            self.owner_to_node_ids.setdefault(owner_aid, set()).add(node_id)
            return node_id

    def delete_node(self, owner_aid: str, node_id: int) -> None:
        """Delete a node."""
        with self.lock:
            if (
                not (node := self.nodes.get(node_id))
                or node.status == NodeStatus.UNREGISTERED
                or owner_aid != self.nodes[node_id].owner_aid
            ):
                raise ValueError(
                    f"Node ID {node_id} already unregistered, not found or "
                    "the request was unauthorized."
                )

            node.status = NodeStatus.UNREGISTERED
            current = now()
            node.unregistered_at = current.isoformat()
            # Set online_until to current timestamp on deletion, if it is in the future
            node.online_until = min(node.online_until, current.timestamp())

    def activate_node(self, node_id: int, heartbeat_interval: float) -> bool:
        """Activate the node with the specified `node_id`."""
        with self.lock:
            self._check_and_tag_offline_nodes(node_ids=[node_id])

            # Check if the node exists
            if not (node := self.nodes.get(node_id)):
                return False

            # Only activate if the node is currently registered or offline
            current_dt = now()
            if node.status in (NodeStatus.REGISTERED, NodeStatus.OFFLINE):
                node.status = NodeStatus.ONLINE
                node.last_activated_at = current_dt.isoformat()
                node.online_until = (
                    current_dt.timestamp() + HEARTBEAT_PATIENCE * heartbeat_interval
                )
                node.heartbeat_interval = heartbeat_interval
                return True
            return False

    def deactivate_node(self, node_id: int) -> bool:
        """Deactivate the node with the specified `node_id`."""
        with self.lock:
            self._check_and_tag_offline_nodes(node_ids=[node_id])

            # Check if the node exists
            if not (node := self.nodes.get(node_id)):
                return False

            # Only deactivate if the node is currently online
            current_dt = now()
            if node.status == NodeStatus.ONLINE:
                node.status = NodeStatus.OFFLINE
                node.last_deactivated_at = current_dt.isoformat()

                # Set online_until to current timestamp
                node.online_until = current_dt.timestamp()
                return True
            return False

    def get_nodes(self, run_id: int) -> set[int]:
        """Return all available nodes.

        Constraints
        -----------
        If the provided `run_id` does not exist or has no matching nodes,
        an empty `Set` MUST be returned.
        """
        with self.lock:
            if run_id not in self.run_ids:
                return set()
            federation = self.run_ids[run_id].run.federation
            node_ids = {
                node.node_id
                for node in self.get_node_info(statuses=[NodeStatus.ONLINE])
            }
            return self.federation_manager.filter_nodes(node_ids, federation)

    def get_node_info(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        owner_aids: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
    ) -> Sequence[NodeInfo]:
        """Retrieve information about nodes based on the specified filters."""
        with self.lock:
            self._check_and_tag_offline_nodes()
            result = []
            for node_id in self.nodes.keys() if node_ids is None else node_ids:
                if (node := self.nodes.get(node_id)) is None:
                    continue
                if owner_aids is not None and node.owner_aid not in owner_aids:
                    continue
                if statuses is not None and node.status not in statuses:
                    continue
                result.append(node)
            return result

    def _check_and_tag_offline_nodes(self, node_ids: list[int] | None = None) -> None:
        with self.lock:
            # Set all nodes of "online" status to "offline" if they've offline
            current_ts = now().timestamp()
            for node_id in node_ids or self.nodes.keys():
                if (node := self.nodes.get(node_id)) is None:
                    continue
                if node.status == NodeStatus.ONLINE:
                    if node.online_until <= current_ts:
                        node.status = NodeStatus.OFFLINE
                        node.last_deactivated_at = datetime.fromtimestamp(
                            node.online_until, tz=timezone.utc
                        ).isoformat()

    def get_node_id_by_public_key(self, public_key: bytes) -> int | None:
        """Get `node_id` for the specified `public_key` if it exists and is not
        deleted."""
        with self.lock:
            node_id = self.node_public_key_to_node_id.get(public_key)

            if node_id is None:
                return None

            node_info = self.nodes[node_id]
            if node_info.status == NodeStatus.UNREGISTERED:
                return None
            return node_id

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def create_run(
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
        with self.lock:
            run_id = generate_rand_int_from_bytes(RUN_ID_NUM_BYTES)

            if run_id not in self.run_ids:
                run_record = RunRecord(
                    run=Run(
                        run_id=run_id,
                        fab_id=fab_id if fab_id else "",
                        fab_version=fab_version if fab_version else "",
                        fab_hash=fab_hash if fab_hash else "",
                        override_config=override_config,
                        pending_at=now().isoformat(),
                        starting_at="",
                        running_at="",
                        finished_at="",
                        status=RunStatus(
                            status=Status.PENDING,
                            sub_status="",
                            details="",
                        ),
                        flwr_aid=flwr_aid if flwr_aid else "",
                        federation=federation,
                        bytes_sent=0,
                        bytes_recv=0,
                        clientapp_runtime=0.0,
                        run_type=run_type,
                    ),
                    federation_config=federation_config,
                )
                self.run_ids[run_id] = run_record
                # Add run_id to the flwr_aid_to_run_ids mapping if flwr_aid is provided
                if flwr_aid:
                    self.flwr_aid_to_run_ids[flwr_aid].add(run_id)

                return run_id
        log(ERROR, "Unexpected run creation failure.")
        return 0

    def get_run_info(
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

        with self.lock:
            # Build candidate set and apply each filter as an AND condition.
            matched_run_ids = set(self.run_ids.keys())

            # Filter by run_ids
            if run_ids is not None:
                if not run_ids:
                    return []
                matched_run_ids &= set(run_ids)

            # Filter by statuses
            if statuses is not None:
                if not statuses:
                    return []
                status_set = set(statuses)
                matched_run_ids &= {
                    run_id
                    for run_id in matched_run_ids
                    if self.run_ids[run_id].run.status.status in status_set
                }

            # Filter by Flower Account IDs
            if flwr_aids is not None:
                if not flwr_aids:
                    return []
                aid_matched: set[int] = set()
                for flwr_aid in flwr_aids:
                    aid_matched |= self.flwr_aid_to_run_ids.get(flwr_aid, set())
                matched_run_ids &= aid_matched

            # Filter by federations
            if federations is not None:
                if not federations:
                    return []
                federation_set = set(federations)
                matched_run_ids &= {
                    run_id
                    for run_id in matched_run_ids
                    if self.run_ids[run_id].run.federation in federation_set
                }

            runs = [self.run_ids[run_id].run for run_id in matched_run_ids]

            if order_by is not None:
                runs = sorted(
                    runs,
                    key=lambda run: run.pending_at,
                    reverse=not ascending,
                )

            if limit is not None:
                runs = runs[:limit]

            return runs

    def get_federation_config(self, run_id: int) -> SimulationConfig | None:
        """Get the resolved federation configuration for the specified `run_id`."""
        with self.lock:
            if run_id not in self.run_ids:
                log(ERROR, "`run_id` invalid for fetching resolved federation config")
                return None
            return self.run_ids[run_id].federation_config

    def get_run_status(self, run_ids: set[int]) -> dict[int, RunStatus]:
        """Retrieve the statuses for the specified runs."""
        # Clean up expired tokens; this will flag inactive runs as needed
        self._cleanup_expired_tokens()

        with self.lock:
            return {
                run_id: self.run_ids[run_id].run.status
                for run_id in set(run_ids)
                if run_id in self.run_ids
            }

    def update_run_status(self, run_id: int, new_status: RunStatus) -> bool:
        """Update the status of the run with the specified `run_id`."""
        # Clean up expired tokens; this will flag inactive runs as needed
        self._cleanup_expired_tokens()

        with self.lock:
            # Check if the run_id exists
            if run_id not in self.run_ids:
                log(ERROR, "`run_id` is invalid")
                return False

        with self.run_ids[run_id].lock:
            # Check if the status transition is valid
            current_status = self.run_ids[run_id].run.status
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

            # Update the run status
            current = now()
            run_record = self.run_ids[run_id]
            if new_status.status == Status.STARTING:
                run_record.run.starting_at = current.isoformat()
            elif new_status.status == Status.RUNNING:
                run_record.run.running_at = current.isoformat()
            elif new_status.status == Status.FINISHED:
                run_record.run.finished_at = current.isoformat()
            run_record.run.status = new_status

        # Report usage if the run is marked as finished after the update
        if new_status.status == Status.FINISHED:
            self.federation_manager.report_run_usage()
        return True

    def acknowledge_node_heartbeat(
        self, node_id: int, heartbeat_interval: float
    ) -> bool:
        """Acknowledge a heartbeat received from a node, serving as a heartbeat.

        A node is considered online as long as it sends heartbeats within
        the tolerated interval: HEARTBEAT_PATIENCE × heartbeat_interval.
        HEARTBEAT_PATIENCE = N allows for N-1 missed heartbeat before
        the node is marked as offline.
        """
        with self.lock:
            if (
                node := self.nodes.get(node_id)
            ) and node.status != NodeStatus.UNREGISTERED:
                current_dt = now()

                # Set timestamp if the status changes
                if node.status != NodeStatus.ONLINE:  # offline or registered
                    node.status = NodeStatus.ONLINE
                    node.last_activated_at = current_dt.isoformat()

                # Refresh `online_until` and `heartbeat_interval`
                node.online_until = (
                    current_dt.timestamp() + HEARTBEAT_PATIENCE * heartbeat_interval
                )
                node.heartbeat_interval = heartbeat_interval
                return True
            return False

    def _on_tokens_expired(self, expired_records: list[tuple[int, float]]) -> None:
        """Transition runs with expired tokens to failed status.

        Parameters
        ----------
        expired_records : list[tuple[int, float]]
            List of tuples containing (run_id, active_until timestamp)
            for expired tokens.
        """
        for run_id, active_until in expired_records:
            if not (run_record := self.run_ids.get(run_id)):
                continue
            with run_record.lock:
                run_record.run.status = RunStatus(
                    status=Status.FINISHED,
                    sub_status=SubStatus.FAILED,
                    details=RUN_FAILURE_DETAILS_NO_HEARTBEAT,
                )
                active_until_dt = datetime.fromtimestamp(active_until, tz=timezone.utc)
                run_record.run.finished_at = active_until_dt.isoformat()

        # Report usage for runs that have been marked as failed due to expired tokens
        self.federation_manager.report_run_usage()

    def get_serverapp_context(self, run_id: int) -> Context | None:
        """Get the context for the specified `run_id`."""
        return self.contexts.get(run_id)

    def set_serverapp_context(self, run_id: int, context: Context) -> None:
        """Set the context for the specified `run_id`."""
        if run_id not in self.run_ids:
            raise ValueError(f"Run {run_id} not found")
        self.contexts[run_id] = context

    def add_serverapp_log(self, run_id: int, log_message: str) -> None:
        """Add a log entry to the serverapp logs for the specified `run_id`."""
        if run_id not in self.run_ids:
            raise ValueError(f"Run {run_id} not found")
        run = self.run_ids[run_id]
        with run.log_lock:
            run.logs.append((now().timestamp(), log_message))

    def get_serverapp_log(
        self, run_id: int, after_timestamp: float | None
    ) -> tuple[str, float]:
        """Get the serverapp logs for the specified `run_id`."""
        if run_id not in self.run_ids:
            raise ValueError(f"Run {run_id} not found")
        run = self.run_ids[run_id]
        if after_timestamp is None:
            after_timestamp = 0.0
        with run.log_lock:
            # Find the index where the timestamp would be inserted
            index = bisect_right(run.logs, (after_timestamp, ""))
            latest_timestamp = run.logs[-1][0] if index < len(run.logs) else 0.0
            return "".join(log for _, log in run.logs[index:]), latest_timestamp

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

        with self.lock:
            if run_id not in self.run_ids:
                raise ValueError(f"Run {run_id} not found")
            run_record = self.run_ids[run_id]

        with run_record.lock:
            run = run_record.run
            run.bytes_sent += bytes_sent
            run.bytes_recv += bytes_recv

    def add_clientapp_runtime(self, run_id: int, runtime: float) -> None:
        """Add ClientApp runtime to the cumulative total for the specified `run_id`."""
        with self.lock:
            if run_id not in self.run_ids:
                raise ValueError(f"Run {run_id} not found")
            self.run_ids[run_id].run.clientapp_runtime += runtime
