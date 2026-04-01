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
"""In-memory NodeState implementation."""


from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock, RLock

from flwr.common import Context, Error, Message, now
from flwr.common.constant import ErrorCode
from flwr.common.typing import Run
from flwr.supercore.constant import MESSAGE_TIME_ENTRY_MAX_AGE_SECONDS
from flwr.supercore.corestate.in_memory_corestate import InMemoryCoreState
from flwr.supercore.inflatable.inflatable_object import (
    get_all_nested_objects,
    get_object_tree,
    no_object_id_recompute,
)
from flwr.supercore.object_store import ObjectStore

from .nodestate import NodeState

CLIENT_APP_CRASHED_ERROR = Error(
    ErrorCode.CLIENT_APP_CRASHED, "ClientApp stopped responding."
)


@dataclass
class MessageEntry:
    """Data class to represent a message entry."""

    message: Message
    is_retrieved: bool = False


@dataclass
class TimeEntry:
    """Data class to represent a time entry."""

    starting_at: float
    finished_at: float | None = None


class InMemoryNodeState(
    NodeState, InMemoryCoreState
):  # pylint: disable=too-many-instance-attributes
    """In-memory NodeState implementation."""

    def __init__(self, object_store: ObjectStore) -> None:
        super().__init__(object_store)
        # Store node_id
        self.node_id: int | None = None
        # Store Object ID to MessageEntry mapping
        self.msg_store: dict[str, MessageEntry] = {}
        self.lock_msg_store = RLock()
        # Store run ID to Run mapping
        self.run_store: dict[int, Run] = {}
        self.lock_run_store = Lock()
        # Store run ID to Context mapping
        self.ctx_store: dict[int, Context] = {}
        self.lock_ctx_store = Lock()
        # Store msg ID to TimeEntry mapping
        self.time_store: dict[str, TimeEntry] = {}
        self.lock_time_store = Lock()

    def set_node_id(self, node_id: int | None) -> None:
        """Set the node ID."""
        self.node_id = node_id

    def get_node_id(self) -> int:
        """Get the node ID."""
        if self.node_id is None:
            raise ValueError("Node ID not set")
        return self.node_id

    def store_message(self, message: Message) -> str | None:
        """Store a message."""
        # No need to check for expired tokens here
        # The ClientAppIo servicer will first verify the token before storing messages
        with self.lock_msg_store:
            msg_id = message.metadata.message_id
            if msg_id == "" or msg_id in self.msg_store:
                return None
            self.msg_store[msg_id] = MessageEntry(message=message)
            return msg_id

    def get_messages(
        self,
        *,
        run_ids: Sequence[int] | None = None,
        is_reply: bool | None = None,
        limit: int | None = None,
    ) -> Sequence[Message]:
        """Retrieve messages based on the specified filters."""
        self._cleanup_expired_tokens()

        selected_messages: list[Message] = []
        with self.lock_msg_store:
            # Iterate through all messages in the store
            for object_id in list(self.msg_store.keys()):
                entry = self.msg_store[object_id]
                message = entry.message

                # Skip messages that have already been retrieved
                if entry.is_retrieved:
                    continue

                # Skip messages whose run_id doesn't match the filter
                if run_ids is not None:
                    if message.metadata.run_id not in run_ids:
                        continue

                # If is_reply filter is set, filter for reply/non-reply messages
                if is_reply is not None:
                    is_reply_message = message.metadata.reply_to_message_id != ""
                    # XOR logic to filter mismatched types (reply vs non-reply)
                    if is_reply ^ is_reply_message:
                        continue

                # Add the message to the result set
                selected_messages.append(message)

                # Mark the message as retrieved
                entry.is_retrieved = True

                # Stop if the number of collected messages reaches the limit
                if limit is not None and len(selected_messages) >= limit:
                    break

        return selected_messages

    def delete_messages(
        self,
        *,
        message_ids: Sequence[str] | None = None,
    ) -> None:
        """Delete messages based on the specified filters."""
        with self.lock_msg_store:
            if message_ids is None:
                # If no message IDs are provided, clear the entire store
                self.msg_store.clear()
                return

            # Remove specified messages from the store
            for msg_id in message_ids:
                self.msg_store.pop(msg_id, None)

    def store_run(self, run: Run) -> None:
        """Store a run."""
        with self.lock_run_store:
            self.run_store[run.run_id] = run

    def get_run(self, run_id: int) -> Run | None:
        """Retrieve a run by its ID."""
        with self.lock_run_store:
            return self.run_store.get(run_id)

    def store_context(self, context: Context) -> None:
        """Store a context."""
        with self.lock_ctx_store:
            self.ctx_store[context.run_id] = context

    def get_context(self, run_id: int) -> Context | None:
        """Retrieve a context by its run ID."""
        with self.lock_ctx_store:
            return self.ctx_store.get(run_id)

    def get_run_ids_with_pending_messages(self) -> Sequence[int]:
        """Retrieve run IDs that have at least one pending message."""
        # Collect run IDs from messages
        with self.lock_msg_store:
            ret = {
                entry.message.metadata.run_id
                for entry in self.msg_store.values()
                if entry.message.metadata.reply_to_message_id == ""
                and not entry.is_retrieved
            }

        # Remove run IDs that have tokens stored (indicating they are in progress)
        with self.lock_token_store:
            ret -= set(self.token_store.keys())
            return list(ret)

    def _on_tokens_expired(self, expired_records: list[tuple[int, float]]) -> None:
        """Insert error replies for messages associated with expired tokens."""
        with self.lock_msg_store:
            # Find all retrieved messages associated with expired run IDs
            expired_run_ids = {run_id for run_id, _ in expired_records}
            messages_to_reply: list[Message] = []
            for entry in self.msg_store.values():
                msg = entry.message
                if msg.metadata.run_id in expired_run_ids and entry.is_retrieved:
                    messages_to_reply.append(msg)

            # Create and store error replies for each message
            for msg in messages_to_reply:
                error_reply = Message(CLIENT_APP_CRASHED_ERROR, reply_to=msg)

                # Insert objects of the error reply into the object store
                with no_object_id_recompute():
                    # pylint: disable-next=W0212
                    error_reply.metadata._message_id = error_reply.object_id  # type: ignore
                    object_tree = get_object_tree(error_reply)
                    self.object_store.preregister(msg.metadata.run_id, object_tree)
                    for obj_id, obj in get_all_nested_objects(error_reply).items():
                        self.object_store.put(obj_id, obj.deflate())

                # Store the error reply message
                self.store_message(error_reply)

    def record_message_processing_start(self, message_id: str) -> None:
        """Record the start time of message processing based on the message ID."""
        with self.lock_time_store:
            self.time_store[message_id] = TimeEntry(starting_at=now().timestamp())

    def record_message_processing_end(self, message_id: str) -> None:
        """Record the end time of message processing based on the message ID."""
        with self.lock_time_store:
            if message_id not in self.time_store:
                raise ValueError(
                    f"Cannot record end time: Message ID {message_id} not found."
                )
            entry = self.time_store[message_id]
            entry.finished_at = now().timestamp()

    def get_message_processing_duration(self, message_id: str) -> float:
        """Get the message processing duration based on the message ID."""
        # Cleanup old message processing times
        self._cleanup_old_message_times()
        with self.lock_time_store:
            if message_id not in self.time_store:
                raise ValueError(f"Message ID {message_id} not found.")

            entry = self.time_store[message_id]
            if entry.starting_at is None or entry.finished_at is None:
                raise ValueError(
                    f"Start time or end time for message ID {message_id} is missing."
                )

            duration = entry.finished_at - entry.starting_at
            return duration

    def _cleanup_old_message_times(self) -> None:
        """Remove time entries older than MESSAGE_TIME_ENTRY_MAX_AGE_SECONDS."""
        with self.lock_time_store:
            cutoff = now().timestamp() - MESSAGE_TIME_ENTRY_MAX_AGE_SECONDS
            # Find message IDs for entries that have a finishing_at time
            # before the cutoff, and those that don't exist in msg_store
            to_delete = [
                msg_id
                for msg_id, entry in self.time_store.items()
                if (entry.finished_at and entry.finished_at < cutoff)
                or msg_id not in self.msg_store
            ]

            # Delete the identified entries
            for msg_id in to_delete:
                del self.time_store[msg_id]
