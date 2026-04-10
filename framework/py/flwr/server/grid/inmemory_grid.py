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
"""Flower in-memory Grid."""


import time
from collections.abc import Iterable
from typing import cast
from uuid import uuid4

from flwr.common import Message, RecordDict
from flwr.common.constant import SUPERLINK_NODE_ID
from flwr.common.logger import warn_deprecated_feature
from flwr.common.typing import Run
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.server.superlink.linkstate import LinkStateFactory

from .grid import Grid


class InMemoryGrid(Grid):
    """`InMemoryGrid` class provides an interface to the ServerAppIo API.

    Parameters
    ----------
    state_factory : StateFactory
        A StateFactory embedding a state that this grid can interface with.
    pull_interval : float (default=0.1)
        Sleep duration between calls to `pull_messages`.
    """

    _deprecation_warning_logged = False

    def __init__(
        self,
        state_factory: LinkStateFactory,
        pull_interval: float = 0.1,
    ) -> None:
        self._run: Run | None = None
        self.state = state_factory.state()
        self.pull_interval = pull_interval
        self.node = Node(node_id=SUPERLINK_NODE_ID)

    def _check_message(self, message: Message) -> None:
        # Check if the message is valid
        if not (
            message.metadata.message_id != ""
            and message.metadata.reply_to_message_id == ""
            and message.metadata.ttl > 0
            and message.metadata.delivered_at == ""
        ):
            raise ValueError(f"Invalid message: {message}")

    def set_run(self, run: Run) -> None:
        """Initialize the run."""
        if not isinstance(run, Run):
            run_type = type(run).__name__
            raise TypeError(f"`run` must be an instance of Run, got {run_type}")
        runs = self.state.get_run_info(run_ids=[run.run_id])
        if not runs:
            raise RuntimeError(f"Cannot find the run with ID: {run.run_id}")
        self._run = run

    @property
    def run(self) -> Run:
        """Run ID."""
        return Run(**vars(cast(Run, self._run)))

    def create_message(  # pylint: disable=too-many-arguments,R0917
        self,
        content: RecordDict,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: float | None = None,
    ) -> Message:
        """Create a new message with specified parameters.

        This method constructs a new `Message` with given content and metadata.
        The `run_id` and `src_node_id` will be set automatically.
        """
        if not InMemoryGrid._deprecation_warning_logged:
            InMemoryGrid._deprecation_warning_logged = True
            warn_deprecated_feature(
                "`Driver.create_message` / `Grid.create_message` is deprecated."
                "Use `Message` constructor instead."
            )
        return Message(content, dst_node_id, message_type, ttl=ttl, group_id=group_id)

    def get_node_ids(self) -> Iterable[int]:
        """Get node IDs."""
        return self.state.get_nodes(cast(Run, self._run).run_id)

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        """Push messages to specified node IDs.

        This method takes an iterable of messages and sends each message
        to the node specified in `dst_node_id`.
        """
        msg_ids: list[str] = []
        for msg in messages:
            # Populate metadata
            msg.metadata.__dict__["_run_id"] = cast(Run, self._run).run_id
            msg.metadata.__dict__["_src_node_id"] = self.node.node_id
            msg.metadata.__dict__["_message_id"] = str(uuid4())
            # Check message
            self._check_message(msg)
            # Store in state
            msg_id = self.state.store_message_ins(msg)
            if msg_id:
                msg_ids.append(str(msg_id))

        return msg_ids

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        """Pull messages based on message IDs.

        This method is used to collect messages from the SuperLink that correspond to a
        set of given message IDs.
        """
        msg_ids = set(message_ids)
        # Pull Messages
        message_res_list = self.state.get_message_res(message_ids=msg_ids)
        # Get IDs of Messages these replies are for
        message_ins_ids_to_delete = {
            msg_res.metadata.reply_to_message_id for msg_res in message_res_list
        }
        # Delete
        self.state.delete_messages(message_ins_ids=message_ins_ids_to_delete)

        return message_res_list

    def send_and_receive(
        self,
        messages: Iterable[Message],
        *,
        timeout: float | None = None,
    ) -> Iterable[Message]:
        """Push messages to specified node IDs and pull the reply messages.

        This method sends a list of messages to their destination node IDs and then
        waits for the replies. It continues to pull replies until either all replies are
        received or the specified timeout duration is exceeded.
        """
        # Push messages
        msg_ids = set(self.push_messages(messages))
        del messages

        # Pull messages
        end_time = time.time() + (timeout if timeout is not None else 0.0)
        ret: list[Message] = []
        while timeout is None or time.time() < end_time:
            res_msgs = self.pull_messages(msg_ids)
            ret.extend(res_msgs)
            msg_ids.difference_update(
                {msg.metadata.reply_to_message_id for msg in res_msgs}
            )
            if len(msg_ids) == 0:
                break
            # Sleep
            time.sleep(self.pull_interval)
        return ret
