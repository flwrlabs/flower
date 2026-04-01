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
"""Abstract base class NodeState."""


from abc import abstractmethod
from collections.abc import Sequence

from flwr.common import Context, Message
from flwr.common.typing import Run
from flwr.supercore.corestate import CoreState


class NodeState(CoreState):
    """Abstract base class for node state."""

    @abstractmethod
    def set_node_id(self, node_id: int) -> None:
        """Set the node ID."""

    @abstractmethod
    def get_node_id(self) -> int:
        """Get the node ID."""

    @abstractmethod
    def store_message(self, message: Message) -> str | None:
        """Store a message.

        Parameters
        ----------
        message : Message
            The message to store.

        Returns
        -------
        Optional[str]
            The object ID of the stored message, or None if storage failed.
        """

    @abstractmethod
    def get_messages(
        self,
        *,
        run_ids: Sequence[int] | None = None,
        is_reply: bool | None = None,
        limit: int | None = None,
    ) -> Sequence[Message]:
        """Retrieve messages based on the specified filters.

        If a filter is set to None, it is ignored.
        If multiple filters are provided, they are combined using AND logic.

        Parameters
        ----------
        run_ids : Optional[Sequence[int]] (default: None)
            Sequence of run IDs to filter by. If a sequence is provided,
            it is treated as an OR condition.
        is_reply : Optional[bool] (default: None)
            If True, filter for reply messages; if False, filter for non-reply
            (instruction) messages.
        limit : Optional[int] (default: None)
            Maximum number of messages to return. If None, no limit is applied.

        Returns
        -------
        Sequence[Message]
            A sequence of messages matching the specified filters.

        Notes
        -----
        **IMPORTANT:** Retrieved messages will **NOT** be returned again by subsequent
        calls to this method, even if the filters match them.
        """

    @abstractmethod
    def delete_messages(
        self,
        *,
        message_ids: Sequence[str] | None = None,
    ) -> None:
        """Delete messages based on the specified filters.

        If a filter is set to None, it is ignored.
        If multiple filters are provided, they are combined using AND logic.

        Parameters
        ----------
        message_ids : Optional[Sequence[str]] (default: None)
            Sequence of message (object) IDs to filter by. If a sequence is provided,
            it is treated as an OR condition.

        Notes
        -----
        **IMPORTANT:** **All messages** will be deleted if no filters are provided.
        """

    @abstractmethod
    def store_run(self, run: Run) -> None:
        """Store a run.

        Parameters
        ----------
        run : Run
            The `Run` instance to store.
        """

    @abstractmethod
    def get_run(self, run_id: int) -> Run | None:
        """Retrieve a run by its ID.

        Parameters
        ----------
        run_id : int
            The ID of the run to retrieve.

        Returns
        -------
        Optional[Run]
            The `Run` instance if found, otherwise None.
        """

    @abstractmethod
    def store_context(self, context: Context) -> None:
        """Store a context.

        Parameters
        ----------
        context : Context
            The context to store.
        """

    @abstractmethod
    def get_context(self, run_id: int) -> Context | None:
        """Retrieve a context by its run ID.

        Parameters
        ----------
        run_id : int
            The ID of the run with which the context is associated.

        Returns
        -------
        Optional[Context]
            The `Context` instance if found, otherwise None.
        """

    @abstractmethod
    def get_run_ids_with_pending_messages(self) -> Sequence[int]:
        """Retrieve run IDs that have at least one pending message.

        Run IDs that are currently in progress (i.e., those associated with tokens)
        will not be returned, even if they have pending messages.

        Returns
        -------
        Sequence[int]
            Sequence of run IDs with pending messages.
        """

    @abstractmethod
    def record_message_processing_start(self, message_id: str) -> None:
        """Record the start time of message processing based on the message ID.

        Parameters
        ----------
        message_id : str
            The ID of the message associated with the start time.
        """

    @abstractmethod
    def record_message_processing_end(self, message_id: str) -> None:
        """Record the end time of message processing based on the message ID.

        Parameters
        ----------
        message_id : str
            The ID of the message associated with the end time.

        Raises
        ------
        ValueError
            If the message ID is not found.
        """

    @abstractmethod
    def get_message_processing_duration(self, message_id: str) -> float:
        """Get the message processing duration based on the message ID.

        Parameters
        ----------
        message_id : str
            The ID of the message.

        Returns
        -------
        float
            The processing duration in seconds.

        Raises
        ------
        ValueError
            If the message ID is not found, or if start/end times are missing.
        """
