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
"""Abstract base class LinkState."""


import abc
from collections.abc import Sequence
from typing import Literal

from flwr.app.user_config import UserConfig
from flwr.common import Context, Message
from flwr.common.typing import Run, RunStatus
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.supercore.corestate import CoreState
from flwr.superlink.federation import FederationManager


class LinkState(CoreState):  # pylint: disable=R0904
    """Abstract LinkState."""

    @property
    @abc.abstractmethod
    def federation_manager(self) -> FederationManager:
        """Return the FederationManager instance."""

    @abc.abstractmethod
    def store_message_ins(self, message: Message) -> str | None:
        """Store one Message.

        Usually, the ServerAppIo API calls this to schedule instructions.

        Stores the value of the `message` in the link state and, if successful,
        returns the `message_id` (str) of the `message`. If, for any reason,
        storing the `message` fails, `None` is returned.

        Constraints
        -----------
        `message.metadata.dst_node_id` MUST be set (not constant.SUPERLINK_NODE_ID)

        If `message.metadata.run_id` is invalid, then
        storing the `message` MUST fail.
        """

    @abc.abstractmethod
    def get_message_ins(self, node_id: int, limit: int | None) -> list[Message]:
        """Get zero or more `Message` objects for the provided `node_id`.

        Usually, the Fleet API calls this for Nodes planning to work on one or more
        Message.

        Constraints
        -----------
        Retrieve all Message where the `message.metadata.dst_node_id` equals `node_id`.

        If `limit` is not `None`, return, at most, `limit` number of `message`. If
        `limit` is set, it has to be greater zero.
        """

    @abc.abstractmethod
    def store_message_res(self, message: Message) -> str | None:
        """Store one Message.

        Usually, the Fleet API calls this for Nodes returning results.

        Stores the Message and, if successful, returns the `message_id` (str) of
        the `message`. If storing the `message` fails, `None` is returned.

        Constraints
        -----------
        `message.metadata.dst_node_id` MUST be set (not constant.SUPERLINK_NODE_ID)

        If `message.metadata.run_id` is invalid, then
        storing the `message` MUST fail.
        """

    @abc.abstractmethod
    def get_message_res(self, message_ids: set[str]) -> list[Message]:
        """Get reply Messages for the given Message IDs.

        This method is typically called by the ServerAppIo API to obtain
        results (type Message) for previously scheduled instructions (type Message).
        For each message_id passed, this method returns one of the following responses:

        - An error Message if there was no message registered with such message IDs
        or has expired.
        - An error Message if the reply Message exists but has expired.
        - The reply Message.
        - Nothing if the Message with the passed message_id is still valid and waiting
        for a reply Message.

        Parameters
        ----------
        message_ids : set[str]
            A set of Message IDs used to retrieve reply Messages responding to them.

        Returns
        -------
        list[Message]
            A list of reply Message responding to the given message IDs or Messages
            carrying an Error.
        """

    @abc.abstractmethod
    def num_message_ins(self) -> int:
        """Calculate the number of Messages awaiting a reply."""

    @abc.abstractmethod
    def num_message_res(self) -> int:
        """Calculate the number of reply Messages in store."""

    @abc.abstractmethod
    def delete_messages(self, message_ins_ids: set[str]) -> None:
        """Delete a Message and its reply based on provided Message IDs.

        Parameters
        ----------
        message_ins_ids : set[str]
            A set of Message IDs. For each ID in the set, the corresponding
            Message and its associated reply Message will be deleted.
        """

    @abc.abstractmethod
    def get_message_ids_from_run_id(self, run_id: int) -> set[str]:
        """Get all instruction Message IDs for the given run_id."""

    @abc.abstractmethod
    def create_node(
        self,
        owner_aid: str,
        owner_name: str,
        public_key: bytes,
        heartbeat_interval: float,
    ) -> int:
        """Create, store in the link state, and return `node_id`."""

    @abc.abstractmethod
    def delete_node(self, owner_aid: str, node_id: int) -> None:
        """Remove `node_id` from the link state."""

    @abc.abstractmethod
    def activate_node(self, node_id: int, heartbeat_interval: float) -> bool:
        """Activate the node with the specified `node_id`.

        Transitions the node status to "online". The transition will fail
        if the current status is not "registered" or "offline".

        Parameters
        ----------
        node_id : int
            The identifier of the node to activate.
        heartbeat_interval : float
            The interval (in seconds) from the current timestamp within which
            the next heartbeat from this node is expected to be received.

        Returns
        -------
        bool
            True if the status transition was successful, False otherwise.
        """

    @abc.abstractmethod
    def deactivate_node(self, node_id: int) -> bool:
        """Deactivate the node with the specified `node_id`.

        Transitions the node status to "offline". The transition will fail
        if the current status is not "online".

        Parameters
        ----------
        node_id : int
            The identifier of the node to deactivate.

        Returns
        -------
        bool
            True if the status transition was successful, False otherwise.
        """

    @abc.abstractmethod
    def get_nodes(self, run_id: int) -> set[int]:
        """Retrieve all currently stored node IDs as a set.

        Constraints
        -----------
        If the provided `run_id` does not exist or has no matching nodes,
        an empty `Set` MUST be returned.
        """

    @abc.abstractmethod
    def get_node_id_by_public_key(self, public_key: bytes) -> int | None:
        """Get `node_id` for the specified `public_key` if it exists and is not deleted.

        Parameters
        ----------
        public_key : bytes
            The public key of the node whose information is to be retrieved.

        Returns
        -------
        Optional[int]
            The `node_id` associated with the specified `public_key` if it exists
            and is not deleted; otherwise, `None`.
        """

    @abc.abstractmethod
    def get_node_info(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        owner_aids: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
    ) -> Sequence[NodeInfo]:
        """Retrieve information about nodes based on the specified filters.

        If a filter is set to None, it is ignored.
        If multiple filters are provided, they are combined using AND logic.

        Parameters
        ----------
        node_ids : Optional[Sequence[int]] (default: None)
            Sequence of node IDs to filter by. If a sequence is provided,
            it is treated as an OR condition.
        owner_aids : Optional[Sequence[str]] (default: None)
            Sequence of owner account IDs to filter by. If a sequence is provided,
            it is treated as an OR condition.
        statuses : Optional[Sequence[str]] (default: None)
            Sequence of node status values (e.g., "created", "activated")
            to filter by. If a sequence is provided, it is treated as an OR condition.

        Returns
        -------
        Sequence[NodeInfo]
            A sequence of NodeInfo objects representing the nodes matching
            the specified filters.
        """

    @abc.abstractmethod
    def create_run(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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
        """Create a new run.

        Parameters
        ----------
        fab_id : str | None
            The ID of the FAB, of format `<publisher>/<app-name>`.
        fab_version : str | None
            The version of the FAB.
        fab_hash : str | None
            The SHA256 hex hash of the FAB.
        override_config : UserConfig
            Configuration overrides for the run config.
        federation : str
            The federation this run belongs to.
        federation_config : SimulationConfig | None
            Optional resolved federation configuration for the run.
        flwr_aid : str | None
            Flower Account ID of the creator.
        run_type : str
            The type of run being created.

        Returns
        -------
        int
            The run ID of the newly created run.

        Notes
        -----
        This method will not verify if the account has permission to create
        a run in the federation.
        """

    @abc.abstractmethod
    def get_run_info(  # pylint: disable=too-many-arguments
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
        """Retrieve information about runs based on the specified filters.

        - If a filter is set to None, it is ignored.
        - If multiple filters are provided, they are combined using AND logic.
        - Within each filter, provided values are combined using OR logic.

        Parameters
        ----------
        run_ids : Optional[Sequence[int]] (default: None)
            Sequence of run IDs to filter by.
        statuses : Optional[Sequence[str]] (default: None)
            Sequence of run status values to filter by.
        flwr_aids : Optional[Sequence[str]] (default: None)
            Sequence of Flower Account IDs to filter by.
        federations : Optional[Sequence[str]] (default: None)
            Sequence of federation names to filter by.
        order_by : Optional[Literal["pending_at"]] (default: None)
            Field used to order the result.
        ascending : bool (default: True)
            Whether sorting should be in ascending order.
        limit : Optional[int] (default: None)
            Maximum number of runs to return. If `None`, no limit is applied.

        Returns
        -------
        Sequence[Run]
            A sequence of Run objects representing runs matching the specified filters.
        """

    @abc.abstractmethod
    def get_federation_config(self, run_id: int) -> SimulationConfig | None:
        """Get the resolved federation configuration for the specified `run_id`."""

    @abc.abstractmethod
    def get_run_status(self, run_ids: set[int]) -> dict[int, RunStatus]:
        """Retrieve the statuses for the specified runs.

        Parameters
        ----------
        run_ids : set[int]
            A set of run identifiers for which to retrieve statuses.

        Returns
        -------
        dict[int, RunStatus]
            A dictionary mapping each valid run ID to its corresponding status.

        Notes
        -----
        Only valid run IDs that exist in the State will be included in the returned
        dictionary. If a run ID is not found, it will be omitted from the result.
        """

    @abc.abstractmethod
    def update_run_status(self, run_id: int, new_status: RunStatus) -> bool:
        """Update the status of the run with the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run.
        new_status : RunStatus
            The new status to be assigned to the run.

        Returns
        -------
        bool
            True if the status update is successful; False otherwise.
        """

    @abc.abstractmethod
    def acknowledge_node_heartbeat(
        self, node_id: int, heartbeat_interval: float
    ) -> bool:
        """Acknowledge a heartbeat received from a node.

        A node is considered online as long as it sends heartbeats within
        the tolerated interval: HEARTBEAT_PATIENCE × heartbeat_interval.
        HEARTBEAT_PATIENCE = N allows for N-1 missed heartbeat before
        the node is marked as offline.

        Parameters
        ----------
        node_id : int
            The `node_id` from which the heartbeat was received.
        heartbeat_interval : float
            The interval (in seconds) from the current timestamp within which the next
            heartbeat from this node must be received. This acts as a hard deadline to
            ensure an accurate assessment of the node's availability.

        Returns
        -------
        is_acknowledged : bool
            True if the heartbeat is successfully acknowledged; otherwise, False.
        """

    @abc.abstractmethod
    def get_serverapp_context(self, run_id: int) -> Context | None:
        """Get the context for the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run for which to retrieve the context.

        Returns
        -------
        Optional[Context]
            The context associated with the specified `run_id`, or `None` if no context
            exists for the given `run_id`.
        """

    @abc.abstractmethod
    def set_serverapp_context(self, run_id: int, context: Context) -> None:
        """Set the context for the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run for which to set the context.
        context : Context
            The context to be associated with the specified `run_id`.
        """

    @abc.abstractmethod
    def add_serverapp_log(self, run_id: int, log_message: str) -> None:
        """Add a log entry to the ServerApp logs for the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run for which to add a log entry.
        log_message : str
            The log entry to be added to the ServerApp logs.
        """

    @abc.abstractmethod
    def get_serverapp_log(
        self, run_id: int, after_timestamp: float | None
    ) -> tuple[str, float]:
        """Get the ServerApp logs for the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run for which to retrieve the ServerApp logs.

        after_timestamp : Optional[float]
            Retrieve logs after this timestamp. If set to `None`, retrieve all logs.

        Returns
        -------
        tuple[str, float]
            A tuple containing:
            - The ServerApp logs associated with the specified `run_id`.
            - The timestamp of the latest log entry in the returned logs.
              Returns `0` if no logs are returned.
        """

    @abc.abstractmethod
    def store_traffic(self, run_id: int, *, bytes_sent: int, bytes_recv: int) -> None:
        """Store traffic data for the specified `run_id`.

        Parameters
        ----------
        run_id : int
            The identifier of the run for which to store traffic data.
        bytes_sent : int
            The number of bytes pulled by SuperNodes from the SuperLink to add to the
            run's total.
        bytes_recv : int
            The number of bytes received by SuperLink from SuperNodes to add to the
            run's total.
        """

    @abc.abstractmethod
    def add_clientapp_runtime(self, run_id: int, runtime: float) -> None:
        """Add ClientApp runtime to the cumulative total for the specified `run_id`.

        This method accumulates the runtime by adding the provided value to the
        existing total runtime for the run. Multiple ClientApps can contribute
        to the same run's total runtime.

        Parameters
        ----------
        run_id : int
            The identifier of the run for which to store each ClientApp's runtime.
        runtime : float
            The runtime in seconds to add to the `run_id`'s cumulative total.
        """
