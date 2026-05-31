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
"""Abstract base class CoreState."""


from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

from flwr.app import Context, Message
from flwr.common.typing import Fab
from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskEvent  # pylint: disable=E0611

from ..object_store import ObjectStore


class CoreState(ABC):  # pylint: disable=R0904
    """Abstract base class for core state."""

    @property
    @abstractmethod
    def object_store(self) -> ObjectStore:
        """Return the ObjectStore instance used by this CoreState."""

    @abstractmethod
    def store_fab(self, fab: Fab) -> str:
        """Store a FAB and return its canonical SHA-256 hash."""

    @abstractmethod
    def get_fab(self, fab_hash: str) -> Fab | None:
        """Return the FAB for the given hash, if present."""

    @abstractmethod
    def get_run_series(
        self,
        *,
        federation: str | None = None,
        updated_before: str | None = None,
        limit: int | None = None,
    ) -> Sequence[RunSeries]:
        """Return RunSeries metadata, optionally filtered by federation.

        Parameters
        ----------
        federation : str | None (default: None)
            Federation name used to filter RunSeries. If `None`, RunSeries from all
            federations are returned.
        updated_before : str | None (default: None)
            If set, return only RunSeries updated before this ISO timestamp.
        limit : int | None (default: None)
            Maximum number of RunSeries records to return. If `None`, no limit is
            applied.

        Returns
        -------
        Sequence[RunSeries]
            RunSeries records ordered by `updated_at` descending.
        """

    @abstractmethod
    def get_run_series_context(self, series_id: int) -> Context | None:
        """Return the shared Context for the specified RunSeries, if present.

        Parameters
        ----------
        series_id : int
            The ID of the RunSeries for which to retrieve shared context.

        Returns
        -------
        Context | None
            The shared RunSeries context, or `None` if no context is stored.
        """

    @abstractmethod
    def set_run_series_context(self, series_id: int, context: Context) -> None:
        """Set the shared Context for the specified RunSeries.

        Parameters
        ----------
        series_id : int
            The ID of the RunSeries for which to persist shared context.
        context : Context
            The shared context to store.
        """

    @abstractmethod
    def store_run_in_series(
        self,
        run_id: int,
        federation: str,
        series_id: int | None,
    ) -> int | None:
        """Store a run in a run series and return the series ID.

        Parameters
        ----------
        run_id : int
            Run ID to associate with the run series.
        federation : str
            Federation the run series belongs to.
        series_id : int | None
            Caller-provided series ID. If `None`, a new series ID is generated
            and creation is attempted. If set, the matching series must already
            exist and belong to `federation`.

        Returns
        -------
        int | None
            The ID of the run series the run was stored in, or `None` if a
            new run series could not be created, the caller-provided run
            series is invalid, or the run could not be associated with the
            run series.
        """

    @abstractmethod
    def add_task_log(self, task_id: int, log_message: str) -> None:
        """Add a log entry to the task logs for the specified `task_id`.

        Parameters
        ----------
        task_id : int
            The identifier of the task for which to add a log entry.
        log_message : str
            The log entry to be added to the task logs.
        """

    @abstractmethod
    def get_task_log(
        self, task_id: int, after_timestamp: float | None
    ) -> tuple[str, float]:
        """Get task logs for the specified `task_id`.

        Parameters
        ----------
        task_id : int
            The identifier of the task for which to retrieve logs.
        after_timestamp : Optional[float]
            Retrieve logs after this timestamp. If set to `None`, retrieve all logs.
            The filter is strict: logs at exactly `after_timestamp` are considered
            already consumed by the caller.

        Returns
        -------
        tuple[str, float]
            A tuple containing:
            - The concatenated task logs associated with the specified `task_id`.
            - The timestamp of the latest log entry in the returned logs.
              Returns `0` if no logs are returned.
        """

    @abstractmethod
    def create_task(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        task_type: str,
        run_id: int,
        fab_hash: str | None = None,
        model_ref: str | None = None,
        connector_ref: str | None = None,
        requesting_task_id: int | None = None,
    ) -> int | None:
        """Create a new task.

        Parameters
        ----------
        task_type : str
            The executor type of the task to create.
        run_id : int
            The run ID this task belongs to.
        fab_hash : Optional[str] (default: None)
            FAB hash associated with the task, if applicable.
        model_ref : Optional[str] (default: None)
            Model reference associated with the task, if applicable.
        connector_ref : Optional[str] (default: None)
            Connector reference associated with the task, if applicable.
        requesting_task_id : Optional[int] (default: None)
            Task requesting creation of the new task. If set, task creation fails
            when the requesting task does not exist or is already finished.

        Returns
        -------
        Optional[int]
            The task ID of the newly created task, or `None` if task creation
            fails.

        Notes
        -----
        Newly created tasks are in the pending status.
        This method only persists task data. It does not validate whether the
        provided fields are required for the given task type.
        """

    @abstractmethod
    def get_tasks(  # pylint: disable=too-many-arguments
        self,
        *,
        task_ids: Sequence[int] | None = None,
        run_ids: Sequence[int] | None = None,
        statuses: Sequence[str] | None = None,
        order_by: Literal["pending_at"] | None = None,
        ascending: bool = True,
        limit: int | None = None,
    ) -> Sequence[Task]:
        """Retrieve information about tasks based on the specified filters.

        - If a filter is set to None, it is ignored.
        - If multiple filters are provided, they are combined using AND logic.
        - Within each filter, provided values are combined using OR logic.

        Parameters
        ----------
        task_ids : Optional[Sequence[int]] (default: None)
            Sequence of task IDs to filter by.
        run_ids : Optional[Sequence[int]] (default: None)
            Sequence of run IDs to filter by.
        statuses : Optional[Sequence[str]] (default: None)
            Sequence of task status values to filter by.
        order_by : Optional[Literal["pending_at"]] (default: None)
            Field used to order the result.
        ascending : bool (default: True)
            Whether sorting should be in ascending order.
        limit : Optional[int] (default: None)
            Maximum number of tasks to return. If `None`, no limit is applied.

        Returns
        -------
        Sequence[Task]
            A sequence of Task objects representing tasks matching the specified
            filters.
        """

    @abstractmethod
    def claim_task(self, task_id: int) -> str | None:
        """Atomically claim a pending task.

        Claiming a task creates a task token, initializes heartbeat state, and
        moves the task from pending to starting.

        Parameters
        ----------
        task_id : int
            The ID of the task to claim.

        Returns
        -------
        Optional[str]
            The generated task token if the claim succeeds, otherwise `None`.
        """

    @abstractmethod
    def activate_task(self, task_id: int) -> bool:
        """Move a task from starting to running.

        Parameters
        ----------
        task_id : int
            The ID of the task to activate.

        Returns
        -------
        bool
            True if the task existed and transitioned from starting to running,
            otherwise False.
        """

    @abstractmethod
    def finish_task(self, task_id: int, sub_status: str, details: str) -> bool:
        """Move an unfinished task to finished.

        Parameters
        ----------
        task_id : int
            The ID of the task to finish.
        sub_status : str
            Terminal task sub-status, such as completed, failed, or stopped.
            Only RUNNING status can be transitioned to FINISHED:COMPLETED
        details : str
            Additional terminal status details.

        Returns
        -------
        bool
            True if the task existed and was not already finished, otherwise
            False.
        """

    @abstractmethod
    def acknowledge_task_heartbeat(self, task_id: int) -> bool:
        """Extend heartbeat state for the claimed task.

        Parameters
        ----------
        task_id : int
            The ID of the task whose heartbeat should be acknowledged.

        Returns
        -------
        bool
            True if the task heartbeat was acknowledged successfully, otherwise
            False.
        """

    @abstractmethod
    def get_task_by_token(self, token: str) -> Task | None:
        """Return the task associated with the task token, if valid.

        Parameters
        ----------
        token : str
            The task token to look up.

        Returns
        -------
        Task | None
            The task if the token is valid, otherwise None.
        """

    @abstractmethod
    def store_task_message(self, message: Message) -> bool:
        """Store one task-addressed Message.

        The source and destination task IDs are read from
        `message.metadata.src_task_id` and `message.metadata.dst_task_id`.

        Parameters
        ----------
        message : Message
            The task-addressed message to store.

        Returns
        -------
        bool
            True if the message was stored, otherwise False.
        """

    @abstractmethod
    def get_task_message(
        self,
        *,
        dst_task_ids: Sequence[int] | None = None,
        limit: int | None = None,
        order_by: Literal["created_at"] | None = None,
    ) -> Sequence[Message]:
        """Retrieve undelivered task-addressed Messages.

        Returned messages are atomically consumed so later calls will not return
        them again.

        Parameters
        ----------
        dst_task_ids : Optional[Sequence[int]] (default: None)
            Sequence of destination task IDs to filter by.
        limit : Optional[int] (default: None)
            Maximum number of messages to return. If `None`, no limit is applied.
        order_by : Optional[Literal["created_at"]] (default: None)
            If set to "created_at", matching messages are returned in ascending
            creation-time order.

        Returns
        -------
        Sequence[Message]
            A sequence of matching messages.
        """

    @abstractmethod
    def store_task_events(
        self,
        events: Sequence[TaskEvent],
    ) -> bool:
        """Store task-produced run events.

        Parameters
        ----------
        events : Sequence[TaskEvent]
            Task events to store. Event IDs and timestamps are assigned by the
            CoreState implementation. Event payloads are validated before any
            events are stored, so one invalid event payload rejects the whole
            batch.

        Returns
        -------
        bool
            True if the events were stored, otherwise False.
        """

    @abstractmethod
    def get_task_events(
        self,
        *,
        run_id: int | None = None,
        after_task_event_id: int | None = None,
    ) -> Sequence[TaskEvent]:
        """Return task-produced run events matching the filters.

        Parameters
        ----------
        run_id : Optional[int] (default: None)
            If set, return only events for this run. If set to `None`, return
            events for all runs.
        after_task_event_id : Optional[int] (default: None)
            Return only events with an ID greater than this cursor. If set to
            `None`, retrieve all events.

        Returns
        -------
        Sequence[TaskEvent]
            Task events ordered by ID.
        """

    @abstractmethod
    def reserve_nonce(self, namespace: str, nonce: str, expires_at: float) -> bool:
        """Atomically reserve a nonce in a namespace until `expires_at`.

        Parameters
        ----------
        namespace : str
            Namespace for the nonce reservation. Empty values are treated as
            invalid and return False.
        nonce : str
            Nonce value to reserve. Empty values are treated as invalid and
            return False.
        expires_at : float
            POSIX timestamp when the reservation expires. Values in the past
            are accepted.

        Returns
        -------
        bool
            True if the nonce was reserved. False if the input is invalid or
            the nonce already exists and is active.
        """
