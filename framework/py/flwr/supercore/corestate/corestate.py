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
from datetime import datetime
from typing import Literal

from flwr.app import Context, Message
from flwr.app.user_config import UserConfig
from flwr.proto.control_pb2 import Automation  # pylint: disable=E0611
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.proto.runseries_pb2 import RunSeries  # pylint: disable=E0611
from flwr.proto.task_pb2 import Task, TaskEvent, TaskUsage  # pylint: disable=E0611
from flwr.supercore.fab import Fab
from flwr.supercore.typing import ConnectorOAuthSessionRecord, ConnectorRecord

from ..constant import AutomationStatus
from ..object_store import ObjectStore


class CoreState(ABC):  # pylint: disable=R0904
    """Abstract base class for core state."""

    @property
    @abstractmethod
    def object_store(self) -> ObjectStore:
        """Return the ObjectStore instance used by this CoreState."""

    @abstractmethod
    def start_session(self, run_id: int) -> str:
        """Start a run-scoped object push session."""

    @abstractmethod
    def delete_sessions_in_run(self, run_id: int) -> None:
        """Delete bookkeeping for all object push sessions in a run.

        This does not delete any messages or objects associated with the sessions.
        """

    @abstractmethod
    def preregister_object_tree(
        self, object_tree: ObjectTree, session_id: str
    ) -> list[str]:
        """Preregister the object tree for the object push session."""

    @abstractmethod
    def store_object(
        self,
        run_id: int,
        session_id: str,
        object_id: str,
        object_content: bytes,
    ) -> bool:
        """Store an object if it is pending for an active push session.

        Parameters
        ----------
        run_id : int
            The ID of the run with which the push session is associated.
        session_id : str
            The ID of the object push session.
        object_id : str
            The ID of the object to store.
        object_content : bytes
            The object content to store.

        Returns
        -------
        bool
            True if the object was stored, otherwise False.
        """

    @abstractmethod
    def get_object(self, run_id: int, object_id: str) -> bytes | None:
        """Get an object and clean up expired push sessions when needed.

        Parameters
        ----------
        run_id : int
            The ID of the run requesting the object.
        object_id : str
            The ID of the object to retrieve.

        Returns
        -------
        bytes | None
            The object content, `b""` if it is known but unavailable, or None if it
            is unknown.
        """

    @abstractmethod
    def _cleanup_push_session(self, session_id: str, *, cleanup_messages: bool) -> None:
        """Remove an object push session and optionally its messages."""

    def _on_push_session_expired(self, message_object_ids: set[str]) -> None:
        """Handle messages when a push session expires."""

    @abstractmethod
    def store_fab(self, fab: Fab) -> str:
        """Store a FAB and return its canonical SHA-256 hash."""

    @abstractmethod
    def get_fab(self, fab_hash: str) -> Fab | None:
        """Return the FAB for the given hash, if present."""

    @abstractmethod
    def upsert_connector(
        self,
        flwr_aid: str,
        connector_ref: str,
        credentials_json: str,
        config_json: str,
    ) -> bool:
        """Create or update a connector for an account.

        Parameters
        ----------
        flwr_aid : str
            Account ID owning the connector.
        connector_ref : str
            Connector reference unique within the account.
        credentials_json : str
            Serialized connector credentials.
        config_json : str
            Serialized connector configuration.

        Returns
        -------
        bool
            ``True`` if the connector was stored, otherwise ``False``.
        """

    @abstractmethod
    def get_connector(
        self, flwr_aid: str, connector_ref: str
    ) -> ConnectorRecord | None:
        """Return an account's connector, if present.

        Parameters
        ----------
        flwr_aid : str
            Account ID owning the connector.
        connector_ref : str
            Connector reference unique within the account.

        Returns
        -------
        ConnectorRecord | None
            The stored connector, or `None` if it does not exist.
        """

    @abstractmethod
    def delete_connector(self, flwr_aid: str, connector_ref: str) -> bool:
        """Delete an account's connector if it exists.

        Parameters
        ----------
        flwr_aid : str
            Account ID owning the connector.
        connector_ref : str
            Connector reference unique within the account.

        Returns
        -------
        bool
            `True` if the connector was deleted, otherwise `False`.
        """

    @abstractmethod
    def bind_connectors_to_run(
        self, run_id: int, connector_refs: Sequence[str]
    ) -> bool:
        """Associate connector references with a run."""

    @abstractmethod
    def get_run_connector_refs(self, run_id: int) -> Sequence[str]:
        """Return connector references associated with a run."""

    @abstractmethod
    def create_connector_oauth_session(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        oauth_session_id: str,
        flwr_aid: str,
        connector_ref: str,
        state: str,
        redirect_uri: str,
        pkce_verifier: str | None,
        expires_at: datetime,
    ) -> ConnectorOAuthSessionRecord | None:
        """Create and return a connector OAuth session.

        Parameters
        ----------
        oauth_session_id : str
            Unique ID of the OAuth session.
        flwr_aid : str
            Account ID owning the OAuth session.
        connector_ref : str
            Reference of the connector being authorized.
        state : str
            OAuth state used to protect against cross-site request forgery.
        redirect_uri : str
            URI to redirect to after authorization.
        pkce_verifier : str | None
            PKCE verifier used for the authorization code exchange, if present.
        expires_at : datetime
            Timezone-aware expiration timestamp, normalized to UTC before storage.

        Returns
        -------
        ConnectorOAuthSessionRecord | None
            The created session, or `None` if the session ID already exists, a
            required identifier is empty, or `expires_at` is timezone-naive.
        """

    @abstractmethod
    def get_connector_oauth_session(
        self, oauth_session_id: str, flwr_aid: str
    ) -> ConnectorOAuthSessionRecord | None:
        """Return an account's connector OAuth session, if present.

        Parameters
        ----------
        oauth_session_id : str
            Unique ID of the OAuth session.
        flwr_aid : str
            Account ID owning the OAuth session.

        Returns
        -------
        ConnectorOAuthSessionRecord | None
            The stored session, or `None` if it does not exist for the account.
        """

    @abstractmethod
    def complete_connector_oauth_session(
        self, oauth_session_id: str, flwr_aid: str
    ) -> bool:
        """Mark a pending connector OAuth session as completed.

        Parameters
        ----------
        oauth_session_id : str
            Unique ID of the OAuth session.
        flwr_aid : str
            Account ID owning the OAuth session.

        Returns
        -------
        bool
            `True` if the session was completed. `False` if the session is missing,
            belongs to another account, is expired, or was already completed.
        """

    @abstractmethod
    def store_message_and_object_tree(
        self, message: Message, object_tree: ObjectTree, session_id: str
    ) -> tuple[bool, list[str]]:
        """Store a Message and preregister its ObjectTree.

        Parameters
        ----------
        message : Message
            The Message to store.
        object_tree : ObjectTree
            The ObjectTree containing the IDs of objects to preregister.
        session_id : str
            The ID of the object push session.

        Returns
        -------
        tuple[bool, list[str]]
            A tuple containing a boolean indicating whether the Message was
            stored and a list of object IDs that still need to be pushed. If
            storing the Message fails, returns `(False, [])`.
        """

    @abstractmethod
    def get_run_series(
        self,
        *,
        series_ids: Sequence[int] | None = None,
        federation_ids: Sequence[str] | None = None,
        updated_before: str | None = None,
        limit: int | None = None,
    ) -> Sequence[RunSeries]:
        """Return RunSeries metadata, optionally filtered by the given filters.

        - If a filter is set to None, it is ignored.
        - If multiple filters are provided, they are combined using AND logic.
        - Within each filter, provided values are combined using OR logic.

        Parameters
        ----------
        series_ids : Optional[Sequence[int]] (default: None)
            Sequence of RunSeries IDs to filter by.
        federation_ids : Optional[Sequence[str]] (default: None)
            Sequence of federation IDs to filter by.
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
        federation_id: str,
        series_id: int | None,
        description: str | None = None,
    ) -> int | None:
        """Store a run in a run series and return the series ID.

        Parameters
        ----------
        run_id : int
            Run ID to associate with the run series.
        federation_id : str
            Federation ID the run series belongs to.
        series_id : int | None
            Caller-provided series ID. If `None`, a new series ID is generated
            and creation is attempted. If set, the matching series must already
            exist and belong to `federation_id`.
        description : str | None (default: None)
            Optional description for a newly created run series. Ignored when
            `series_id` refers to an existing run series. `None` means no
            description was provided; an empty string is an explicit description.

        Returns
        -------
        int | None
            The ID of the run series the run was stored in, or `None` if a
            new run series could not be created, the caller-provided run
            series is invalid, or the run could not be associated with the
            run series.
        """

    @abstractmethod
    def store_automation(  # pylint: disable=too-many-arguments
        self,
        *,
        federation_id: str,
        flwr_aid: str,
        fab_id: str | None,
        fab_version: str | None,
        fab_hash: str | None,
        override_config: UserConfig,
        federation_config: SimulationConfig | None,
        primary_task_type: str,
        series_id: int,
        next_run_at: str,
        fixed_interval: int | None = None,
        max_runs: int | None = None,
    ) -> Automation:
        """Store an automation and return its metadata.

        Parameters
        ----------
        federation_id : str
            Federation ID the automation belongs to.
        flwr_aid : str
            FLWR account ID used to dispatch the automation.
        fab_id : str | None
            FAB ID used by future runs.
        fab_version : str | None
            FAB version used by future runs.
        fab_hash : str | None
            FAB hash used by future runs.
        override_config : UserConfig
            Run override config used by future runs.
        federation_config : SimulationConfig | None
            Federation config override used by future runs.
        primary_task_type : str
            Primary task type used by future runs.
        series_id : int
            Run series ID to use when dispatching automation runs.
        next_run_at : str
            Initial due time as a timestamp string. This is required when
            storing an automation. For one-shot automations, this is the
            requested `start_at`; for recurring automations, this is the first
            scheduled run time.
        fixed_interval : int | None (default: None)
            Recurring interval in seconds.
        max_runs : int | None (default: None)
            Maximum number of runs, if finite. The value initializes the
            persisted `remaining_runs` counter.

        Returns
        -------
        Automation
            Stored automation metadata.
        """

    @abstractmethod
    def list_automations(  # pylint: disable=too-many-arguments
        self,
        *,
        federation: str | None = None,
        statuses: Sequence[str] | None = None,
        due_before: datetime | None = None,
        order_by: Literal["next_run_at", "updated_at"],
        limit: int | None = None,
    ) -> Sequence[Automation]:
        """Return automations matching the given filters.

        Parameters
        ----------
        federation : str | None (default: None)
            Federation ID to filter by.
        statuses : Sequence[str] | None (default: None)
            Automation statuses to filter by.
        due_before : datetime | None (default: None)
            If set, return only automations with `next_run_at` at or before this
            timestamp.
        order_by : Literal["next_run_at", "updated_at"]
            Field used to order the result. `next_run_at` orders ascending;
            `updated_at` orders descending.
        limit : int | None (default: None)
            Maximum number of automation records to return.

        Returns
        -------
        Sequence[Automation]
            Automation metadata ordered by `order_by`.
        """

    @abstractmethod
    def stop_automation(self, automation_id: int) -> bool:
        """Stop an active automation.

        Parameters
        ----------
        automation_id : int
            Automation ID to stop.

        Returns
        -------
        bool
            True if an active automation was stopped, otherwise False.
        """

    @abstractmethod
    def advance_automation(
        self,
        automation_id: int,
        *,
        previous_next_run_at: str,
        next_run_at: str | None,
    ) -> bool:
        """Advance an active automation occurrence.

        Parameters
        ----------
        automation_id : int
            Automation ID to advance.
        previous_next_run_at : str
            Previously observed due time timestamp string. The update only
            succeeds if the stored `next_run_at` still matches this value, preventing
            multiple workers from executing the same scheduled run concurrently.
        next_run_at : str | None
            Next due time timestamp string. If `None`, the current occurrence is
            treated as the last finite occurrence and no next due time is stored.

        Returns
        -------
        bool
            True if the active automation occurrence was advanced, otherwise
            False.
        """

    @abstractmethod
    def finish_automation(
        self,
        automation_id: int,
        *,
        status: Literal[AutomationStatus.COMPLETED, AutomationStatus.FAILED],
    ) -> bool:
        """Finish an active automation with a terminal status.

        Parameters
        ----------
        automation_id : int
            Automation ID to finish.
        status : AutomationStatus
            Terminal target status. Must be `completed` or `failed`.

        Returns
        -------
        bool
            True if the active automation was finished, otherwise False.
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
    def add_task_usage(self, task_id: int, usage: TaskUsage) -> None:
        """Record usage for the specified task.

        Parameters
        ----------
        task_id : int
            The identifier of the task that incurred the usage.
        usage : TaskUsage
            Usage payload to persist.

        Notes
        -----
        Each successful call appends a new usage record for the task.
        """

    @abstractmethod
    def get_task_usage(
        self,
        *,
        run_ids: Sequence[int] | None = None,
        task_ids: Sequence[int] | None = None,
    ) -> Sequence[TaskUsage]:
        """Retrieve task usage records based on the specified filters.

        - If a filter is set to None, it is ignored.
        - If multiple filters are provided, they are combined using AND logic.
        - Within each filter, provided values are combined using OR logic.

        Parameters
        ----------
        run_ids : Optional[Sequence[int]] (default: None)
            Sequence of run IDs to filter by.
        task_ids : Optional[Sequence[int]] (default: None)
            Sequence of task IDs to filter by.

        Returns
        -------
        Sequence[TaskUsage]
            Usage records ordered by insertion order.
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
