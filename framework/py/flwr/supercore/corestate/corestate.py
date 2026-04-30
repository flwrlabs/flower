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

from flwr.common.typing import Fab
from flwr.proto.task_pb2 import Task  # pylint: disable=E0611

from ..object_store import ObjectStore


class CoreState(ABC):
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
    def create_task(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        task_type: str,
        run_id: int,
        fab_hash: str | None = None,
        model_ref: str | None = None,
        connector_ref: str | None = None,
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
    def get_task_id_by_token(self, token: str) -> int | None:
        """Return the task ID associated with the task token, if valid.

        Parameters
        ----------
        token : str
            The task token to look up.

        Returns
        -------
        Optional[int]
            The task ID if the token is valid, otherwise None.
        """

    @abstractmethod
    def create_token(self, run_id: int) -> str | None:
        """Create a token for the given run ID.

        Parameters
        ----------
        run_id : int
            The ID of the run for which to create a token.

        Returns
        -------
        str
            The newly generated token if one does not already exist
            for the given run ID, otherwise None.
        """

    @abstractmethod
    def verify_token(self, run_id: int, token: str) -> bool:
        """Verify a token for the given run ID.

        Parameters
        ----------
        run_id : int
            The ID of the run for which to verify the token.
        token : str
            The token to verify.

        Returns
        -------
        bool
            True if the token is valid for the run ID, False otherwise.
        """

    @abstractmethod
    def delete_token(self, run_id: int) -> None:
        """Delete the token for the given run ID.

        Parameters
        ----------
        run_id : int
            The ID of the run for which to delete the token.
        """

    @abstractmethod
    def get_run_id_by_token(self, token: str) -> int | None:
        """Get the run ID associated with a given token.

        Parameters
        ----------
        token : str
            The token to look up.

        Returns
        -------
        Optional[int]
            The run ID if the token is valid, otherwise None.
        """

    @abstractmethod
    def acknowledge_app_heartbeat(self, token: str) -> bool:
        """Acknowledge an app heartbeat with the provided token.

        Parameters
        ----------
        token : str
            The token associated with the app.

        Returns
        -------
        bool
            True if the heartbeat is acknowledged successfully, False otherwise.
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
