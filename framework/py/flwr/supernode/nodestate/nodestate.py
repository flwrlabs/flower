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

from flwr.supercore.corestate import CoreState
from flwr.supercore.run import Run


class NodeState(CoreState):
    """Abstract base class for node state."""

    @abstractmethod
    def set_node_id(self, node_id: int) -> None:
        """Set the node ID."""

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

        Notes
        -----
        This method is best-effort. Implementations should log and return if
        the message processing start time is unavailable.
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
            The processing duration in seconds, or 0.0 if the duration could
            not be calculated.
        """
