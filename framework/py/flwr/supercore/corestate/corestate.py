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

from flwr.common.constant import HEARTBEAT_CLIENTAPP_LEASE

from ..object_store import ObjectStore


class CoreState(ABC):
    """Abstract base class for core state."""

    def set_clientapp_token_lease(self, lease_seconds: int) -> None:
        """Set the lease duration used for ClientApp heartbeat tokens."""
        if lease_seconds <= 0:
            raise ValueError("ClientApp token lease must be positive")
        self._clientapp_token_lease = lease_seconds

    @property
    def clientapp_token_lease(self) -> int:
        """Return the configured ClientApp heartbeat token lease."""
        return getattr(self, "_clientapp_token_lease", HEARTBEAT_CLIENTAPP_LEASE)

    @property
    @abstractmethod
    def object_store(self) -> ObjectStore:
        """Return the ObjectStore instance used by this CoreState."""

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
