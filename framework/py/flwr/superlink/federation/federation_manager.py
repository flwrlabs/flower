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
"""Abstract base class FederationManager."""


from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from flwr.common.typing import Federation
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.federation_pb2 import Invitation  # pylint: disable=E0611
from flwr.supercore.constant import ActionType
from flwr.supercore.typing import ActionContext

if TYPE_CHECKING:
    from flwr.server.superlink.linkstate.linkstate import LinkState


# pylint: disable=too-many-public-methods
class FederationManager(ABC):
    """Abstract base class for FederationManager."""

    @property
    def linkstate(self) -> "LinkState":
        """Return the LinkState instance."""
        if not (ret := getattr(self, "_linkstate", None)):
            raise RuntimeError("linkstate not set. Assign to linkstate property first.")
        return ret  # type: ignore

    @linkstate.setter
    def linkstate(self, linkstate: "LinkState") -> None:
        """Set the LinkState instance."""
        self._linkstate = linkstate

    @abstractmethod
    def exists(self, federation: str) -> bool:
        """Check if a federation exists."""

    @abstractmethod
    def has_member(self, flwr_aid: str, federation: str) -> bool:
        """Check if the given account is a member of the federation."""

    @abstractmethod
    def filter_nodes(self, node_ids: set[int], federation: str) -> set[int]:
        """Given a list of node IDs, return sublist with nodes in federation."""

    @abstractmethod
    def has_node(self, node_id: int, federation: str) -> bool:
        """Given a node ID, check if it is in the federation."""

    @abstractmethod
    def get_federations(self, flwr_aid: str) -> list[Federation]:
        """Get federations of which the account is a member.

        Only the name, description and whether the federation is archived are returned.
        """

    @abstractmethod
    def get_details(self, federation: str) -> Federation:
        """Get details of the federation."""

    @abstractmethod
    def get_simulation_config(self, federation: str) -> SimulationConfig | None:
        """Get the simulation configuration for a federation. This method is called by
        the SuperLink only.

        Note that this method will treat non-simulation federations and non-existent
        federations differently.

        Parameters
        ----------
        federation : str
            The name of the federation.

        Returns
        -------
        SimulationConfig | None
            The simulation configuration stored for the federation. If the federation
            is not configured for simulation, None is returned.

        Raises
        ------
        FlowerError
            If the federation does not exist.
        """

    @abstractmethod
    def set_simulation_config(
        self, flwr_aid: str, federation: str, config: SimulationConfig
    ) -> None:
        """Set the simulation configuration for a federation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account setting the simulation configuration.
        federation : str
            The name of the federation.
        config : SimulationConfig
            The simulation configuration to store for the federation.

        Raises
        ------
        FlowerError
            If the federation does not exist, the caller account is not a
            member, or the federation is not configured for simulation.
        """

    @abstractmethod
    def create_federation(
        self,
        flwr_aid: str,
        name: str,
        description: str,
        simulation: bool | None = None,
    ) -> Federation:
        """Create a new federation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account creating the federation.
        name : str
            The unique name of the federation.
        description : str
            A human-readable description of the federation.
        simulation : bool | None
            Whether this federation is intended for simulation. If unset
            (``None``), the manager assumes a deployment runtime should be used.

        Returns
        -------
        Federation
            The created federation.
        """

    @abstractmethod
    def archive_federation(self, flwr_aid: str, name: str) -> None:
        """Archive an existing federation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account archiving the federation.
        name : str
            The name of the federation to archive.
        """

    @abstractmethod
    def add_supernode(self, flwr_aid: str, federation: str, node_id: int) -> None:
        """Add a SuperNode to a federation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account adding the SuperNode.
        federation : str
            The name of the federation.
        node_id : int
            The ID of the SuperNode to add.
        """

    @abstractmethod
    def remove_supernode(self, flwr_aid: str, federation: str, node_id: int) -> None:
        """Remove a SuperNode from a federation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account removing the SuperNode.
        federation : str
            The name of the federation.
        node_id : int
            The ID of the SuperNode to remove.
        """

    @abstractmethod
    def remove_account(
        self, flwr_aid: str, federation: str, target_account_name: str | None
    ) -> str:
        """Remove an account from a federation.

        If `target_account_name` is `None` the caller removes themselves
        (leave). Otherwise only the owner may remove another account. The
        owner can never be removed. All supernodes owned by the removed
        account are also soft-removed from the federation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account initiating the removal (or leaving).
        federation : str
            The name of the federation.
        target_account_name : str | None
            The name of the account to remove. If `None`, the caller removes
            themselves from the federation. The owner cannot remove themselves.

        Returns
        -------
        str
            The Flower account ID (`flwr_aid`) of the removed account.

        Raises
        ------
        FlowerError
            If the federation does not exist, the target account is not a
            member, the owner tries to remove themselves, or a non-owner
            tries to remove another account.
        """

    @abstractmethod
    def create_invitation(
        self, flwr_aid: str, federation: str, invitee_account_name: str
    ) -> None:
        """Create an invitation for an account to join a federation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account creating the invitation (inviter).
        federation : str
            The name of the federation.
        invitee_account_name : str
            The name of the account being invited.

        Raises
        ------
        ValueError
            If the federation does not exist.
        PermissionError
            If the caller is not the owner, the invitee is already a member,
            or a pending invitation already exists for the invitee.
        """

    @abstractmethod
    def list_invitations(
        self, flwr_aid: str
    ) -> tuple[list[Invitation], list[Invitation]]:
        """List all invitations visible to the given account.

        Returns invitations split into those created by the account
        (as inviter) and those received (as invitee). Each list is
        ordered by creation time (oldest first).

        Parameters
        ----------
        flwr_aid : str
            The ID of the account listing invitations.

        Returns
        -------
        tuple[list[Invitation], list[Invitation]]
            A tuple of (created_invitations, received_invitations).
        """

    @abstractmethod
    def accept_invitation(self, flwr_aid: str, federation: str) -> None:
        """Accept a pending invitation and become a member of the federation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account accepting the invitation (invitee).
        federation : str
            The name of the federation.

        Raises
        ------
        ValueError
            If the federation does not exist, or no pending
            invitation exists for the account in the federation.
        """

    @abstractmethod
    def reject_invitation(self, flwr_aid: str, federation: str) -> None:
        """Reject a pending invitation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account rejecting the invitation (invitee).
        federation : str
            The name of the federation.

        Raises
        ------
        ValueError
            If the federation does not exist, or no pending invitation exists
            for the account in the federation.
        """

    @abstractmethod
    def revoke_invitation(
        self, flwr_aid: str, federation: str, invitee_account_name: str
    ) -> None:
        """Revoke a pending invitation.

        Parameters
        ----------
        flwr_aid : str
            The ID of the account revoking the invitation.
        federation : str
            The name of the federation.
        invitee_account_name : str
            The name of the account whose invitation is being revoked.

        Raises
        ------
        ValueError
            If the federation does not exist, or no pending invitation exists
            for the invitee.
        PermissionError
            If the caller is not an owner of the federation.
        """

    @abstractmethod
    def report_run_usage(self) -> None:
        """Call hook to report usage for runs.

        This method is called on successful run status transition to FINISHED and when
        runs are marked as failed due to expired tokens.
        """

    @abstractmethod
    def can_execute(
        self, flwr_aid: str, action: ActionType, context: ActionContext
    ) -> None:
        """Check if an account can execute an action under a given context.

        Parameters
        ----------
        flwr_aid : str
            Flower account ID of the subject.
        action : ActionType
            The action to authorize.
        context : ActionContext
            Action-specific context required for authorization.

        Raises
        ------
        EntitlementError
            If the action is not allowed under the given entitlement context.
        """
