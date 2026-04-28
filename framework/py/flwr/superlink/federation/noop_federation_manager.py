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
"""NoOp implementation of FederationManager."""


from typing import cast

from flwr.common.constant import NOOP_ACCOUNT_NAME, NOOP_FLWR_AID
from flwr.common.typing import Federation
from flwr.proto.federation_config_pb2 import SimulationConfig  # pylint: disable=E0611
from flwr.proto.federation_pb2 import (  # pylint: disable=E0611
    Account,
    Invitation,
    Member,
)
from flwr.supercore.constant import (
    DEFAULT_SIMULATION_CONFIG,
    NOOP_FEDERATION,
    NOOP_FEDERATION_DESCRIPTION,
    ActionType,
)
from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.typing import ActionContext

from .federation_manager import FederationManager


class UnsupportedError(FlowerError):
    """Exception raised when operation is unsupported by NoOpFederationManager."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            code=ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT,
        )


class NoOpFederationManager(FederationManager):
    """No-Op FederationManager implementation."""

    def __init__(self, simulation: bool = False) -> None:
        self._simulation = simulation
        self._simulation_config: SimulationConfig | None = None
        if self._simulation:
            self._simulation_config = SimulationConfig()
            self._simulation_config.CopyFrom(DEFAULT_SIMULATION_CONFIG)

    def exists(self, federation: str) -> bool:
        """Check if a federation exists."""
        return federation == NOOP_FEDERATION

    def has_member(self, flwr_aid: str, federation: str) -> bool:
        """Check if the given account is a member of the federation."""
        if not self.exists(federation):
            raise ValueError(f"Federation '{federation}' does not exist.")
        return flwr_aid == NOOP_FLWR_AID

    def filter_nodes(self, node_ids: set[int], federation: str) -> set[int]:
        """Given a list of node IDs, return sublist with nodes in federation."""
        if not self.exists(federation):
            raise ValueError(f"Federation '{federation}' does not exist.")
        return node_ids

    def has_node(self, node_id: int, federation: str) -> bool:
        """Given a node ID, check if it is in the federation."""
        if not self.exists(federation):
            raise ValueError(f"Federation '{federation}' does not exist.")
        return True

    def get_federations(self, flwr_aid: str) -> list[Federation]:
        """Get federations of which the account is a member."""
        if flwr_aid != NOOP_FLWR_AID:
            return []
        return [
            Federation(
                name=NOOP_FEDERATION,
                description=NOOP_FEDERATION_DESCRIPTION,
                members=[],
                nodes=[],
                runs=[],
                archived=False,
                simulation=self._simulation,
                config=self._simulation_config,
            )
        ]

    def get_details(self, federation: str) -> Federation:
        """Get details of the federation."""
        if federation != NOOP_FEDERATION:
            raise ValueError(f"Federation '{federation}' does not exist.")

        runs = list(self.linkstate.get_run_info(flwr_aids=[NOOP_FLWR_AID]))
        nodes = list(self.linkstate.get_node_info(owner_aids=[NOOP_FLWR_AID]))
        only_account = Account(id=NOOP_FLWR_AID, name=NOOP_ACCOUNT_NAME)
        return Federation(
            name=NOOP_FEDERATION,
            description=NOOP_FEDERATION_DESCRIPTION,
            members=[
                Member(account=only_account, role="owner"),
            ],
            nodes=nodes,
            runs=runs,
            archived=False,
            simulation=self._simulation,
            config=self._simulation_config,
        )

    def get_simulation_config(self, federation: str) -> SimulationConfig | None:
        """Get the simulation configuration."""
        if federation != NOOP_FEDERATION:
            raise FlowerError(
                ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PERMISSION,
                f"Simulation configuration unavailable for federation '{federation}'.",
            ) from None

        if not self._simulation:
            return None

        config = SimulationConfig()
        config.CopyFrom(cast(SimulationConfig, self._simulation_config))
        return config

    def set_simulation_config(
        self, flwr_aid: str, federation: str, config: SimulationConfig
    ) -> None:
        """Set the simulation configuration."""
        _ = flwr_aid
        if federation != NOOP_FEDERATION or not self._simulation:
            raise FlowerError(
                ApiErrorCode.FEDERATION_NOT_FOUND_OR_NO_PERMISSION,
                f"Cannot set simulation configuration for federation '{federation}'.",
            ) from None
        cast(SimulationConfig, self._simulation_config).MergeFrom(config)

    def create_federation(
        self,
        flwr_aid: str,
        name: str,
        description: str,
        simulation: bool | None = None,
    ) -> Federation:
        """Create a new federation."""
        raise UnsupportedError(
            "`create_federation` is not supported by NoOpFederationManager."
        )

    def archive_federation(self, flwr_aid: str, name: str) -> None:
        """Archive an existing federation."""
        raise UnsupportedError(
            "`archive_federation` is not supported by NoOpFederationManager."
        )

    def add_supernode(self, flwr_aid: str, federation: str, node_id: int) -> None:
        """Add a SuperNode to a federation."""
        raise UnsupportedError(
            "`add_supernode` is not supported by NoOpFederationManager."
        )

    def remove_supernode(self, flwr_aid: str, federation: str, node_id: int) -> None:
        """Remove a SuperNode from a federation."""
        raise UnsupportedError(
            "`remove_supernode` is not supported by NoOpFederationManager."
        )

    def remove_account(
        self, flwr_aid: str, federation: str, target_account_name: str | None
    ) -> str:
        """Remove an account from a federation."""
        raise UnsupportedError(
            "`remove_account` is not supported by NoOpFederationManager."
        )

    def create_invitation(
        self, flwr_aid: str, federation: str, invitee_account_name: str
    ) -> None:
        """Create an invitation for an account to join a federation."""
        raise UnsupportedError(
            "`create_invitation` is not supported by NoOpFederationManager."
        )

    def list_invitations(
        self, flwr_aid: str
    ) -> tuple[list[Invitation], list[Invitation]]:
        """List invitations visible to the given account."""
        raise UnsupportedError(
            "`list_invitations` is not supported by NoOpFederationManager."
        )

    def accept_invitation(self, flwr_aid: str, federation: str) -> None:
        """Accept a pending invitation to join a federation."""
        raise UnsupportedError(
            "`accept_invitation` is not supported by NoOpFederationManager."
        )

    def reject_invitation(self, flwr_aid: str, federation: str) -> None:
        """Reject a pending invitation to join a federation."""
        raise UnsupportedError(
            "`reject_invitation` is not supported by NoOpFederationManager."
        )

    def revoke_invitation(
        self, flwr_aid: str, federation: str, invitee_account_name: str
    ) -> None:
        """Revoke a pending invitation."""
        raise UnsupportedError(
            "`revoke_invitation` is not supported by NoOpFederationManager."
        )

    def report_run_usage(self) -> None:
        """Call hook to report usage for runs.

        This method is called on successful run status transition to FINISHED and when
        runs are marked as failed due to expired tokens.
        """

    def can_execute(
        self, flwr_aid: str, action: ActionType, context: ActionContext
    ) -> None:
        """Check if an account can execute an action under a given context."""
        _ = (flwr_aid, action, context)
