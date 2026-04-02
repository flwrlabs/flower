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
"""Abstract base class ExecPlugin."""


from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from flwr.common.typing import Run


class ExecPlugin(ABC):
    """Abstract base class for SuperExec plugins."""

    def __init__(
        self,
        appio_api_address: str,
        get_run: Callable[[int], Run],
    ) -> None:
        self.appio_api_address = appio_api_address
        self.get_run = get_run

    @abstractmethod
    def select_run_id(self, candidate_run_ids: Sequence[int]) -> int | None:
        """Select a run ID to execute from a sequence of candidates.

        A candidate run ID is one that has at least one pending message and is
        not currently in progress (i.e., not associated with a token).

        Parameters
        ----------
        candidate_run_ids : Sequence[int]
            A sequence of candidate run IDs to choose from.

        Returns
        -------
        Optional[int]
            The selected run ID, or None if no suitable candidate is found.
        """

    @abstractmethod
    def launch_app(self, token: str, run_id: int) -> None:
        """Launch the application associated with a given run ID and token.

        This method starts the application process using the given `token`.
        The `run_id` is used solely for bookkeeping purposes, allowing any
        plugin implementation to associate this launch with a specific run.

        Parameters
        ----------
        token : str
            The token required to run the application.
        run_id : int
           The ID of the run associated with the token, used for tracking or
           logging purposes.
        """

    # This method is optional to implement
    def load_config(self, yaml_config: dict[str, Any]) -> None:
        """Load configuration from a YAML dictionary.

        Parameters
        ----------
        yaml_config : dict[str, Any]
            A dictionary representing the YAML configuration.
        """
