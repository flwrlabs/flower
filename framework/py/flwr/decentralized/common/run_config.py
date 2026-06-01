# Copyright 2026 Inria (cyrille kenfack & davide frey). All Rights Reserved.
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
import dataclasses

from flwr.decentralized.common.typing import Mode


@dataclasses.dataclass
class DLRunConfig:
    rounds: int

    communication_probability: float = 0.5
    n_aggregation_steps: int = 1
    protocol: Mode = Mode.PUSHPULL
    n_nodes_to_share: int = 1

    seed: int = 42

    def get_cycles(self) -> int:
        n_aggregation_steps = self._get_n_aggregation_steps()
        steps_per_round = 1 + n_aggregation_steps if n_aggregation_steps > 0 else 2
        return (
            (self.rounds * steps_per_round) + 1
            if n_aggregation_steps > 0 and self.rounds > 0
            else -1
        )

    def get_steps_per_round(self) -> int:
        n_aggregation_steps = self._get_n_aggregation_steps()
        return 1 + n_aggregation_steps if n_aggregation_steps > 0 else 1

    def _get_n_aggregation_steps(self) -> int:
        """Get the effective number of aggregation steps as an integer."""
        n_aggregation_steps = self.n_aggregation_steps
        return 1 if n_aggregation_steps is None else n_aggregation_steps
