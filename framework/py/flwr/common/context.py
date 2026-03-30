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
"""Context."""


from dataclasses import dataclass

from flwr.app.user_config import UserConfig

from .record import RecordDict


@dataclass
class Context:
    """Context of your run.

    Parameters
    ----------
    run_id : int
        The ID that identifies the run.
    node_id : int
        The ID that identifies the node.
    node_config : UserConfig
        A config (key/value mapping) unique to the node and independent of the
        `run_config`. This config persists across all runs this node participates in.
    state : RecordDict
        Holds records added by the entity in a given `run_id` and that will stay local.
        This means that the data it holds will never leave the system it's running from.
        This can be used as an intermediate storage or scratchpad when
        executing mods. It can also be used as a memory to access
        at different points during the lifecycle of this entity (e.g. across
        multiple rounds)
    run_config : UserConfig
        A config (key/value mapping) held by the entity in a given `run_id` and that
        will stay local. It can be used at any point during the lifecycle of this entity
        (e.g. across multiple rounds)
    """

    run_id: int
    node_id: int
    node_config: UserConfig
    state: RecordDict
    run_config: UserConfig

    def __init__(  # pylint: disable=too-many-arguments, too-many-positional-arguments
        self,
        run_id: int,
        node_id: int,
        node_config: UserConfig,
        state: RecordDict,
        run_config: UserConfig,
    ) -> None:
        self.run_id = run_id
        self.node_id = node_id
        self.node_config = node_config
        self.state = state
        self.run_config = run_config
