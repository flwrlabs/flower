# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""AgentApp session."""


from __future__ import annotations

from flwr.common.typing import Run
from flwr.supercore.typing import JSONObject


class AgentSession:
    """Runtime session passed to AgentApp main functions."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        task_id: int,
        run: Run,
        agent_ref: str,
        conversation_id: str,
        input_items: list[JSONObject],
    ) -> None:
        if task_id <= 0:
            raise ValueError("`task_id` must be greater than zero.")
        if not agent_ref:
            raise ValueError("`agent_ref` must be a non-empty string.")
        if not conversation_id:
            raise ValueError("`conversation_id` must be a non-empty string.")
        if not all(isinstance(item, dict) for item in input_items):
            raise ValueError("`input_items` must be a list of JSON objects.")
        self.task_id = task_id
        self.run = run
        self.agent_ref = agent_ref
        self.conversation_id = conversation_id
        self.input_items = input_items
