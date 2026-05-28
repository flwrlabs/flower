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
"""Tests for AgentSession."""


from __future__ import annotations

import pytest

from flwr.agentapp import AgentSession
from flwr.common.typing import Run
from flwr.supercore.typing import JSONObject


def test_agent_session_stores_constructor_arguments() -> None:
    """Test AgentSession stores its constructor arguments."""
    run = Run.create_empty(1)
    input_items: list[JSONObject] = [{"role": "user", "content": "hi"}]

    session = AgentSession(
        task_id=101,
        run=run,
        agent_ref="test-agent",
        conversation_id="conv-1",
        input_items=input_items,
    )

    assert session.task_id == 101
    assert session.run is run
    assert session.agent_ref == "test-agent"
    assert session.conversation_id == "conv-1"
    assert session.input_items == input_items
