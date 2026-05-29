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
"""Tests for AgentApp."""


from __future__ import annotations

from unittest.mock import Mock

from flwr.agentapp import AgentApp, AgentSession
from flwr.common import Context, RecordDict


def _context() -> Context:
    """Create an empty Context."""
    return Context(
        run_id=1,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config={},
    )


def test_agentapp_registers_and_calls_main() -> None:
    """Test AgentApp calls the registered main function."""
    app = AgentApp()
    session = Mock(spec=AgentSession)
    context = _context()
    calls = []

    @app.main()
    def main(session_arg: AgentSession, context_arg: Context) -> None:
        calls.append((session_arg, context_arg))

    app(session, context)

    assert calls == [(session, context)]
