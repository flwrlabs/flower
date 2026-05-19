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


from typing import Any
from unittest.mock import Mock

import pytest

import flwr
from flwr.agentapp import AgentApp, AgentAppError, AgentSession
from flwr.supercore.task_message import JsonObject


def test_agentapp_public_import_path() -> None:
    """Test the public AgentApp import path."""
    assert flwr.agentapp.AgentApp is AgentApp


def test_agentapp_registers_and_calls_main() -> None:
    """Test AgentApp calls the registered main function."""
    app = AgentApp()
    session = Mock(spec=AgentSession)
    calls = []

    @app.main()
    def main(session_arg: AgentSession) -> JsonObject:
        calls.append(session_arg)
        return {"id": "resp-1"}

    result = app(session)

    assert calls == [session]
    assert result == {"id": "resp-1"}


def test_agentapp_rejects_duplicate_main() -> None:
    """Test AgentApp rejects duplicate main registrations."""
    app = AgentApp()

    @app.main()
    def main(_: AgentSession) -> JsonObject:
        return {}

    with pytest.raises(ValueError, match="already registered"):

        @app.main()
        def duplicate(_: AgentSession) -> JsonObject:
            return {}


def test_agentapp_rejects_missing_main() -> None:
    """Test AgentApp requires a main function."""
    app = AgentApp()
    session = Mock(spec=AgentSession)

    with pytest.raises(AgentAppError, match="no main"):
        app(session)


def test_agentapp_rejects_non_object_main_result() -> None:
    """Test AgentApp requires main to return a JSON object."""
    app = AgentApp()
    session = Mock(spec=AgentSession)

    @app.main()
    def main(_: AgentSession) -> Any:
        return None

    with pytest.raises(AgentAppError, match="must return a JSON object"):
        app(session)
