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
"""Tests for resolving pending AgentApps and requesting Run admission."""

import hashlib
from typing import Any
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    Agent,
    AgentConfig,
    StartRunRequest,
)
from flwr.server.superlink.linkstate import LinkState
from flwr.supercore.auth.typing import AccountInfo
from flwr.supercore.fab import Fab

from . import control_handlers
from .control_handlers import process_pending_agent_runs, start_run


@pytest.fixture(name="agent_app_config")
def agent_app_bundle_config() -> dict[str, Any]:
    """Return metadata from a resolved AgentApp FAB."""
    return {
        "project": {"name": "app", "version": "1.0"},
        "tool": {
            "flwr": {
                "app": {"publisher": "publisher", "components": {"agentapp": "app:app"}}
            }
        },
    }


def test_process_pending_agent_runs_resolves_app(
    agent_app_config: dict[str, Any],
) -> None:
    """Store the resolved AgentApp and pass its metadata to state admission."""
    state = Mock(spec=LinkState)
    space_id = uuid4()
    agent = Agent(id=7, config=AgentConfig(app_spec="@publisher/app"))
    state.get_agent_run_candidates.return_value = [(space_id, agent)]
    state.store_fab.return_value = "stored-hash"
    content, verifications = b"agent-fab", {"signature": "verified"}
    with (
        patch.object(
            control_handlers,
            "_get_remote_fab",
            return_value=(content, verifications, None),
        ) as resolve,
        patch.object(control_handlers, "get_fab_config", return_value=agent_app_config),
    ):
        process_pending_agent_runs(state)
    resolve.assert_called_once_with(agent.config.app_spec)
    state.store_fab.assert_called_once_with(
        Fab(hashlib.sha256(content).hexdigest(), content, verifications)
    )
    state.create_agent_run.assert_called_once_with(
        space_id,
        agent.id,
        fab_id="publisher/app",
        fab_version="1.0",
        fab_hash="stored-hash",
    )


def test_process_pending_agent_runs_limits_successes(
    agent_app_config: dict[str, Any],
) -> None:
    """Failed and stale candidates do not exhaust the successful-admission limit."""
    state = Mock(spec=LinkState)
    space_id = uuid4()
    state.get_agent_run_candidates.return_value = [
        (space_id, Agent(id=i, config=AgentConfig(app_spec="@publisher/app")))
        for i in range(1, 15)
    ]
    state.create_agent_run.side_effect = [None, 123]
    with (
        patch.object(
            control_handlers,
            "_get_remote_fab",
            side_effect=[ValueError("Unavailable app")] * 11
            + [(b"agent-fab", {}, None)] * 3,
        ) as resolve,
        patch.object(control_handlers, "get_fab_config", return_value=agent_app_config),
    ):
        process_pending_agent_runs(state, limit=1)
    assert resolve.call_count == 13
    assert state.create_agent_run.call_count == 2


@pytest.mark.parametrize("invalid_bundle", ["oversized", "serverapp"])
def test_process_pending_agent_runs_rejects_invalid_bundle(
    agent_app_config: dict[str, Any], invalid_bundle: str
) -> None:
    """Reject oversized FABs and bundles without an AgentApp component."""
    state = Mock(spec=LinkState)
    state.get_agent_run_candidates.return_value = [
        (uuid4(), Agent(id=7, config=AgentConfig(app_spec="@publisher/app")))
    ]
    if invalid_bundle == "serverapp":
        agent_app_config["tool"]["flwr"]["app"]["components"] = {"serverapp": "app:app"}
    with (
        patch.object(
            control_handlers,
            "FAB_MAX_SIZE",
            0 if invalid_bundle == "oversized" else 100,
        ),
        patch.object(
            control_handlers, "_get_remote_fab", return_value=(b"fab", {}, None)
        ),
        patch.object(control_handlers, "get_fab_config", return_value=agent_app_config),
    ):
        process_pending_agent_runs(state)
    state.store_fab.assert_not_called()
    state.create_agent_run.assert_not_called()


def test_start_run_preserves_oversized_fab_response() -> None:
    """Shared FAB preparation preserves the legacy empty response for oversized FABs."""
    state = Mock(spec=LinkState)
    state.federation_manager = Mock()
    request = StartRunRequest(federation="federation")
    request.fab.content = b"oversized"
    with (
        patch.object(control_handlers, "FAB_MAX_SIZE", 0),
        patch.object(control_handlers, "get_fab_config") as parse_config,
    ):
        response = start_run(request, AccountInfo("alice", "alice"), state)
    assert not response.HasField("run_id")
    parse_config.assert_not_called()
    state.create_run.assert_not_called()
