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
"""Tests for AgentApp run configuration handling."""


from pathlib import Path

import pytest

from flwr.app import Context, RecordDict
from flwr.app.user_config import UserConfig

from .context_items import get_items
from .run_agentapp import _set_agentapp_run_config

_AGENTAPP_PYPROJECT = """
[project]
name = "agentapp-test"
version = "1.0.0"

[tool.flwr.app]
publisher = "flwrlabs"

[tool.flwr.app.components]
agentapp = "agentapp_test:app"

[tool.flwr.app.config.agent]
input = ""
model = "default-model"
instructions = "Answer concisely."
"""


def _context() -> Context:
    """Create a test context."""
    return Context(
        run_id=1,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config={},
    )


def _write_pyproject(app_dir: Path) -> None:
    """Write a minimal AgentApp pyproject.toml."""
    (app_dir / "pyproject.toml").write_text(_AGENTAPP_PYPROJECT, encoding="utf-8")


def test_set_agentapp_run_config_uses_fused_config(tmp_path: Path) -> None:
    """AgentApp should apply overrides through the fused config helper."""
    _write_pyproject(tmp_path)
    context = _context()

    _set_agentapp_run_config(context, str(tmp_path), {"agent.model": "override-model"})

    assert context.run_config == {
        "agent.input": "",
        "agent.model": "override-model",
        "agent.instructions": "Answer concisely.",
    }


@pytest.mark.parametrize(
    ("override_config", "expected_error"),
    [
        pytest.param(
            {"agent.unknown": "unused"},
            "Key 'agent.unknown' is not present in the main dictionary",
            id="unknown-override",
        ),
        pytest.param(
            {"agent.input": 1},
            "context.run_config['agent.input'] must be a string",
            id="non-string-agent-input",
        ),
    ],
)
def test_set_agentapp_run_config_rejects_invalid_config(
    tmp_path: Path, override_config: UserConfig, expected_error: str
) -> None:
    """AgentApp should fail before app execution for invalid run config."""
    _write_pyproject(tmp_path)

    with pytest.raises(ValueError) as exc:
        _set_agentapp_run_config(_context(), str(tmp_path), override_config)

    assert expected_error in str(exc.value)


def test_set_agentapp_run_config_appends_agent_input(tmp_path: Path) -> None:
    """AgentApp should store initial user input as an OpenResponses item."""
    _write_pyproject(tmp_path)
    context = _context()

    _set_agentapp_run_config(context, str(tmp_path), {"agent.input": "Hello"})

    assert get_items(context) == [
        {
            "type": "message",
            "role": "user",
            "content": "Hello",
        }
    ]
