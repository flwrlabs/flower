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
"""Tests for lazy SuperCore runtime imports."""


import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest.mock import Mock

import pytest

from flwr.supercore.task_process.connector.browser_use import (
    invoke_browser_use_provider,
)

_ENTRYPOINT_MODULES = [
    "flwr.supercore.cli.flower_superexec",
    "flwr.supercore.cli.flwr_agentapp",
    "flwr.supercore.cli.flwr_connector",
    "flwr.supercore.cli.flwr_model",
]
_TASK_PROCESS_MODULES = [
    "flwr.supercore.task_process.agent",
    "flwr.supercore.task_process.connector",
    "flwr.supercore.task_process.model",
]
_UNNEEDED_AGENT_RUNNERS = [
    "flwr.supercore.task_process.connector.run_connector",
    "flwr.supercore.task_process.model.run_model",
]
_CONNECTOR_RUNNER = "flwr.supercore.task_process.connector.run_connector"
_MODEL_RUNNER = "flwr.supercore.task_process.model.run_model"
_BROWSER_USE_PROVIDER = "flwr.supercore.task_process.connector.browser_use.browser_use"


def _fresh_modules(script: str) -> list[str]:
    """Run a fresh interpreter and return the selected loaded modules."""
    source_root = Path(__file__).parents[2]
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(
            [str(source_root), os.environ.get("PYTHONPATH", "")]
        )
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    return cast(list[str], json.loads(result.stdout))


def test_console_entrypoints_are_lazily_imported() -> None:
    """Verify importing the CLI package does not import every entrypoint."""
    loaded_modules = _fresh_modules(
        "import json, sys\n"
        "import flwr.supercore.cli\n"
        f"entrypoints = {json.dumps(_ENTRYPOINT_MODULES)}\n"
        "print(json.dumps([name for name in entrypoints if name in sys.modules]))\n"
    )

    assert loaded_modules == []


def test_console_entrypoints_appear_in_dir() -> None:
    """Verify interactive completion includes lazy entrypoints."""
    entrypoint_names = [
        module.rsplit(".", maxsplit=1)[-1] for module in _ENTRYPOINT_MODULES
    ]
    visible_names = _fresh_modules(
        "import json, flwr.supercore.cli as cli\n"
        f"names = {json.dumps(entrypoint_names)}\n"
        "print(json.dumps([name for name in names if name in dir(cli)]))\n"
    )

    assert visible_names == entrypoint_names


def test_console_entrypoint_import_does_not_load_siblings() -> None:
    """Verify resolving flwr-agentapp does not import other entrypoints."""
    loaded_modules = _fresh_modules(
        "import json, sys\n"
        "from flwr.supercore.cli import flwr_agentapp\n"
        f"entrypoints = {json.dumps(_ENTRYPOINT_MODULES)}\n"
        "print(json.dumps([name for name in entrypoints if name in sys.modules]))\n"
    )

    assert loaded_modules == ["flwr.supercore.cli.flwr_agentapp"]


def test_task_process_entrypoints_are_lazily_imported() -> None:
    """Verify importing task_process does not import each process type."""
    loaded_modules = _fresh_modules(
        "import json, sys\n"
        "import flwr.supercore.task_process\n"
        f"task_process_modules = {json.dumps(_TASK_PROCESS_MODULES)}\n"
        "print(json.dumps([name for name in task_process_modules "
        "if name in sys.modules]))\n"
    )

    assert loaded_modules == []


def test_task_process_entrypoints_appear_in_dir() -> None:
    """Verify interactive completion includes lazy task-process entrypoints."""
    visible_names = _fresh_modules(
        "import json, flwr.supercore.task_process as task_process\n"
        "names = ['run_agentapp', 'run_connector', 'run_model']\n"
        "print(json.dumps([name for name in names if name in dir(task_process)]))\n"
    )

    assert visible_names == ["run_agentapp", "run_connector", "run_model"]


def test_agent_entrypoint_does_not_import_other_process_runners() -> None:
    """Verify resolving run_agentapp does not import Model or Connector runners."""
    loaded_modules = _fresh_modules(
        "import json, sys\n"
        "from flwr.supercore.task_process import run_agentapp\n"
        f"unneeded_runners = {json.dumps(_UNNEEDED_AGENT_RUNNERS)}\n"
        "print(json.dumps([name for name in unneeded_runners "
        "if name in sys.modules]))\n"
    )

    assert loaded_modules == []


def test_connector_metadata_does_not_import_connector_runner() -> None:
    """Verify Agent connector metadata imports do not load the task runner."""
    loaded_modules = _fresh_modules(
        "import json, sys\n"
        "import flwr.supercore.task_process.connector.automation\n"
        f"runner = {json.dumps(_CONNECTOR_RUNNER)}\n"
        "print(json.dumps([runner] if runner in sys.modules else []))\n"
    )

    assert loaded_modules == []


def test_browser_use_schema_does_not_import_optional_provider() -> None:
    """Verify Browser Use metadata does not load the optional SDK."""
    loaded_modules = _fresh_modules(
        "import json, sys\n"
        "from flwr.supercore.task_process.connector.browser_use "
        "import make_browser_use_tool\n"
        "make_browser_use_tool()\n"
        f"provider = {json.dumps(_BROWSER_USE_PROVIDER)}\n"
        "print(json.dumps([provider] if provider in sys.modules else []))\n"
    )

    assert loaded_modules == []


def test_browser_use_provider_loads_only_when_invoked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Browser Use loads its optional provider only for a tool call."""
    provider = Mock(return_value={"status": "completed"})
    module = ModuleType(_BROWSER_USE_PROVIDER)
    cast(Any, module).invoke_browser_use_provider = provider
    monkeypatch.setitem(sys.modules, _BROWSER_USE_PROVIDER, module)

    result = invoke_browser_use_provider(
        "Open the Flower documentation.",
        allowed_domains=["flower.ai"],
        model="gpt-5",
        usage_recorder=Mock(),
    )

    assert result == {"status": "completed"}
    provider.assert_called_once()


def test_model_provider_does_not_import_model_runner() -> None:
    """Verify Agent Model requests do not import the Model task runner."""
    loaded_modules = _fresh_modules(
        "import json, sys\n"
        "import flwr.supercore.task_process.model.provider\n"
        f"runner = {json.dumps(_MODEL_RUNNER)}\n"
        "print(json.dumps([runner] if runner in sys.modules else []))\n"
    )

    assert loaded_modules == []
