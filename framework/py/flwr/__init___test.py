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
"""Test for flwr __init__.py."""


import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import semver

_PUBLIC_SUBPACKAGES = [
    "flwr.agentapp",
    "flwr.app",
    "flwr.client",
    "flwr.clientapp",
    "flwr.common",
    "flwr.server",
    "flwr.serverapp",
]
_ROOT_IMPORT_SUBPACKAGES = ["flwr.app", "flwr.common"]


def _fresh_modules(script: str) -> list[str]:
    """Run a fresh interpreter and return the selected loaded modules."""
    source_root = Path(__file__).parents[1]
    python_path = [str(source_root)]
    if existing_python_path := os.environ.get("PYTHONPATH"):
        python_path.append(existing_python_path)
    environment = os.environ | {
        "PYTHONPATH": os.pathsep.join(python_path)
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    return cast(list[str], json.loads(result.stdout))


def test_version() -> None:
    """Tests if version is correctly imported."""
    # Execute
    from flwr import __version__  # pylint: disable=import-outside-toplevel

    # Assert
    semver.VersionInfo.parse(__version__)


def test_unrelated_public_subpackages_are_lazily_imported() -> None:
    """Verify importing flwr does not import unrelated public subpackages."""
    loaded_modules = _fresh_modules(
        "import json, sys\n"
        "import flwr\n"
        f"public = {json.dumps(_PUBLIC_SUBPACKAGES + _ROOT_IMPORT_SUBPACKAGES)}\n"
        "print(json.dumps([name for name in public if name in sys.modules]))\n"
    )

    assert set(loaded_modules) == set(_ROOT_IMPORT_SUBPACKAGES)


def test_public_subpackages_appear_in_dir() -> None:
    """Verify interactive completion includes lazy public subpackages."""
    public_names = [
        module.rsplit(".", maxsplit=1)[-1] for module in _PUBLIC_SUBPACKAGES
    ]
    visible_names = _fresh_modules(
        "import json, flwr\n"
        f"names = {json.dumps(public_names)}\n"
        "print(json.dumps([name for name in names if name in dir(flwr)]))\n"
    )

    assert visible_names == public_names


def test_public_subpackages_remain_available() -> None:
    """Verify lazily imported public subpackages retain their import paths."""
    loaded_modules = _fresh_modules(
        "import json, sys\n"
        "from flwr import agentapp, app, client, clientapp, common, server, serverapp\n"
        f"public = {json.dumps(_PUBLIC_SUBPACKAGES)}\n"
        "print(json.dumps([name for name in public if name in sys.modules]))\n"
    )

    assert loaded_modules == _PUBLIC_SUBPACKAGES
