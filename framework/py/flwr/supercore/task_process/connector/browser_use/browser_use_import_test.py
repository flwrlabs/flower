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
"""Tests for Browser Use's deferred optional import."""


import json
import os
import subprocess
import sys
from pathlib import Path

_IMPLEMENTATION_MODULE = "flwr.supercore.task_process.connector.browser_use.browser_use"


def test_tool_schema_does_not_import_browser_use_implementation() -> None:
    """Verify registering the tool does not load the optional implementation."""
    source_root = Path(__file__).parents[5]
    python_path = [str(source_root)]
    if existing_python_path := os.environ.get("PYTHONPATH"):
        python_path.append(existing_python_path)
    environment = os.environ | {"PYTHONPATH": os.pathsep.join(python_path)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys\n"
                "from flwr.supercore.task_process.connector.browser_use "
                "import make_browser_use_tool\n"
                "make_browser_use_tool()\n"
                f"print(json.dumps({_IMPLEMENTATION_MODULE!r} in sys.modules))\n"
            ),
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert json.loads(result.stdout) is False
