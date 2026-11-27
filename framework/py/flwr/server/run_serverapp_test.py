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
"""Tests for ServerApp loading."""


from __future__ import annotations

import random
import sys
from pathlib import Path
from unittest.mock import Mock

from flwr.app import Context, RecordDict
from flwr.common.seeding import RANDOM_SEED_KEY, clear_run_seed

from .run_serverapp import run


def test_run_seeds_before_importing_the_server_module(tmp_path: Path) -> None:
    """Module-level initialization sees the configured seed."""
    module_name = "server_seed_mod"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "import random\n"
        "from flwr.serverapp import ServerApp\n"
        "IMPORT_DRAW = random.random()\n"
        "app = ServerApp()\n"
        "\n"
        "@app.main()\n"
        "def main(grid, context):\n"
        "    return None\n",
        encoding="utf-8",
    )
    py_state = random.getstate()
    sys.path.insert(0, str(tmp_path))
    sys.modules.pop(module_name, None)
    try:
        random.seed(21)
        expected = random.random()
        context = Context(
            run_id=1,
            node_id=0,
            node_config={},
            state=RecordDict(),
            run_config={RANDOM_SEED_KEY: 21},
        )
        run(
            grid=Mock(),
            context=context,
            server_app_dir=str(tmp_path),
            server_app_attr=f"{module_name}:app",
        )
        imported = sys.modules[module_name]
        assert imported.IMPORT_DRAW == expected
    finally:
        sys.modules.pop(module_name, None)
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
        random.setstate(py_state)
        clear_run_seed()
