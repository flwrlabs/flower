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


import subprocess
import sys
from pathlib import Path

import pytest
import semver


def test_version() -> None:
    """Tests if version is correctly imported."""
    # Execute
    from flwr import __version__  # pylint: disable=import-outside-toplevel

    # Assert
    semver.VersionInfo.parse(__version__)


@pytest.mark.parametrize(
    "code",
    [
        "import sys\nimport flwr\n"
        "assert 'flwr.app' not in sys.modules\n"
        "assert 'flwr.supercore.version' not in sys.modules\n"
        "assert set(flwr.__all__).issubset(dir(flwr))",
        "import inspect\nfrom flwr import common\n"
        "assert set(common.__all__).issubset(dict(inspect.getmembers(common)))\n"
        "from flwr.app import Message",
        "from flwr.supercore import log\nimport flwr.supercore.superexec.executor",
        "import inspect\nfrom flwr import server\n"
        "assert set(server.__all__).issubset(dict(inspect.getmembers(server)))\n"
        "from flwr import client",
        "import flwr.compat.server.grpc_bidi.grpc_server",
        "from flwr.client.mod import make_ffn as legacy_make_ffn\n"
        "from flwr.clientapp.mod.utils import make_ffn\n"
        "assert legacy_make_ffn is make_ffn",
        "from flwr.clientapp import ClientApp\n"
        "from flwr.clientapp.mod.utils import make_ffn\n"
        "from flwr.client.mod import make_ffn as legacy_make_ffn\n"
        "assert legacy_make_ffn is make_ffn",
        "import flwr.supercore.agentapp_worker\n"
        "import flwr.supercore.connector_worker",
    ],
)
def test_package_imports_in_fresh_process(code: str) -> None:
    """Keep package imports lazy and independent of application import order."""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parent.parent,
        check=True,
        timeout=30,
    )
