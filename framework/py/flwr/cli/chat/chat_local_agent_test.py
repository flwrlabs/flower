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
"""Tests for local AgentApp helpers in Flower Chat."""

import hashlib
from pathlib import Path
from unittest.mock import patch

from flwr.cli.chat.chat_local_agent import (
    build_local_agent,
    parse_local_agent_path,
)


def test_parse_and_build_local_agent(tmp_path: Path) -> None:
    """Parse a quoted path and build its local AgentApp metadata."""
    config = {
        "project": {"name": "custom-agent", "version": "1.0.0"},
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "local",
                    "components": {"agentapp": "custom_agent.app:app"},
                }
            }
        },
    }
    fab_content = b"local-fab"
    assert parse_local_agent_path('/load "../my agent"') == Path("../my agent")

    with (
        patch(
            "flwr.cli.chat.chat_local_agent.load_and_validate",
            return_value=(config, ["Add a description."]),
        ),
        patch(
            "flwr.cli.chat.chat_local_agent.build_fab_from_disk",
            return_value=fab_content,
        ),
    ):
        local_agent = build_local_agent(tmp_path)

    assert local_agent.path == tmp_path.resolve()
    assert local_agent.app_spec == "@local/custom-agent"
    assert local_agent.fab_hash == hashlib.sha256(fab_content).hexdigest()
    assert local_agent.fab_content == fab_content
    assert local_agent.warnings == ("Add a description.",)
