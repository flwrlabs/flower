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
"""Built-in AgentApp FAB resolution for Control StartRun."""

from flwr.cli.build import build_fab_from_files

BUILTIN_GPT_CHAT_AGENT_REF = "gpt-chat"

_GPT_CHAT_PYPROJECT = b"""
[project]
name = "gpt-chat"
version = "0.0.1"

[tool.flwr.app]
publisher = "flwr"

[tool.flwr.app.components]
agentapp = "flwr.agentapp.builtin.gpt_chat:app"
serverapp = "flwr.agentapp.builtin.gpt_chat:app"
clientapp = "flwr.agentapp.builtin.gpt_chat:app"

[tool.flwr.app.config.agent]
ref = "gpt-chat"
input = ""
"""


def resolve_builtin_agent_fab(agent_ref: str) -> tuple[bytes, dict[str, str]]:
    """Resolve built-in AgentApp FAB bytes and verification metadata."""
    if agent_ref != BUILTIN_GPT_CHAT_AGENT_REF:
        raise ValueError(f"Unsupported agent.ref: {agent_ref}.")

    fab_bytes, _ = build_fab_from_files({"pyproject.toml": _GPT_CHAT_PYPROJECT})
    return fab_bytes, {}
