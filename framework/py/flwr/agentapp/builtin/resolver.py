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
"""Built-in AgentApp FAB resolution."""


from importlib.resources import files

from flwr.cli.build import build_fab_from_files

_BUILTIN_AGENT_APP_SPEC_PREFIX = "@flwragent"
_BUILTIN_AGENT_APP_SPEC = f"{_BUILTIN_AGENT_APP_SPEC_PREFIX}/flwr-agent"


def try_resolve_builtin_agent_fab(
    app_spec: str,
) -> tuple[bytes, dict[str, str]] | None:
    """Try to resolve a built-in AgentApp app spec into FAB bytes."""
    if app_spec != _BUILTIN_AGENT_APP_SPEC:
        return None

    builtin_files = files("flwr.agentapp.builtin")
    pyproject_toml = builtin_files.joinpath("pyproject.toml").read_bytes()
    flwr_agent_py = builtin_files.joinpath("flwr_agent.py").read_bytes()
    fab_file, _ = build_fab_from_files(
        {
            "pyproject.toml": pyproject_toml,
            "flwr_agent.py": flwr_agent_py,
        }
    )
    return fab_file, {}
