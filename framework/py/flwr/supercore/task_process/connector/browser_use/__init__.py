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
"""Built-in browser use connector."""


import os

from flwr.supercore.typing import JSONObject

# Prevent browser-use from configuring logging and duplicating Flower logs
os.environ["BROWSER_USE_SETUP_LOGGING"] = "false"

try:
    from .browser_use import invoke_browser_use_provider
except ImportError:

    def invoke_browser_use_provider(
        task: str,
        allowed_domains: list[str] | None = None,
        model: str | None = None,
    ) -> JSONObject:
        """."""
        raise ImportError(
            "Flower's built-in Browser Use connector requires the optional "
            "'browser-use' dependency. To use this feature, add `flwr[agent]` "
            "to `[project].dependencies` in your Flower App's `pyproject.toml`. "
            "If runtime dependency installation is disabled or unavailable, "
            "install `flwr[agent]` in the runtime environment instead."
        )


BROWSER_USE_CONNECTOR_NAME = "browser_use"


def make_browser_use_tool() -> JSONObject:
    """Return the browser use function tool schema."""
    return {
        "type": "function",
        "name": BROWSER_USE_CONNECTOR_NAME,
        "description": "Use a headless browser to complete a web task.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The browser task to complete.",
                },
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    }


__all__ = [
    "BROWSER_USE_CONNECTOR_NAME",
    "invoke_browser_use_provider",
    "make_browser_use_tool",
]
