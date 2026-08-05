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
"""Built-in automation tool."""

from flwr.supercore.constant import AUTOMATION_MIN_FIXED_INTERVAL
from flwr.supercore.typing import JSONObject

START_AUTOMATION_TOOL_NAME = "start_automation"


def make_start_automation_tool() -> JSONObject:
    """Return the model-facing start-automation tool schema."""
    return {
        "type": "function",
        "name": START_AUTOMATION_TOOL_NAME,
        "description": (
            "Schedule work only when the user explicitly asks for future or "
            "recurring execution."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "input": {
                    "type": "string",
                    "description": "The input for each automated run.",
                },
                "start_at": {
                    "type": "string",
                    "description": (
                        "When to start, for example 2026-07-28T12:00:00+00:00."
                    ),
                },
                "fixed_interval": {
                    "type": "integer",
                    "minimum": AUTOMATION_MIN_FIXED_INTERVAL,
                    "description": "Seconds between runs. Omit for one execution.",
                },
                "max_runs": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Maximum executions. Valid only with fixed_interval."
                    ),
                },
            },
            "required": ["input", "start_at"],
            "additionalProperties": False,
        },
    }
