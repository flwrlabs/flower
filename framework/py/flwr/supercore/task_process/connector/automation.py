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

from flwr.supercore.typing import JSONObject

START_AUTOMATION_TOOL_NAME = "start_automation"


def make_start_automation_tool() -> JSONObject:
    """Return the model-facing start-automation tool schema."""
    return {
        "type": "function",
        "name": START_AUTOMATION_TOOL_NAME,
        "description": (
            "Create one automation per explicitly requested future or recurring "
            "schedule. For recurring work, use start_at for the first run and "
            "fixed_interval for later runs; do not create separate one-time "
            "automations. Ask if the start time, recurrence interval, or timezone "
            "for a local clock time is unknown. Do not retry if the creation "
            "outcome is unknown."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "A short, descriptive title for the automation.",
                },
                "input": {
                    "type": "string",
                    "description": (
                        "Work to perform on each run, without scheduling "
                        "instructions or requests to create another automation."
                    ),
                },
                "start_at": {
                    "type": "string",
                    "description": (
                        "First future run as an ISO 8601 timestamp with a UTC "
                        "offset. Resolve local times using the user's timezone."
                    ),
                },
                "fixed_interval": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Elapsed seconds between runs (600 = 10 minutes). Omit "
                        "for one run. Fixed intervals may shift local clock time "
                        "across daylight saving changes."
                    ),
                },
                "max_runs": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Total run limit. Set only when requested; omit for "
                        "unlimited recurrence. A value of 1 means only one run, "
                        "even with fixed_interval."
                    ),
                },
            },
            "required": ["title", "input", "start_at"],
            "additionalProperties": False,
        },
    }
