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
"""Builders for model-facing connector tool schemas."""


from collections.abc import Sequence

from flwr.supercore.typing import JSONObject


def function_tool(
    name: str,
    description: str,
    *,
    properties: JSONObject,
    required: Sequence[str] = (),
) -> JSONObject:
    """Build one strict function-tool schema."""
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
    }


def string_property(description: str) -> JSONObject:
    """Build a non-empty string property schema."""
    return {"type": "string", "minLength": 1, "description": description}


def integer_property(description: str, *, minimum: int, maximum: int) -> JSONObject:
    """Build a bounded integer property schema."""
    return {
        "type": "integer",
        "minimum": minimum,
        "maximum": maximum,
        "description": description,
    }
