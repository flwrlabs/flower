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
"""File system action definitions."""

from ..definition import ActionAccess, ActionDefinition
from ..tool_schema import string_property

_PATH = string_property("Absolute path inside one of the allowed directories.")

ACTIONS = (
    ActionDefinition(
        name="list_directory",
        description=(
            "List the entries of an allowed directory, sorted by name. Each entry "
            "includes its name and whether it is a file or directory."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {"path": _PATH},
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="read_file",
        description=(
            "Read one UTF-8 text file inside an allowed directory and return its "
            "content."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "path": string_property(
                    "Absolute path of the file to read, inside one of the allowed "
                    "directories."
                ),
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    ),
)
