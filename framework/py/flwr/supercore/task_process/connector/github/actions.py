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
"""GitHub action definitions."""

from flwr.supercore.typing import JSONObject

from ..definition import ActionAccess, ActionDefinition
from ..tool_schema import integer_property, string_property

_REPOSITORY: JSONObject = {
    "owner": string_property("GitHub organization or repository owner."),
    "repo": string_property("Public GitHub repository name."),
}

ACTIONS = (
    ActionDefinition(
        name="search_code",
        description="Search code in one public GitHub repository.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                **_REPOSITORY,
                "query": string_property("Code search query without repo qualifier."),
                "limit": integer_property(
                    "Maximum number of matches to return.", minimum=1, maximum=10
                ),
            },
            "required": ["owner", "repo", "query"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_file_content",
        description="Read one UTF-8 text file from a public GitHub repository.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                **_REPOSITORY,
                "path": string_property("Repository-relative path to the file."),
                "ref": string_property("Optional branch, tag, or commit."),
            },
            "required": ["owner", "repo", "path"],
            "additionalProperties": False,
        },
    ),
)
