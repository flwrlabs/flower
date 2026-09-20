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
from ..tool_schema import string_property

GITHUB_PAGINATION_MINIMUM = 1
GITHUB_PER_PAGE_MAXIMUM = 100

_REPOSITORY: JSONObject = {
    "owner": string_property("GitHub organization or repository owner."),
    "repo": string_property("Public GitHub repository name."),
}

ACTIONS = (
    ActionDefinition(
        name="search_code",
        description="Search GitHub code with GitHub search syntax.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "query": string_property("GitHub code search query."),
                "sort": {
                    "type": "string",
                    "enum": ["indexed"],
                    "description": "Field used to sort results.",
                },
                "order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort direction.",
                },
                "per_page": {
                    "type": "integer",
                    "minimum": GITHUB_PAGINATION_MINIMUM,
                    "maximum": GITHUB_PER_PAGE_MAXIMUM,
                    "description": "Number of results to return per page.",
                },
                "page": {
                    "type": "integer",
                    "minimum": GITHUB_PAGINATION_MINIMUM,
                    "description": "Page number to return.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_file_contents",
        description="Read a repository file.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                **_REPOSITORY,
                "path": string_property("Repository-relative path to the file."),
                "ref": {
                    "type": "string",
                    "description": "Optional branch, tag, or commit.",
                },
            },
            "required": ["owner", "repo", "path"],
            "additionalProperties": False,
        },
    ),
)
