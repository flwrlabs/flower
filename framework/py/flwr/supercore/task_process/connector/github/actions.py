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
"""GitHub action definitions aligned with Open Connector."""

from flwr.supercore.typing import JSONObject

from ..definition import ActionAccess, ActionDefinition

_USER_SUMMARY: JSONObject = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "login": {"type": "string"},
        "avatar_url": {"type": "string"},
        "html_url": {"type": "string"},
        "type": {"type": "string"},
    },
    "additionalProperties": True,
}
_SEARCH_ITEM: JSONObject = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "path": {"type": "string"},
        "sha": {"type": "string"},
        "url": {"type": "string"},
        "git_url": {"type": "string"},
        "html_url": {"type": "string"},
        "repository": {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "full_name": {"type": "string"},
                "html_url": {"type": "string"},
                "owner": _USER_SUMMARY,
            },
            "additionalProperties": True,
        },
    },
    "additionalProperties": True,
}

ACTIONS = (
    ActionDefinition(
        name="search_code",
        description="Search GitHub code with GitHub search syntax.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "sort": {"type": "string", "enum": ["indexed", "updated"]},
                "order": {"type": "string", "enum": ["asc", "desc"]},
                "perPage": {"type": "integer"},
                "page": {"type": "integer"},
            },
            "additionalProperties": False,
            "required": ["query"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "total_count": {"type": "integer"},
                "incomplete_results": {"type": "boolean"},
                "items": {"type": "array", "items": _SEARCH_ITEM},
            },
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_file_contents",
        description=(
            "Read a repository file and return both base64 and decoded text when "
            "available."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "owner": {"type": "string", "minLength": 1},
                "repo": {"type": "string", "minLength": 1},
                "path": {"type": "string", "minLength": 1},
                "ref": {"type": "string"},
            },
            "additionalProperties": False,
            "required": ["owner", "repo", "path"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "type": {"const": "file", "type": "string"},
                "name": {"type": "string"},
                "path": {"type": "string"},
                "sha": {"type": "string"},
                "size": {"type": "integer"},
                "html_url": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "download_url": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "content_base64": {"type": "string"},
                "decoded_content": {"type": "string"},
                "encoding": {"type": "string"},
            },
            "additionalProperties": True,
        },
    ),
)
