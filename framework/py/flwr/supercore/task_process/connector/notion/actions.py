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
"""Notion action definitions."""

from ..definition import ActionAccess, ActionDefinition
from ..tool_schema import integer_property, string_property

_CURSOR = string_property(
    "Opaque cursor returned in next_cursor by the previous response for the same "
    "search parameters. Omit to retrieve the first page."
)
_PAGE_SIZE = integer_property(
    "Number of results per page. Omit to use Notion's default.",
    minimum=1,
    maximum=100,
)

ACTIONS = (
    ActionDefinition(
        name="search",
        description=(
            "Search Notion pages and data sources with optional filter, sort, and "
            "pagination controls."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "query": string_property(
                    "Text to match against page and data-source titles. Omit to "
                    "return all content shared with the connection."
                ),
                "filter": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "property": {
                                    "type": "string",
                                    "enum": ["object"],
                                    "description": "Property being filtered.",
                                },
                                "value": {
                                    "type": "string",
                                    "enum": ["page", "data_source"],
                                    "description": "Type of Notion object to return.",
                                },
                                "in_trash": {
                                    "type": "boolean",
                                    "description": (
                                        "Whether to return content in the trash."
                                    ),
                                },
                            },
                            "required": ["property", "value"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "in_trash": {
                                    "type": "boolean",
                                    "description": (
                                        "Whether to return content in the trash."
                                    ),
                                },
                            },
                            "required": ["in_trash"],
                            "additionalProperties": False,
                        },
                    ],
                    "description": (
                        "Use either {property: 'object', value: 'page' or "
                        "'data_source'}, optionally with in_trash, or use "
                        "{in_trash: boolean} by itself."
                    ),
                },
                "sort": {
                    "type": "object",
                    "properties": {
                        "timestamp": {
                            "type": "string",
                            "enum": ["last_edited_time"],
                            "description": "Timestamp used to sort results.",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["ascending", "descending"],
                            "description": "Sort direction.",
                        },
                    },
                    "required": ["timestamp", "direction"],
                    "additionalProperties": False,
                    "description": "Sort results by their last-edited time.",
                },
                "page_size": _PAGE_SIZE,
                "start_cursor": _CURSOR,
            },
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_page",
        description=(
            "Get a Notion page together with all its direct child blocks. Nested "
            "child blocks are not retrieved."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "page_id": string_property("The page ID to retrieve."),
            },
            "required": ["page_id"],
            "additionalProperties": False,
        },
    ),
)
