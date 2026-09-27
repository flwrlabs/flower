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
    "request parameters. Omit to retrieve the first page."
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
            "Retrieve a Notion page and its property values. This does not retrieve "
            "page content or child blocks. Some properties can be truncated; use "
            "notion_get_page_property with the property's returned ID when you need "
            "its complete value."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "page_id": string_property("The page ID to retrieve."),
                "filter_properties": {
                    "type": "array",
                    "items": string_property("A property ID to include."),
                    "maxItems": 100,
                    "description": (
                        "Property IDs to include in the response. Omit to return all "
                        "available properties."
                    ),
                },
            },
            "required": ["page_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_page_property",
        description=(
            "Retrieve one property from a Notion page. Title, rich text, people, "
            "relation, and rollup properties can return paginated lists. Pagination "
            "is optional; only continue with next_cursor when has_more is true. A "
            "rollup's calculation is final only on the last page."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "page_id": string_property(
                    "The ID of the page containing the property."
                ),
                "property_id": string_property(
                    "The stable property ID found at properties.<property name>.id "
                    "in the notion_get_page response. This is not the property name, "
                    "type, or value."
                ),
                "page_size": _PAGE_SIZE,
                "start_cursor": _CURSOR,
            },
            "required": ["page_id", "property_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="list_users",
        description=(
            "List workspace users with optional pagination. Personal access tokens "
            "cannot use this action."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "page_size": _PAGE_SIZE,
                "start_cursor": _CURSOR,
            },
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_user",
        description="Retrieve a workspace user by ID.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "user_id": string_property("The user ID to retrieve."),
            },
            "required": ["user_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_self",
        description="Retrieve the user associated with the current access token.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
)
