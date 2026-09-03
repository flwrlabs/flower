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
"""Notion action definitions aligned with Open Connector."""

from flwr.supercore.typing import JSONObject

from ..definition import ActionAccess, ActionDefinition

_NOTION_VALUE: JSONObject = {"description": "A Notion API field value."}
_NOTION_PARENT: JSONObject = {
    "type": "object",
    "additionalProperties": _NOTION_VALUE,
    "description": "The official Notion parent object.",
}
_NOTION_PROPERTIES: JSONObject = {
    "type": "object",
    "additionalProperties": _NOTION_VALUE,
    "description": "Notion properties keyed by property name.",
}
_RICH_TEXT: JSONObject = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": _NOTION_VALUE,
        "description": "A Notion API object.",
    },
    "description": "Notion rich text objects.",
}
_PAGE: JSONObject = {
    "type": "object",
    "properties": {
        "object": {
            "const": "page",
            "type": "string",
            "description": "The Notion object type.",
        },
        "id": {"type": "string", "description": "The page ID."},
        "created_time": {
            "type": "string",
            "format": "date-time",
            "description": "The time when the page was created.",
        },
        "last_edited_time": {
            "type": "string",
            "format": "date-time",
            "description": "The time when the page was last edited.",
        },
        "parent": _NOTION_PARENT,
        "properties": _NOTION_PROPERTIES,
        "url": {
            "type": "string",
            "format": "uri",
            "description": "The canonical Notion URL for the page.",
        },
        "archived": {
            "type": "boolean",
            "description": "Whether the page is archived.",
        },
        "in_trash": {
            "type": "boolean",
            "description": "Whether the page is in the trash.",
        },
    },
    "additionalProperties": True,
    "description": "A Notion page object.",
}
_DATA_SOURCE: JSONObject = {
    "type": "object",
    "properties": {
        "object": {
            "const": "data_source",
            "type": "string",
            "description": "The Notion object type.",
        },
        "id": {"type": "string", "description": "The data source ID."},
        "title": _RICH_TEXT,
        "properties": _NOTION_PROPERTIES,
        "parent": _NOTION_PARENT,
        "url": {
            "type": "string",
            "format": "uri",
            "description": "The canonical Notion URL for the data source.",
        },
        "in_trash": {
            "type": "boolean",
            "description": "Whether the data source is in the trash.",
        },
    },
    "additionalProperties": True,
    "description": "A Notion data source object.",
}
_BLOCK: JSONObject = {
    "type": "object",
    "properties": {
        "object": {
            "const": "block",
            "type": "string",
            "description": "The Notion object type.",
        },
        "id": {"type": "string", "description": "The block ID."},
        "parent": _NOTION_PARENT,
        "type": {"type": "string", "description": "The block type."},
        "has_children": {
            "type": "boolean",
            "description": "Whether this block has child blocks.",
        },
        "in_trash": {
            "type": "boolean",
            "description": "Whether the block is in the trash.",
        },
    },
    "additionalProperties": True,
    "description": "A Notion block object.",
}


def _list_output(items: JSONObject, description: str) -> JSONObject:
    """Build the exact Open Connector Notion list output schema."""
    return {
        "type": "object",
        "properties": {
            "object": {
                "const": "list",
                "type": "string",
                "description": "The Notion object type.",
            },
            "results": {
                "type": "array",
                "items": items,
                "description": "Returned Notion objects.",
            },
            "next_cursor": {
                "anyOf": [
                    {
                        "type": "string",
                        "description": "Cursor for the next page.",
                    },
                    {"type": "null"},
                ]
            },
            "has_more": {
                "type": "boolean",
                "description": "Whether more results are available.",
            },
        },
        "additionalProperties": True,
        "required": ["object", "results", "has_more"],
        "description": description,
    }


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
                "query": {
                    "type": "string",
                    "description": "The search query text.",
                },
                "filter": {
                    "type": "object",
                    "additionalProperties": _NOTION_VALUE,
                    "description": "The filter object to narrow results.",
                },
                "sort": {
                    "type": "object",
                    "additionalProperties": _NOTION_VALUE,
                    "description": "The sort object to order results.",
                },
                "pageSize": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "The number of results per page.",
                },
                "startCursor": {
                    "type": "string",
                    "description": "The cursor for pagination.",
                },
            },
            "additionalProperties": False,
            "required": ["query"],
            "description": "The input payload for this action.",
        },
        output_schema=_list_output(
            {"anyOf": [_PAGE, _DATA_SOURCE]},
            "Search results returned by Notion.",
        ),
    ),
    ActionDefinition(
        name="get_page",
        description=(
            "Get a Notion page together with its first-level child blocks. This is "
            "an aggregate helper over page retrieval plus block-children listing."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "pageId": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The page ID to retrieve.",
                }
            },
            "additionalProperties": False,
            "required": ["pageId"],
            "description": "The input payload for this action.",
        },
        output_schema={
            "type": "object",
            "properties": {
                "page": _PAGE,
                "block_children": _list_output(_BLOCK, "First-level child blocks."),
            },
            "additionalProperties": False,
            "required": ["page", "block_children"],
            "description": "Page with child block list.",
        },
    ),
)
