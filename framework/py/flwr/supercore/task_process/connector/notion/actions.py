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
    "request parameters. Continue only when has_more is true. Omit to retrieve "
    "the first page."
)
_PAGE_SIZE = integer_property(
    "Maximum number of results per page (1-100). Lower values reduce response "
    "size. Omit to use Notion's default.",
    minimum=1,
    maximum=100,
)
ACTIONS = (
    ActionDefinition(
        name="search",
        description=(
            "Search page and data-source titles shared with the Notion connection. "
            "Use short, specific keywords; page body content and workspace users "
            "are not searched. Omit optional filters and sorting unless needed. "
            "Continue with next_cursor only when has_more is true."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "query": string_property(
                    "Non-empty keywords to match against page and data-source "
                    "titles. Omit to browse pages and data sources shared with "
                    "the connection."
                ),
                "filter": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "property": {
                                    "type": "string",
                                    "enum": ["object"],
                                    "description": (
                                        "Use 'object' to filter by object type."
                                    ),
                                },
                                "value": {
                                    "type": "string",
                                    "enum": ["page", "data_source"],
                                    "description": (
                                        "Return only pages or only data sources."
                                    ),
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
                        "{in_trash: boolean} by itself. Keep these fields nested "
                        "inside filter. Omit when no restriction is needed."
                    ),
                },
                "sort": {
                    "type": "object",
                    "properties": {
                        "timestamp": {
                            "type": "string",
                            "enum": ["last_edited_time"],
                            "description": (
                                "Use 'last_edited_time' to order by edit time."
                            ),
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["ascending", "descending"],
                            "description": (
                                "Use 'ascending' for oldest edits first or "
                                "'descending' for newest edits first."
                            ),
                        },
                    },
                    "required": ["timestamp", "direction"],
                    "additionalProperties": False,
                    "description": (
                        "Sort results by their last-edited time. Omit to use "
                        "Notion's default ordering."
                    ),
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
            "its complete value. Use notion_get_block_children with the page ID "
            "to read page content."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "page_id": string_property(
                    "The page ID returned by notion_search or another Notion "
                    "response. Provide the ID, not a page URL."
                ),
                "filter_properties": {
                    "type": "array",
                    "items": string_property(
                        "A property ID from properties.<property name>.id. "
                        "URL-encoded IDs are accepted."
                    ),
                    "maxItems": 100,
                    "description": (
                        "Property IDs to include in the response. Omit to return all "
                        "available properties. Use IDs from a notion_get_page "
                        "response, not property names."
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
                    "The ID of the page containing the property, as returned by "
                    "notion_get_page. Provide the ID, not a page URL."
                ),
                "property_id": string_property(
                    "The stable property ID found at properties.<property name>.id "
                    "in the notion_get_page response. This is not the property name, "
                    "type, or value. URL-encoded property IDs are accepted."
                ),
                "page_size": _PAGE_SIZE,
                "start_cursor": _CURSOR,
            },
            "required": ["page_id", "property_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_database",
        description=(
            "Retrieve a Notion database container, including its metadata and "
            "data source IDs and names. This does not return database rows or "
            "data-source property schemas. Database container IDs and data-source "
            "IDs identify different objects."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "database_id": string_property(
                    "The database container ID from a Notion response, such as "
                    "a page's parent.database_id. Provide the ID, not a database "
                    "URL or a data-source ID."
                ),
            },
            "required": ["database_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_block",
        description=(
            "Retrieve a single Notion block and its type-specific content. "
            "If has_children is true, use "
            "notion_get_block_children with the block ID to retrieve its direct "
            "children."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "block_id": string_property(
                    "The block ID from a Notion block response, such as a result "
                    "from notion_get_block_children. Provide the ID, not a URL."
                ),
            },
            "required": ["block_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_block_children",
        description=(
            "Retrieve one page of direct children for a Notion block or page. This "
            "does not retrieve nested descendants. Continue with next_cursor only "
            "when has_more is true. For a returned block with has_children set to "
            "true, call this action again with that block's ID."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "block_id": string_property(
                    "The block or page ID whose direct children should be "
                    "retrieved. Use a page ID to read its top-level content or "
                    "a block ID to read that block's children. Provide the ID, "
                    "not a URL."
                ),
                "page_size": _PAGE_SIZE,
                "start_cursor": _CURSOR,
            },
            "required": ["block_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="list_users",
        description=(
            "List workspace members and bots, including their IDs, types, names, "
            "and emails when available. Guests are excluded. Requires user "
            "information capabilities; personal access tokens cannot use this "
            "action. Continue with next_cursor only when has_more is true. "
            "Use notion_get_user for a known user ID."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "page_size": integer_property(
                    "Maximum number of users per page (1-100; default 100). "
                    "Lower values reduce response size.",
                    minimum=1,
                    maximum=100,
                ),
                "start_cursor": _CURSOR,
            },
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_user",
        description=(
            "Retrieve a workspace person or bot by known user ID, including "
            "profile fields when available. Use notion_list_users to discover "
            "member and bot IDs, or IDs from other Notion responses for guests. "
            "Use notion_get_self for the identity associated with the access token."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "user_id": string_property(
                    "The user ID returned by notion_list_users or another Notion "
                    "response. Provide an ID, not a name, email, or 'self'."
                ),
            },
            "required": ["user_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_self",
        description=(
            "Retrieve the identity associated with the current access token. "
            "For an OAuth connection, this is the connection's bot user. "
            "Call with no arguments."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
)
