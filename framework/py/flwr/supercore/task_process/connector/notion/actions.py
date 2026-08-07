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

_CURSOR = string_property("Cursor returned by the previous response.")

ACTIONS = (
    ActionDefinition(
        name="search",
        description="Search pages and data sources shared with Notion.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "query": string_property("Text contained in the Notion title."),
                "limit": integer_property(
                    "Maximum number of results to return.", minimum=1, maximum=100
                ),
                "cursor": _CURSOR,
            },
            "required": ["query"],
        },
    ),
    ActionDefinition(
        name="get_page_content",
        description="Read one page of a Notion page's block content.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "page_id": string_property("Notion page ID returned by search."),
                "max_blocks": integer_property(
                    "Maximum number of blocks to return.", minimum=1, maximum=100
                ),
                "cursor": _CURSOR,
            },
            "required": ["page_id"],
        },
    ),
)
