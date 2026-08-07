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
# ===============================================================================
"""Attio action definitions."""

from ..definition import ActionAccess, ActionDefinition
from ..tool_schema import integer_property, string_property

_CURSOR = string_property("Cursor returned by the previous Attio response.")
_PAGE_LIMIT = integer_property(
    "Maximum number of results to return.", minimum=1, maximum=50
)

ACTIONS = (
    ActionDefinition(
        name="search_records",
        description="Search records in Attio.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "query": string_property("Attio record search query."),
                "objects": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Attio object types to search.",
                },
                "limit": integer_property(
                    "Maximum number of matches to return.", minimum=1, maximum=25
                ),
            },
            "required": ["query", "objects"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="list_meetings",
        description="List meetings in Attio.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "limit": _PAGE_LIMIT,
                "cursor": _CURSOR,
                "linked_object": string_property("Attio linked object type."),
                "linked_record_id": string_property("Attio linked record ID."),
                "participants": string_property("Meeting participant filter."),
            },
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="list_call_recordings",
        description="List call recordings for an Attio meeting.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "meeting_id": string_property("Attio meeting ID."),
                "limit": integer_property(
                    "Maximum number of recordings to return.", minimum=1, maximum=200
                ),
                "cursor": _CURSOR,
            },
            "required": ["meeting_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_call_transcript",
        description="Read a call transcript from Attio.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "meeting_id": string_property("Attio meeting ID."),
                "call_recording_id": string_property("Attio call recording ID."),
                "cursor": _CURSOR,
            },
            "required": ["meeting_id", "call_recording_id"],
            "additionalProperties": False,
        },
    ),
)
