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

from flwr.supercore.typing import JSONObject

from ..definition import ActionAccess, ActionDefinition
from ..tool_schema import string_property

_CURSOR: JSONObject = {
    "type": "string",
    "description": (
        "Opaque cursor returned in pagination.next_cursor by the previous Attio "
        "response for the same action and filters."
    ),
}
_PAGE_LIMIT: JSONObject = {
    "type": "integer",
    "minimum": 1,
    "description": "The maximum number of items to return.",
}


def _uuid_property(description: str) -> JSONObject:
    """Build an Attio UUID property schema."""
    return {"type": "string", "format": "uuid", "description": description}


ACTIONS = (
    ActionDefinition(
        name="identify",
        description=(
            "Identify the current Attio access token, its workspace, permissions, "
            "and authorizing workspace member."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="get_workspace_member",
        description=(
            "Get an Attio workspace member, including their email address, by UUID."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "workspace_member_id": _uuid_property(
                    "Attio workspace member UUID returned by identify."
                ),
            },
            "required": ["workspace_member_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="search_records",
        description="Search records in Attio.",
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "maxLength": 256,
                    "description": "Attio record search query.",
                },
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "Attio object slug or UUID.",
                    },
                    "minItems": 1,
                    "description": "Attio object types to search.",
                },
                "limit": {
                    **_PAGE_LIMIT,
                    "description": "The maximum number of matches to return.",
                },
                "request_as": {
                    "description": "Context in which to perform the search.",
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string", "enum": ["workspace"]}
                            },
                            "required": ["type"],
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["workspace-member"],
                                },
                                "workspace_member_id": _uuid_property(
                                    "Attio workspace member UUID."
                                ),
                            },
                            "required": ["type", "workspace_member_id"],
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["workspace-member"],
                                },
                                "email_address": {
                                    "type": "string",
                                    "format": "email",
                                    "description": "Attio workspace member email.",
                                },
                            },
                            "required": ["type", "email_address"],
                        },
                    ],
                },
            },
            "required": ["query", "objects", "request_as"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="list_meetings",
        description=(
            "List meetings in the authenticated Attio workspace, optionally filtering "
            "by linked record or participant email address and sorting by start time."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "limit": _PAGE_LIMIT,
                "cursor": _CURSOR,
                "linked_object": string_property(
                    "Attio object slug or ID. Must be provided together with "
                    "linked_record_id."
                ),
                "linked_record_id": _uuid_property(
                    "Attio record UUID. Must be provided together with linked_object."
                ),
                "participants": {
                    "type": "string",
                    "description": "Comma-separated participant email addresses.",
                },
                "sort": {
                    "type": "string",
                    "enum": ["start_asc", "start_desc"],
                    "description": (
                        "Meeting start-time order. Use start_desc for the latest "
                        "meeting."
                    ),
                },
                "ends_from": {
                    "type": ["string", "null"],
                    "description": "Inclusive lower bound for meeting end time.",
                },
                "starts_before": {
                    "type": ["string", "null"],
                    "description": "Exclusive upper bound for meeting start time.",
                },
                "timezone": {
                    "type": "string",
                    "description": (
                        "Timezone for evaluating all-day meeting time filters."
                    ),
                },
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
                "meeting_id": _uuid_property("Attio meeting UUID."),
                "limit": {
                    **_PAGE_LIMIT,
                    "description": "The maximum number of recordings to return.",
                },
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
                "meeting_id": _uuid_property("Attio meeting UUID."),
                "call_recording_id": _uuid_property("Attio call recording UUID."),
                "cursor": _CURSOR,
            },
            "required": ["meeting_id", "call_recording_id"],
            "additionalProperties": False,
        },
    ),
)
