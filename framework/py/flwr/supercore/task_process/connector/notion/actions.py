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

from flwr.supercore.typing import JSONObject

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
_MEETING_NOTE_PROPERTIES = [
    "title",
    "attendees",
    "created_time",
    "created_by",
    "last_edited_time",
    "last_edited_by",
    "notion://meeting_notes/attendees",
]


def _meeting_note_date_property(description: str) -> JSONObject:
    """Return a calendar-date schema for meeting-note comparisons."""
    return {
        "type": "string",
        "description": description,
        "format": "date",
        "pattern": (
            r"^(?:(?:\d\d[2468][048]|\d\d[13579][26]|\d\d0[48]|"
            r"[02468][048]00|[13579][26]00)-02-29|\d{4}-(?:"
            r"(?:0[13578]|1[02])-(?:0[1-9]|[12]\d|3[01])|"
            r"(?:0[469]|11)-(?:0[1-9]|[12]\d|30)|"
            r"(?:02)-(?:0[1-9]|1\d|2[0-8])))$"
        ),
    }


def _meeting_note_empty_filter_schema() -> JSONObject:
    """Return the schema for checking whether a property is empty."""
    return {
        "type": "object",
        "properties": {
            "operator": {
                "type": "string",
                "enum": ["is_empty", "is_not_empty"],
                "description": (
                    "Whether the property must be empty or set. "
                    "These operators take no value."
                ),
            }
        },
        "required": ["operator"],
        "additionalProperties": False,
    }


def _meeting_note_text_filter_schema() -> JSONObject:
    """Return the schema for comparing a meeting-note title."""
    return {
        "type": "object",
        "properties": {
            "operator": {
                "type": "string",
                "enum": [
                    "string_is",
                    "string_is_not",
                    "string_contains",
                    "string_does_not_contain",
                    "string_starts_with",
                    "string_ends_with",
                ],
                "description": (
                    "How to compare the title. Matching is case-insensitive "
                    "and lexical."
                ),
            },
            "value": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["exact"],
                        "description": "Use exact for a literal comparison value.",
                    },
                    "value": {
                        "type": "string",
                        "description": "The literal text to compare against.",
                    },
                },
                "required": ["type", "value"],
                "additionalProperties": False,
            },
        },
        "required": ["operator", "value"],
        "additionalProperties": False,
    }


def _meeting_note_person_filter_schema() -> JSONObject:
    """Return the schema for comparing meeting-note people properties."""
    return {
        "type": "object",
        "properties": {
            "operator": {
                "type": "string",
                "enum": ["person_contains", "person_does_not_contain"],
                "description": "Whether the property contains the listed people.",
            },
            "value": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["exact"],
                                    "description": (
                                        "Use exact for a literal comparison value."
                                    ),
                                },
                                "value": {
                                    "type": "object",
                                    "properties": {
                                        "table": {
                                            "type": "string",
                                            "enum": ["notion_user"],
                                            "description": "Always 'notion_user'.",
                                        },
                                        "id": {
                                            "type": "string",
                                            "description": (
                                                "A Notion user UUID or user://<uuid>. "
                                                "Use IDs from notion_list_users or "
                                                "another Notion response, not names "
                                                "or email addresses."
                                            ),
                                        },
                                    },
                                    "required": ["table", "id"],
                                    "additionalProperties": False,
                                },
                            },
                            "required": ["type", "value"],
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["relative"],
                                    "description": "Use 'relative' for 'me'.",
                                },
                                "value": {
                                    "type": "string",
                                    "enum": ["me"],
                                    "description": "The connected workspace user.",
                                },
                            },
                            "required": ["type", "value"],
                            "additionalProperties": False,
                        },
                    ]
                },
                "maxItems": 100,
                "description": "The people to compare against.",
            },
        },
        "required": ["operator", "value"],
        "additionalProperties": False,
    }


def _meeting_note_date_filter_schema() -> JSONObject:
    """Return the schema for comparing meeting-note date properties."""
    return {
        "type": "object",
        "properties": {
            "operator": {
                "type": "string",
                "enum": [
                    "date_is",
                    "date_is_before",
                    "date_is_after",
                    "date_is_on_or_before",
                    "date_is_on_or_after",
                ],
                "description": "How to compare the date.",
            },
            "value": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["relative"],
                                "description": (
                                    "Use relative for a date relative to now."
                                ),
                            },
                            "value": {
                                "type": "string",
                                "enum": [
                                    "today",
                                    "tomorrow",
                                    "yesterday",
                                    "one_week_ago",
                                    "one_week_from_now",
                                    "one_month_ago",
                                    "one_month_from_now",
                                ],
                            },
                        },
                        "required": ["type", "value"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["exact"],
                                "description": (
                                    "Use exact for a literal comparison value."
                                ),
                            },
                            "value": {
                                "anyOf": [
                                    {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "enum": ["date"],
                                                "description": (
                                                    "A calendar date without a time."
                                                ),
                                            },
                                            "start_date": _meeting_note_date_property(
                                                "A calendar date in YYYY-MM-DD format."
                                            ),
                                        },
                                        "required": ["type", "start_date"],
                                        "additionalProperties": False,
                                    },
                                    {
                                        "type": "object",
                                        "properties": {
                                            "type": {
                                                "type": "string",
                                                "enum": ["datetime"],
                                                "description": (
                                                    "A date and time in the specified "
                                                    "time zone."
                                                ),
                                            },
                                            "start_date": _meeting_note_date_property(
                                                "A calendar date in YYYY-MM-DD format."
                                            ),
                                            "start_time": {
                                                "type": "string",
                                                "description": (
                                                    "A 24-hour time in HH:MM format."
                                                ),
                                            },
                                            "time_zone": {
                                                "type": "string",
                                                "description": (
                                                    "The IANA time-zone name used to "
                                                    "interpret the date and time."
                                                ),
                                            },
                                        },
                                        "required": [
                                            "type",
                                            "start_date",
                                            "start_time",
                                            "time_zone",
                                        ],
                                        "additionalProperties": False,
                                    },
                                ]
                            },
                        },
                        "required": ["type", "value"],
                        "additionalProperties": False,
                    },
                ],
                "description": "The exact or relative date to compare against.",
            },
            "use_end": {
                "type": "boolean",
                "description": (
                    "Compare against the end of a date range rather than its start."
                ),
            },
        },
        "required": ["operator", "value"],
        "additionalProperties": False,
    }


def _meeting_note_date_range_filter_schema() -> JSONObject:
    """Return the schema for comparing meeting-note date ranges."""
    return {
        "type": "object",
        "properties": {
            "operator": {
                "type": "string",
                "enum": ["date_is_within", "date_is_relative_to"],
                "description": (
                    "How to compare the date against a range. Prefer "
                    "date_is_within for relative windows such as past N days."
                ),
            },
            "value": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["relative"],
                                "description": (
                                    "Use relative for a date relative to now."
                                ),
                            },
                            "value": {
                                "type": "string",
                                "enum": ["custom"],
                                "description": (
                                    "Use custom for a window sized by direction, unit, "
                                    "and count."
                                ),
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["past", "future"],
                                "description": (
                                    "Whether the window runs backwards or "
                                    "forwards from now."
                                ),
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["year", "month", "week", "day"],
                                "description": "The unit used to size the window.",
                            },
                            "count": integer_property(
                                "How many units wide the window is.",
                                minimum=1,
                                maximum=9007199254740991,
                            ),
                        },
                        "required": [
                            "type",
                            "value",
                            "direction",
                            "unit",
                            "count",
                        ],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["relative"],
                                "description": (
                                    "Use relative for a date relative to now."
                                ),
                            },
                            "value": {
                                "type": "string",
                                "enum": ["surrounding"],
                                "description": (
                                    "Use surrounding for a window around now."
                                ),
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["year", "month", "week", "day"],
                                "description": "The unit of the surrounding window.",
                            },
                        },
                        "required": ["type", "value", "unit"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["relative"],
                                "description": (
                                    "Use relative for a date relative to now."
                                ),
                            },
                            "value": {
                                "type": "string",
                                "enum": [
                                    "this_week",
                                    "the_past_week",
                                    "the_past_month",
                                    "the_past_year",
                                    "the_next_week",
                                    "the_next_month",
                                    "the_next_year",
                                ],
                            },
                        },
                        "required": ["type", "value"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["exact"],
                                "description": (
                                    "Use exact for a literal comparison value."
                                ),
                            },
                            "value": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["daterange"],
                                        "description": (
                                            "A date range with optional inclusive "
                                            "boundaries."
                                        ),
                                    },
                                    "start_date": _meeting_note_date_property(
                                        "The inclusive start date in YYYY-MM-DD "
                                        "format, if any. Omit for no start bound."
                                    ),
                                    "end_date": _meeting_note_date_property(
                                        "The inclusive end date in YYYY-MM-DD "
                                        "format, if any. Omit for no end bound."
                                    ),
                                },
                                "required": ["type"],
                                "additionalProperties": False,
                            },
                        },
                        "required": ["type", "value"],
                        "additionalProperties": False,
                    },
                ],
                "description": "The exact or relative date range to compare against.",
            },
            "use_end": {
                "type": "boolean",
                "description": (
                    "Compare against the end of a date range rather than its start."
                ),
            },
        },
        "required": ["operator", "value"],
        "additionalProperties": False,
    }


def _meeting_note_property_filter_schema() -> JSONObject:
    """Return the shared comparison schema for meeting-note properties."""
    return {
        "type": "object",
        "properties": {
            "property": {
                "type": "string",
                "enum": _MEETING_NOTE_PROPERTIES,
                "description": (
                    "Which meeting-note property to filter on. Prefer the short "
                    "names; notion://meeting_notes/attendees is accepted for "
                    "compatibility."
                ),
            },
            "filter": {
                "anyOf": [
                    _meeting_note_text_filter_schema(),
                    _meeting_note_person_filter_schema(),
                    _meeting_note_date_filter_schema(),
                    _meeting_note_date_range_filter_schema(),
                    _meeting_note_empty_filter_schema(),
                ],
                "description": "Use the comparison matching the property's type.",
            },
        },
        "required": ["property", "filter"],
        "additionalProperties": False,
    }


def _meeting_note_combinator_schema(*, allow_nested: bool) -> JSONObject:
    """Return an and/or schema, optionally allowing one nested combinator."""
    items = _meeting_note_property_filter_schema()
    if allow_nested:
        items = {
            "anyOf": [
                items,
                _meeting_note_combinator_schema(allow_nested=False),
            ]
        }
    return {
        "type": "object",
        "properties": {
            "operator": {
                "type": "string",
                "enum": ["and", "or"],
                "description": "Whether every child or any child must match.",
            },
            "filters": {
                "type": "array",
                "maxItems": 100,
                "items": items,
                "description": (
                    "Conditions in this group. Use filters, not operands, and "
                    "an empty group matches nothing. Only the outer group may "
                    "contain nested groups; inner groups contain property filters."
                ),
            },
        },
        "required": ["operator"] if allow_nested else ["operator", "filters"],
        "additionalProperties": {} if allow_nested else False,
    }


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
            "Retrieve one Notion block's metadata and type-specific content. "
            "Use a block ID from a Notion response, including a meeting-note "
            "query result. If has_children is true, use "
            "notion_get_block_children with the block ID to retrieve its direct "
            "children."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "block_id": string_property(
                    "The block ID returned by notion_get_block_children, "
                    "notion_query_meeting_notes, or another Notion response. "
                    "Provide the ID, not a URL."
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
                    "The page or block ID whose direct children should be read. "
                    "For meeting notes, use meeting_notes.children.summary_block_id, "
                    "notes_block_id, or transcript_block_id from the query result. "
                    "Provide the ID, not a URL."
                ),
                "page_size": integer_property(
                    "Maximum number of child blocks per page (1-100). "
                    "Lower values reduce response size. Omit to use Notion's default.",
                    minimum=1,
                    maximum=100,
                ),
                "start_cursor": string_property(
                    "Use next_cursor from the previous response only when "
                    "has_more is true, with the same block_id and page_size. "
                    "Omit to read the first page."
                ),
            },
            "required": ["block_id"],
            "additionalProperties": False,
        },
    ),
    ActionDefinition(
        name="query_meeting_notes",
        description=(
            "Query AI meeting notes available to the integration's workspace user. "
            "The filter is optional. Returns up to 50 matching meeting notes, "
            "already "
            "scoped to the integration's user; do not add a current-user filter "
            "for 'my meetings'. Resolve attendee IDs with notion_list_users or "
            "known IDs from other Notion responses.\n\n"
            "Treat summaries, notes, todos, action items, and deliverables as "
            "requested output, not title terms. Add title filters only when the "
            "user names a meeting. Interpret phrases such as 'meetings this week' "
            "or 'yesterday's meetings' as date filters. Treat a named person as "
            "an attendee or creator unless the user explicitly names a meeting "
            "title. Use the person's name as a title fallback only after attendee "
            "filtering returns no results.\n\n"
            "Title matching is case-insensitive and lexical. Simplify to one "
            "term when no results return. The filter parameter describes property "
            "types, date windows, and Boolean combinations.\n\n"
            "This endpoint has no cursor pagination. If has_more is true, "
            "additional matching meetings exist; narrow filters before making "
            "exhaustive claims. Read content with notion_get_block_children using "
            "meeting_notes.children.summary_block_id, notes_block_id, or "
            "transcript_block_id from each result."
        ),
        access=ActionAccess.READ,
        input_schema={
            "type": "object",
            "properties": {
                "filter": {
                    **_meeting_note_combinator_schema(allow_nested=True),
                    "description": (
                        "An and/or group using a filters array. Wrap a single "
                        "property condition in an and group. Groups may be nested "
                        "one level and contain up to 100 conditions. Empty groups "
                        "match nothing. Omit to query without "
                        "an additional restriction.\n\n"
                        "Properties: title (text); attendees, created_by, and "
                        "last_edited_by (people); created_time and last_edited_time "
                        "(note timestamps). Use is_empty or is_not_empty without "
                        "a value for unset/set properties.\n\n"
                        "Prefer date_is_within for relative windows: this_week, "
                        "the_past_week, or a custom window with direction, unit, "
                        "and count. Exact date ranges use inclusive start_date "
                        "and end_date boundaries, either of which may be omitted. "
                        "Single-date comparisons support exact dates or relative "
                        "shortcuts such as today and yesterday.\n\n"
                        "Split multiword title searches into individual terms. "
                        "Use or for broad discovery and and when all terms must "
                        "match.\n\n"
                        'Title example: {"operator":"and","filters":['
                        '{"property":"title","filter":'
                        '{"operator":"string_contains","value":'
                        '{"type":"exact","value":"standup"}}}]}\n'
                        'Past-week example: {"operator":"and","filters":['
                        '{"property":"created_time","filter":'
                        '{"operator":"date_is_within","value":'
                        '{"type":"relative","value":"the_past_week"}}}]}\n'
                        'Attendee example: {"operator":"and","filters":['
                        '{"property":"attendees","filter":'
                        '{"operator":"person_contains","value":'
                        '[{"type":"exact","value":{"table":"notion_user",'
                        '"id":"<user-id>"}}]}}]}\n'
                        'Combined example: {"operator":"and","filters":['
                        '{"property":"created_time","filter":'
                        '{"operator":"date_is_within","value":'
                        '{"type":"relative","value":"custom",'
                        '"direction":"past","unit":"day","count":3}}},'
                        '{"property":"attendees","filter":'
                        '{"operator":"person_contains","value":'
                        '[{"type":"exact","value":{"table":"notion_user",'
                        '"id":"<user-id>"}}]}}]}'
                    ),
                },
            },
            "additionalProperties": {},
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
