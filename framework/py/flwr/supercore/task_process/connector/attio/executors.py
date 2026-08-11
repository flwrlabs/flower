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
"""Attio action executors."""

from urllib.parse import quote

from flwr.supercore.typing import JSONObject

from ..definition import ConnectorExecutionContext, ConnectorExecutor
from ..http import ConnectorApiError
from ..json_utils import optional_string, require_int_range, require_string


class AttioApiError(ConnectorApiError):
    """Secret-safe Attio API failure."""

    provider = "Attio"


def search_records(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Search records in one Attio workspace."""
    objects = arguments.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("Attio objects must be a non-empty list.")
    return _call_attio_api(
        "POST",
        "/objects/records/search",
        context,
        json_body={
            "query": require_string(arguments.get("query"), "Attio", "query"),
            "objects": [require_string(item, "Attio", "object") for item in objects],
            "request_as": {"type": "workspace"},
            "limit": _limit(arguments, default=25, maximum=25),
        },
    )


def list_meetings(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """List meetings in one Attio workspace."""
    return _call_attio_api(
        "GET",
        "/meetings",
        context,
        params={
            "limit": str(_limit(arguments, default=50, maximum=50)),
            "cursor": _optional(arguments, "cursor"),
            "linked_object": _optional(arguments, "linked_object"),
            "linked_record_id": _optional(arguments, "linked_record_id"),
            "participants": _optional(arguments, "participants"),
        },
    )


def list_call_recordings(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """List call recordings for one Attio meeting."""
    meeting_id = _path_segment(arguments.get("meeting_id"), "meeting_id")
    return _call_attio_api(
        "GET",
        f"/meetings/{meeting_id}/call_recordings",
        context,
        params={
            "limit": str(_limit(arguments, default=50, maximum=200)),
            "cursor": _optional(arguments, "cursor"),
        },
    )


def get_call_transcript(
    arguments: JSONObject, context: ConnectorExecutionContext
) -> JSONObject:
    """Read one page of transcript segments for an Attio call recording."""
    meeting_id = _path_segment(arguments.get("meeting_id"), "meeting_id")
    recording_id = _path_segment(
        arguments.get("call_recording_id"), "call_recording_id"
    )
    return _call_attio_api(
        "GET",
        f"/meetings/{meeting_id}/call_recordings/{recording_id}/transcript",
        context,
        params={"cursor": _optional(arguments, "cursor")},
    )


EXECUTORS: dict[str, ConnectorExecutor] = {
    "search_records": search_records,
    "list_meetings": list_meetings,
    "list_call_recordings": list_call_recordings,
    "get_call_transcript": get_call_transcript,
}


def _call_attio_api(
    method: str,
    path: str,
    context: ConnectorExecutionContext,
    *,
    params: dict[str, str | None] | None = None,
    json_body: JSONObject | None = None,
) -> JSONObject:
    """Call one Attio REST endpoint."""
    if context.http is None:
        raise RuntimeError("Attio HTTP client is not configured.")
    return context.http.request(
        method,
        path,
        error=AttioApiError,
        params={k: v for k, v in (params or {}).items() if v is not None},
        json=json_body,
        http_error_code=lambda response: (
            "rate_limited" if response.status_code == 429 else "http_error"
        ),
    )


def _path_segment(value: object, name: str) -> str:
    """Validate and encode one Attio path segment."""
    return quote(require_string(value, "Attio", name), safe="")


def _limit(arguments: JSONObject, *, default: int, maximum: int) -> int:
    """Return one validated Attio page limit."""
    return require_int_range(
        arguments.get("limit", default), "Attio", "limit", maximum=maximum
    )


def _optional(arguments: JSONObject, name: str) -> str | None:
    """Return one optional Attio string argument."""
    return optional_string(arguments.get(name), "Attio", name)
