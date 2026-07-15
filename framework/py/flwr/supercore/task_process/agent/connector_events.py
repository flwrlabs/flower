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
"""Connector activity event helpers."""


from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from flwr.supercore.task_process.connector.web_fetch import WEB_FETCH_CONNECTOR_NAME
from flwr.supercore.task_process.connector.web_search import WEB_SEARCH_CONNECTOR_NAME
from flwr.supercore.typing import JSONObject, JSONValue
from flwr.supercore.utils import strict_json_dumps

ConnectorStatus = Literal["started", "completed", "failed"]

_SUPPORTED_CONNECTORS = {WEB_FETCH_CONNECTOR_NAME, WEB_SEARCH_CONNECTOR_NAME}


def call_with_events(
    *,
    name: str,
    call_id: str,
    arguments: JSONObject,
    create_response: Callable[..., JSONValue],
    append_and_push_events: Callable[[list[JSONObject]], None],
    append_context_items: Callable[[list[JSONObject]], None],
) -> JSONObject:
    """Call a connector and emit/persist its activity events."""
    def event(
        status: ConnectorStatus,
        *,
        output: JSONValue = None,
        message: str | None = None,
    ) -> list[JSONObject]:
        if name not in _SUPPORTED_CONNECTORS:
            return []

        payload: JSONObject = {
            "type": f"response.tool_call.{status}",
            "tool_call_id": call_id,
            "connector_ref": name,
            "arguments": arguments,
        }

        query = arguments.get("query")
        if isinstance(query, str) and query:
            payload["query"] = query

        url = arguments.get("url")
        if name == WEB_FETCH_CONNECTOR_NAME and isinstance(url, str) and url:
            payload["links"] = [url]

        if status == "completed":
            payload["output"] = output
        elif status == "failed" and message is not None:
            payload["error"] = {"code": "connector_error", "message": message}

        return [payload]

    append_and_push_events(event("started"))

    try:
        output = create_response(
            name=name,
            call_id=call_id,
            arguments=arguments,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        append_and_push_events(event("failed", message=str(exc)))
        raise

    output_item: JSONObject = {
        "type": "function_call_output",
        "call_id": call_id,
        "output": strict_json_dumps(output, compact=True),
    }
    append_and_push_events(event("completed", output=output))
    append_context_items([output_item])
    return output_item
