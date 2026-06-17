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
"""Connector registry."""

from collections.abc import Callable

from flwr.supercore.typing import JSONObject, JSONValue

from . import web_search

ConnectorHandler = Callable[..., JSONValue]

_CONNECTOR_HANDLERS: dict[str, ConnectorHandler] = {
    web_search.WEB_SEARCH_CONNECTOR_NAME: web_search.search,
}


def invoke_connector(name: str, arguments: JSONObject) -> JSONValue:
    """Invoke one connector by name."""
    handler = _CONNECTOR_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unsupported connector '{name}'.")
    return handler(**arguments)
