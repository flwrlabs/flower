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
"""OpenResponses item storage helpers for AgentApp context state."""


from __future__ import annotations

from typing import cast

from flwr.app import ConfigRecord, Context
from flwr.supercore.typing import JSONObject, JSONValue
from flwr.supercore.utils import strict_json_dumps, strict_json_loads

ITEMS_KEY = "items"
JSON_KEY = "json"


def get_items(context: Context) -> list[JSONObject]:
    """Return OpenResponses items stored in ``context.state``."""
    record = context.state.config_records.get(ITEMS_KEY)
    if record is None:
        return []

    raw = record.get(JSON_KEY)
    if not isinstance(raw, str):
        raise ValueError("context.state['items'] must contain a JSON string.")

    return _validate_items(strict_json_loads(raw), "context.state['items']")


def append_items(context: Context, new_items: JSONValue) -> None:
    """Append OpenResponses items to ``context.state``."""
    items = [*get_items(context), *_validate_items(new_items, "new items")]
    context.state[ITEMS_KEY] = ConfigRecord(
        {JSON_KEY: strict_json_dumps(cast(JSONValue, items), compact=True)}
    )


def _validate_items(value: JSONValue, source: str) -> list[JSONObject]:
    """Validate a JSON object array."""
    if not isinstance(value, list):
        raise ValueError(f"{source} must be a JSON array of item objects.")

    items: list[JSONObject] = []
    for idx, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{source}[{idx}] must be a JSON object.")
        items.append(cast(JSONObject, item))
    return items
