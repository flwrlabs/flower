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
from flwr.supercore.typing import JSONObject
from flwr.supercore.utils import strict_json_dumps

ITEMS_KEY = "items"
JSON_KEY = "json"


def append_items(context: Context, new_items: list[JSONObject]) -> None:
    """Append OpenResponses items to ``context.state``."""
    # Initialize the items storage if it doesn't exist yet
    record = context.state.setdefault(ITEMS_KEY, ConfigRecord({JSON_KEY: []}))
    items = cast(list[str], record[JSON_KEY])

    # Add the new items to the list
    items.extend(strict_json_dumps(item, compact=True) for item in new_items)
