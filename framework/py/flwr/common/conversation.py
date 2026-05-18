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
"""Conversation payload helpers."""


import json

from flwr.common.typing import ConversationItemPayload


def normalize_conversation_item_json(item_json: str) -> str:
    """Validate and normalize a single conversation item JSON object."""
    try:
        parsed = json.loads(item_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Conversation item must be valid JSON.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Conversation item must be a JSON object.")

    return json.dumps(parsed, separators=(",", ":"), allow_nan=False)


def conversation_item_payloads_from_input_json(
    input_json: str,
) -> list[ConversationItemPayload]:
    """Convert StartRun input JSON into conversation item payloads."""
    try:
        parsed = json.loads(input_json)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Input must be valid JSON.") from exc

    if isinstance(parsed, dict):
        items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        raise ValueError("Input must be a JSON object or a list of JSON objects.")

    payloads = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Input list entries must be JSON objects.")
        payloads.append(
            ConversationItemPayload(
                item_json=json.dumps(item, separators=(",", ":"), allow_nan=False)
            )
        )

    return payloads
