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
"""Generate short RunSeries titles from AgentApp conversation context."""

from __future__ import annotations

from logging import ERROR
from typing import TYPE_CHECKING

from flwr.app import ConfigRecord, Context
from flwr.common.logger import log
from flwr.supercore.typing import JSONObject, JSONValue
from flwr.supercore.utils import strict_json_loads

from .context_items import ITEMS_KEY, JSON_KEY

if TYPE_CHECKING:
    from .session import RuntimeAgentResponses

_TITLE_DEFAULT = "New conversation"
_TITLE_MAX_LENGTH = 80
_TITLE_MODEL = "openai/gpt-5-nano"
_TITLE_SYSTEM_INSTRUCTION = (
    "Create a concise title for this conversation. "
    "Return title only, no quotes, no markdown, maximum 4 words."
)


def select_title_seed(context: Context, current_input: str | None) -> str:
    """Return the first stored user message, or the current AgentApp input."""
    record = context.state.get(ITEMS_KEY)
    if isinstance(record, ConfigRecord):
        raw_items = record.get(JSON_KEY)
        if isinstance(raw_items, list):
            for raw_item in raw_items:
                if not isinstance(raw_item, str):
                    continue
                try:
                    item = strict_json_loads(raw_item)
                except ValueError:
                    continue
                if (
                    isinstance(item, dict)
                    and item.get("type") == "message"
                    and item.get("role") == "user"
                ):
                    text = _extract_message_text(item.get("content"))
                    if text:
                        return text
    return current_input.strip() if current_input else ""


def normalize_title(raw_title: str) -> str:
    """Normalize and bound a generated title."""
    title = " ".join(raw_title.strip().strip('"').strip("'").split())
    if not title:
        return _TITLE_DEFAULT
    if len(title) <= _TITLE_MAX_LENGTH:
        return title
    clipped = title[: _TITLE_MAX_LENGTH - 3].rstrip()
    return f"{clipped}..." if clipped else _TITLE_DEFAULT


def fallback_title(seed: str) -> str:
    """Derive a deterministic title from the first four seed words."""
    words = seed.strip().split()
    if not words:
        return _TITLE_DEFAULT
    return normalize_title(" ".join(words[:4]))


def extract_response_text(response: JSONObject) -> str | None:
    """Extract title text from supported Open Responses response shapes."""
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    output = response.get("output")
    if not isinstance(output, list):
        return None
    text_parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "output_text":
                continue
            text = part.get("text")
            if isinstance(text, str):
                text_parts.append(text)
    joined = "".join(text_parts)
    return joined if joined.strip() else None


def generate_series_description(responses: RuntimeAgentResponses, seed: str) -> str:
    """Generate a title privately, falling back to a seed excerpt on failure."""
    fallback = fallback_title(seed)
    try:
        response = responses.create_private(
            {
                "model": _TITLE_MODEL,
                "instructions": _TITLE_SYSTEM_INSTRUCTION,
                "input": seed,
                "stream": False,
                "max_output_tokens": 32,
                "reasoning": {"effort": "minimal"},
            }
        )
        text = extract_response_text(response)
        return normalize_title(text) if text else fallback
    except Exception as ex:  # pylint: disable=W0718
        log(ERROR, "Failed to generate RunSeries description: %s", ex)
        return fallback


def _extract_message_text(content: JSONValue | None) -> str:
    """Extract text from string or text-part message content."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") not in ("input_text", "output_text"):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            text_parts.append(text.strip())
    return " ".join(text_parts)
