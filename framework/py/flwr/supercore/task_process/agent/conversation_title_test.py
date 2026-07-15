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
"""Tests for AgentApp RunSeries title helpers."""

from unittest.mock import Mock

from flwr.app import ConfigRecord, Context, RecordDict
from flwr.supercore.utils import strict_json_dumps

from .context_items import ITEMS_KEY, JSON_KEY
from .conversation_title import (
    extract_response_text,
    fallback_title,
    generate_series_description,
    normalize_title,
    select_title_seed,
)


def _context(*items: str) -> Context:
    state = RecordDict()
    if items:
        state[ITEMS_KEY] = ConfigRecord({JSON_KEY: list(items)})
    return Context(1, 0, {}, state, {}, series_id=1)


def test_select_title_seed_uses_first_valid_user_message() -> None:
    """Malformed and non-user items are ignored."""
    context = _context(
        "not JSON",
        strict_json_dumps({"type": "message", "role": "assistant", "content": "A"}),
        strict_json_dumps({"type": "message", "role": "user", "content": "First"}),
        strict_json_dumps({"type": "message", "role": "user", "content": "Second"}),
    )

    assert select_title_seed(context, "Current") == "First"


def test_select_title_seed_supports_text_parts_and_current_input() -> None:
    """Text parts are joined and current input is the final fallback."""
    context = _context(
        strict_json_dumps(
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "First part"},
                    {"type": "input_text", "text": "second part"},
                ],
            }
        )
    )
    assert select_title_seed(context, "Current") == "First part second part"
    assert select_title_seed(_context(), "  Current input  ") == "Current input"


def test_normalize_and_fallback_title() -> None:
    """Titles are normalized, bounded, and deterministically derived."""
    assert normalize_title('  "A   concise title"  ') == "A concise title"
    clipped = normalize_title("a" * 90)
    assert len(clipped) == 80
    assert clipped.endswith("...")
    assert fallback_title("one two three four five") == "one two three four"
    assert fallback_title("  ") == "New conversation"


def test_extract_response_text_supports_both_shapes() -> None:
    """Top-level and message content response text can be extracted."""
    assert extract_response_text({"output_text": "Top-level title"}) == (
        "Top-level title"
    )
    assert (
        extract_response_text(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Nested title"}],
                    }
                ]
            }
        )
        == "Nested title"
    )


def test_generate_series_description_uses_private_response() -> None:
    """Successful title generation uses the private model response path."""
    responses = Mock()
    responses.create_private.return_value = {"output_text": "  'Model title'  "}

    title = generate_series_description(responses, "A prompt with several words")

    assert title == "Model title"
    responses.create_private.assert_called_once_with(
        {
            "model": "openai/gpt-5-nano",
            "instructions": (
                "Create a concise title for this conversation. Return title only, "
                "no quotes, no markdown, maximum 4 words."
            ),
            "input": "A prompt with several words",
            "stream": False,
            "max_output_tokens": 32,
            "reasoning": {"effort": "minimal"},
        }
    )


def test_generate_series_description_falls_back_on_model_failure() -> None:
    """Provider errors and empty responses return the prompt excerpt."""
    responses = Mock()
    responses.create_private.side_effect = RuntimeError("provider failed")
    assert generate_series_description(responses, "one two three four five") == (
        "one two three four"
    )

    responses.create_private.side_effect = None
    responses.create_private.return_value = {"output": []}
    assert generate_series_description(responses, "one two three four five") == (
        "one two three four"
    )
