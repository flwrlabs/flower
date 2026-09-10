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
"""Tests for AgentApp RunSeries title generation."""

from unittest.mock import Mock, patch

from .conversation_title import (
    generate_series_description,
    generate_series_description_in_background,
    resolve_series_description,
)


def test_generate_series_description_and_fallback() -> None:
    """Model output is normalized and provider errors use a prompt excerpt."""
    responses = Mock()
    responses.create.return_value = {
        "output": [{"content": [{"type": "output_text", "text": " 'Model title' "}]}]
    }

    assert (
        generate_series_description(responses, "A prompt with several words")
        == "Model title"
    )
    responses.create.side_effect = RuntimeError("provider failed")

    assert generate_series_description(responses, "one two three four five") == (
        "one two three four"
    )


def test_generate_series_description_in_background() -> None:
    """Background generation runs in a daemon thread."""
    responses = Mock()
    responses.create.return_value = {"output": [{"content": [{"text": "Model title"}]}]}

    with patch("flwr.supercore.task_process.agent.conversation_title.Thread") as thread:
        future = generate_series_description_in_background(responses, "Prompt")
        thread.call_args.kwargs["target"]()

    thread.assert_called_once_with(
        target=thread.call_args.kwargs["target"],
        name="flwr-conversation-title",
        daemon=True,
    )
    assert resolve_series_description(future) == "Model title"
