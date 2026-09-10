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

import os
from unittest.mock import Mock, patch

from .conversation_title import (
    generate_series_description,
    generate_series_description_in_background,
    resolve_series_description,
)


def test_generate_series_description_and_fallback() -> None:
    """Model output is normalized and provider errors use a prompt excerpt."""
    response = Mock()
    response.json.return_value = {
        "output": [{"content": [{"type": "output_text", "text": " 'Model title' "}]}]
    }
    env = {
        "FLWR_RUNTIME_BASE_URL": "http://runtime/v1/runtime",
        "FLWR_RUNTIME_API_KEY": "token",
    }

    with (
        patch.dict(os.environ, env),
        patch(
            "flwr.supercore.task_process.agent.conversation_title.httpx.post",
            return_value=response,
        ) as post,
    ):
        assert generate_series_description("A prompt with several words") == (
            "Model title"
        )
        post.side_effect = RuntimeError("provider failed")
        assert generate_series_description("one two three four five") == (
            "one two three four"
        )

    response.raise_for_status.assert_called_once()


def test_generate_series_description_in_background() -> None:
    """Background generation runs in a daemon thread."""
    with (
        patch(
            "flwr.supercore.task_process.agent.conversation_title.generate_series_description",
            return_value="Model title",
        ),
        patch("flwr.supercore.task_process.agent.conversation_title.Thread") as thread,
    ):
        future = generate_series_description_in_background("Prompt")
        thread.call_args.kwargs["target"]()

    thread.assert_called_once_with(
        target=thread.call_args.kwargs["target"],
        name="flwr-conversation-title",
        daemon=True,
    )
    assert resolve_series_description(future) == "Model title"
