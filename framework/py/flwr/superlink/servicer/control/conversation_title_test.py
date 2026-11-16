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
"""Tests for RunSeries title generation."""

import os
from unittest.mock import Mock, patch

from flwr.server.superlink.linkstate import LinkState

from .conversation_title import start_title_generation


def test_start_title_generation() -> None:
    """Call the model provider with bounded input and persist its response."""
    state = Mock(spec=LinkState)
    response = Mock()
    response.json.return_value = {
        "output": [{"content": [{"type": "output_text", "text": " Model title "}]}]
    }

    with (
        patch.dict(
            os.environ,
            {"FLWR_MODEL_API_ENDPOINT": "http://model/v1/responses"},
            clear=True,
        ),
        patch(
            "flwr.superlink.servicer.control.conversation_title.requests.post",
            return_value=response,
        ) as post,
        patch("flwr.superlink.servicer.control.conversation_title.Thread") as thread,
    ):
        start_title_generation(state, 33, "x" * 5000)
        thread.call_args.kwargs["target"](*thread.call_args.kwargs["args"])

    assert post.call_args.kwargs["json"]["input"] == "x" * 4096
    assert post.call_args.args == ("http://model/v1/responses",)
    state.set_run_series_description.assert_called_once_with(33, "Model title")
    assert thread.call_args.kwargs["daemon"] is True


def test_default_endpoint_requires_api_key() -> None:
    """Do not send the prompt to the default endpoint without an API key."""
    state = Mock(spec=LinkState)
    with (
        patch.dict(os.environ, {}, clear=True),
        patch(
            "flwr.superlink.servicer.control.conversation_title.requests.post"
        ) as post,
        patch("flwr.superlink.servicer.control.conversation_title.Thread") as thread,
    ):
        start_title_generation(state, 33, "Private prompt")
        thread.call_args.kwargs["target"](*thread.call_args.kwargs["args"])

    post.assert_not_called()
    state.set_run_series_description.assert_not_called()
