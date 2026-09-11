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
    """Call the model provider and persist its response in a daemon thread."""
    state = Mock(spec=LinkState)
    response = Mock()
    response.json.return_value = {
        "output": [{"content": [{"type": "output_text", "text": " Model title "}]}]
    }

    with (
        patch.dict(os.environ, {"FLWR_MODEL_API_KEY": "key"}),
        patch(
            "flwr.superlink.servicer.control.conversation_title.requests.post",
            return_value=response,
        ) as post,
        patch("flwr.superlink.servicer.control.conversation_title.Thread") as thread,
    ):
        start_title_generation(state, 33, "Prompt")
        thread.call_args.kwargs["target"](*thread.call_args.kwargs["args"])

    assert post.call_args.kwargs["json"]["input"] == "Prompt"
    state.set_run_series_description.assert_called_once_with(33, "Model title")
    assert thread.call_args.kwargs["daemon"] is True
