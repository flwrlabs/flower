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
"""Tests for executor-bound AgentApp sessions."""

from typing import cast
from unittest.mock import Mock, patch

from flwr.app import Context, RecordDict

from .context_items import append_items
from .session import RuntimeAgentResponses


def test_create_private_does_not_append_response_output() -> None:
    """Private model responses leave shared conversation items unchanged."""
    context = Context(1, 0, {}, RecordDict(), {}, series_id=1)
    append_items(
        context,
        [{"type": "message", "role": "user", "content": "Prompt"}],
    )
    items = cast(list[str], context.state["items"]["json"])
    items_before = list(items)
    responses = RuntimeAgentResponses(stub=Mock(), run_id=1, task_id=2, context=context)
    response = {
        "output": [{"type": "message", "role": "assistant", "content": "Private title"}]
    }

    with patch.object(responses, "_create_model_response", return_value=response):
        assert responses.create_private({"model": "openai/gpt-4o"}) == response

    assert items == items_before
