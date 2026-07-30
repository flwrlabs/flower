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
"""Tests for the CLI `chat` command."""

import importlib
from unittest.mock import Mock, patch

import click
import pytest

from flwr.cli.constant import (
    CHAT_AGENT_INPUT_KEY,
    CHAT_FLOWER_AGENT_APP_SPEC,
    CHAT_SUPERGRID_CONNECTION_NAME,
    CHAT_USER_PROMPT,
)
from flwr.cli.typing import SuperLinkConnection
from flwr.common.serde import user_config_from_proto
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    StartRunResponse,
    StreamRunEventsResponse,
)
from flwr.proto.task_pb2 import TaskEvent  # pylint: disable=E0611

chat_module = importlib.import_module("flwr.cli.chat")


def test_chat_requires_login_before_prompt() -> None:
    """Chat should fail before prompting if the user has not logged in."""
    superlink_connection = SuperLinkConnection(
        name=CHAT_SUPERGRID_CONNECTION_NAME,
        address="supergrid.flower.ai",
    )
    channel = Mock()
    stub = Mock()
    stub.ListFederations.side_effect = click.ClickException(
        "Missing authentication tokens. Please login first."
    )

    with (
        patch.object(
            chat_module,
            "read_superlink_connection",
            return_value=superlink_connection,
        ),
        patch.object(
            chat_module,
            "init_channel_from_connection",
            return_value=channel,
        ),
        patch.object(chat_module, "ControlStub", return_value=stub),
        patch("builtins.input") as mock_input,
    ):
        with pytest.raises(click.ClickException, match="login first"):
            chat_module.chat()

    mock_input.assert_not_called()
    channel.close.assert_called_once()


def test_chat_submits_prompt_to_flower_agent_and_streams_response(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Chat should submit prompts as agent.input and print streamed deltas."""
    superlink_connection = SuperLinkConnection(
        name=CHAT_SUPERGRID_CONNECTION_NAME,
        address="supergrid.flower.ai",
    )
    channel = Mock()
    stub = Mock()
    stub.ListFederations.return_value = Mock()
    stub.StartRun.return_value = StartRunResponse(run_id=123)
    stub.StreamRunEvents.return_value = iter(
        [
            StreamRunEventsResponse(
                task_event=TaskEvent(
                    event="response.output_text.delta",
                    data='{"type":"response.output_text.delta","delta":"Hel"}',
                )
            ),
            StreamRunEventsResponse(
                task_event=TaskEvent(
                    event="response.output_text.delta",
                    data='{"type":"response.output_text.delta","delta":"lo"}',
                )
            ),
            StreamRunEventsResponse(
                task_event=TaskEvent(
                    event="response.completed",
                    data='{"type":"response.completed"}',
                )
            ),
        ]
    )

    with (
        patch.object(
            chat_module,
            "read_superlink_connection",
            return_value=superlink_connection,
        ),
        patch.object(
            chat_module,
            "init_channel_from_connection",
            return_value=channel,
        ),
        patch.object(chat_module, "ControlStub", return_value=stub),
        patch.object(chat_module.sys.stdin, "isatty", return_value=False),
        patch.object(chat_module.sys.stdout, "isatty", return_value=False),
        patch("builtins.input", side_effect=["Hello", "/quit"]) as mock_input,
    ):
        chat_module.chat()

    start_run_request = stub.StartRun.call_args.args[0]
    stub.ListFederations.assert_called_once()
    assert start_run_request.app_spec == CHAT_FLOWER_AGENT_APP_SPEC
    assert user_config_from_proto(start_run_request.override_config) == {
        CHAT_AGENT_INPUT_KEY: "Hello"
    }
    assert mock_input.call_args_list[0].args[0] == CHAT_USER_PROMPT
    assert "Hello\n" in click.unstyle(capsys.readouterr().out)
    channel.close.assert_called_once()
