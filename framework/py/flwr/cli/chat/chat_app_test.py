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
"""Tests for federation selection in the CLI `chat` application."""

from unittest.mock import Mock, patch

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from flwr.cli.chat.chat_app import ChatApplication, _ChatCompleter
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611


def test_chat_selects_federation_from_dropdown() -> None:
    """The federation command should offer and apply federation selections."""
    federations = [
        Federation(name="@flower/flower-agent-execution", description="Default"),
        Federation(name="@flower/other", description="Other"),
    ]
    completer = _ChatCompleter(Mock(), None, federations)
    completions = list(
        completer.get_completions(Document("/federation @flower/o"), CompleteEvent())
    )
    assert [completion.text for completion in completions] == ["@flower/other"]

    application = Mock()
    with patch.object(ChatApplication, "_create_application", return_value=application):
        chat = ChatApplication(Mock(), federations, Mock())
    chat.input_buffer = Mock()
    event = Mock(app=application)

    assert chat._handle_command(  # pylint: disable=protected-access
        event, "/federation"
    )
    assert chat.input_buffer.text == "/federation "
    chat.input_buffer.start_completion.assert_called_once_with(select_first=False)

    chat.transcript = [("", "Previous conversation\n\n")]
    chat.series_id = 123
    assert chat._handle_command(  # pylint: disable=protected-access
        event, "/federation @flower/other"
    )
    assert chat.federation == "@flower/other"
    assert chat.completer.federation == "@flower/other"
    assert chat.series_id is None
    assert not chat.transcript
