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
"""Tests for prompt history in the CLI `chat` application."""


from unittest.mock import Mock, patch

from flwr.cli.chat_app import ChatApplication


def test_submit_adds_nonempty_prompt_to_history() -> None:
    """Submitted prompts should be available for recall from the input buffer."""
    application = Mock()
    with patch.object(ChatApplication, "_create_application", return_value=application):
        chat = ChatApplication(Mock(), None, Mock())
    event = Mock(app=application)

    chat.input_buffer.text = "hello"
    chat._submit_prompt(event)  # pylint: disable=protected-access

    assert chat.input_buffer.history.get_strings() == ["hello"]
    run_coroutine = application.create_background_task.call_args.args[0]
    run_coroutine.close()


def test_submit_does_not_add_blank_prompt_to_history() -> None:
    """Blank submissions should not appear in prompt history."""
    application = Mock()
    with patch.object(ChatApplication, "_create_application", return_value=application):
        chat = ChatApplication(Mock(), None, Mock())

    chat.input_buffer.text = "   "
    chat._submit_prompt(Mock(app=application))  # pylint: disable=protected-access

    assert chat.input_buffer.history.get_strings() == []
