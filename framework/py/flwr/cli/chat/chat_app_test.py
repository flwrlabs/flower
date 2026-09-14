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
"""Tests for the CLI `chat` application."""

import asyncio
from pathlib import Path
from unittest.mock import Mock, patch

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from flwr.cli.chat.chat_app import ChatApplication, _ChatCompleter, start_chat_run
from flwr.cli.chat.chat_local_agent import LocalAgent
from flwr.cli.constant import CHAT_AGENT_NAME, CHAT_DEFAULT_FEDERATION_NAME
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    Connector,
    ListConnectorsResponse,
    StartRunResponse,
)
from flwr.proto.federation_pb2 import Federation  # pylint: disable=E0611
from flwr.supercore.constant import FLOWER_AGENT_APP_ID

_CHAT_FED_ID = f"@flower/{CHAT_DEFAULT_FEDERATION_NAME}"


def test_chat_selects_federation_from_dropdown() -> None:
    """The federation command should offer and apply federation selections."""
    federations = [
        Federation(name=_CHAT_FED_ID, description="Default"),
        Federation(name="@flower/other", description="Other"),
    ]
    completer = _ChatCompleter(Mock(), Mock(), federations[0].name, federations)
    completions = list(
        completer.get_completions(Document("/federation @flower/o"), CompleteEvent())
    )
    assert [completion.text for completion in completions] == ["@flower/other"]

    application = Mock()
    with patch.object(ChatApplication, "_create_application", return_value=application):
        chat = ChatApplication(Mock(), federations, Mock())
    assert chat.federation == _CHAT_FED_ID
    chat.input_buffer = Mock()
    event = Mock(app=application)

    assert chat._handle_command(  # pylint: disable=protected-access
        event, "/federation"
    )
    assert chat.input_buffer.text == "/federation "
    chat.input_buffer.start_completion.assert_called_once_with(select_first=False)

    chat.transcript = [("", "Previous conversation\n\n")]
    chat.series_id = 123
    chat.agent_app_spec = "@flower/custom-agent"
    chat.agent_fab_hash = "fab-hash"
    chat.agent_name = "Custom Agent"
    assert chat._handle_command(  # pylint: disable=protected-access
        event, "/federation @flower/other"
    )
    assert chat.federation == "@flower/other"
    assert chat.completer.federation == "@flower/other"
    assert chat.agent_app_spec == FLOWER_AGENT_APP_ID
    assert chat.agent_fab_hash is None
    assert chat.agent_name == CHAT_AGENT_NAME
    assert chat.series_id is None
    assert not chat.transcript

    transcript = [("", "Active conversation\n\n")]
    chat.transcript = transcript
    chat.series_id = 456
    application.reset_mock()
    assert chat._handle_command(  # pylint: disable=protected-access
        event, "/federation @flower/other"
    )
    assert chat.series_id == 456
    assert chat.transcript == transcript
    application.invalidate.assert_not_called()


def test_chat_loads_and_rebuilds_local_agent_before_prompt() -> None:
    """A loaded local AgentApp should be rebuilt before each prompt."""
    application = Mock()
    with patch.object(ChatApplication, "_create_application", return_value=application):
        chat = ChatApplication(Mock(), [Federation(name=_CHAT_FED_ID)], Mock())
    event = Mock(app=application)
    first_agent = LocalAgent(
        path=Path(r"C:\Users\me\agent"),
        app_spec="@local/custom-agent",
        fab_hash="first-hash",
        fab_content=b"first-fab",
        warnings=(),
    )
    rebuilt_agent = LocalAgent(
        path=first_agent.path,
        app_spec=first_agent.app_spec,
        fab_hash="second-hash",
        fab_content=b"second-fab",
        warnings=(),
    )

    with patch(
        "flwr.cli.chat.chat_app.build_local_agent", return_value=first_agent
    ) as mock_build:
        assert chat._handle_command(  # pylint: disable=protected-access
            event, r"/load C:\Users\me\agent"
        )
        asyncio.run(event.app.create_background_task.call_args.args[0])
    mock_build.assert_called_once_with(first_agent.path)
    assert chat.local_agent == first_agent

    chat.series_id = 123
    with (
        patch(
            "flwr.cli.chat.chat_app.build_local_agent", return_value=rebuilt_agent
        ) as mock_build,
        patch.object(chat, "_run_prompt_sync") as mock_run,
    ):
        asyncio.run(
            chat._run_prompt(  # pylint: disable=protected-access
                "Hello", first_agent.app_spec, first_agent.fab_hash
            )
        )
    mock_build.assert_called_once_with(first_agent.path)
    assert chat.local_agent == rebuilt_agent
    assert chat.series_id is None
    mock_run.assert_called_once_with(
        "Hello",
        rebuilt_agent.app_spec,
        rebuilt_agent.fab_hash,
        rebuilt_agent.fab_content,
    )


def test_start_chat_run_uploads_local_fab() -> None:
    """Uploading local FAB content should omit the app spec."""
    stub = Mock()
    stub.StartRun.return_value = StartRunResponse(run_id=1, series_id=2)

    assert start_chat_run(
        stub,
        "Hello",
        _CHAT_FED_ID,
        None,
        "@local/custom-agent",
        fab_content=b"fab-content",
    ) == (1, 2)
    request = stub.StartRun.call_args.args[0]
    assert request.app_spec == ""
    assert request.fab.content == b"fab-content"


def test_chat_selects_connector_from_dropdown() -> None:
    """The connector command should offer and apply connected connectors."""
    application = Mock()
    stub = Mock()
    stub.ListConnectors.return_value = ListConnectorsResponse(
        connectors=[
            Connector(
                connector_ref="github",
                display_name="GitHub",
                description="Search GitHub",
                connected=True,
            ),
            Connector(
                connector_ref="attio",
                display_name="Attio",
                description="Search Attio",
                connected=True,
            ),
            Connector(
                connector_ref="slack",
                display_name="Slack",
                description="Search Slack",
                connected=False,
            ),
        ]
    )
    with patch.object(ChatApplication, "_create_application", return_value=application):
        chat = ChatApplication(stub, [Federation(name=_CHAT_FED_ID)], Mock())
    chat.input_buffer = Mock()
    event = Mock(app=application)

    assert chat._handle_command(event, "/connector")  # pylint: disable=W0212
    assert chat.input_buffer.text == "/connector "
    chat.input_buffer.start_completion.assert_called_once_with(select_first=False)
    request = stub.ListConnectors.call_args.args[0]
    assert request.federation == _CHAT_FED_ID

    completions = list(
        chat.completer.get_completions(Document("/connector git"), CompleteEvent())
    )
    assert [completion.text for completion in completions] == ["github"]

    assert chat._handle_command(  # pylint: disable=protected-access
        event, "/connector github"
    )
    assert chat.connector_refs == ["github"]
    assert chat._handle_command(  # pylint: disable=protected-access
        event, "/connector attio"
    )
    assert chat._handle_command(  # pylint: disable=protected-access
        event, "/connector github"
    )
    assert chat.connector_refs == ["github", "attio"]
    assert chat._render_agent_name() == [  # pylint: disable=protected-access
        (
            "class:agent.name",
            f" ✿ {CHAT_AGENT_NAME} · {_CHAT_FED_ID} · connectors: github, attio ",
        )
    ]
    clear_completions = list(
        chat.completer.get_completions(Document("/connector cl"), CompleteEvent())
    )
    assert [completion.text for completion in clear_completions] == ["clear"]
    assert clear_completions[0].display_text == (
        "clear         Clear selected connectors"
    )

    assert chat._handle_command(  # pylint: disable=protected-access
        event, "/connector clear"
    )
    assert not chat.connector_refs
    assert chat._render_agent_name() == [  # pylint: disable=protected-access
        ("class:agent.name", f" ✿ {CHAT_AGENT_NAME} · {_CHAT_FED_ID} ")
    ]
    assert not chat.transcript


def test_chat_connector_command_directs_to_webui_when_empty() -> None:
    """The connector command should refresh after directing setup to WebUI."""
    application = Mock()
    stub = Mock()
    stub.ListConnectors.side_effect = [
        ListConnectorsResponse(),
        ListConnectorsResponse(
            connectors=[
                Connector(
                    connector_ref="github",
                    display_name="GitHub",
                    connected=True,
                )
            ]
        ),
    ]
    with patch.object(ChatApplication, "_create_application", return_value=application):
        chat = ChatApplication(stub, [Federation(name=_CHAT_FED_ID)], Mock())
    chat.input_buffer = Mock()

    event = Mock(app=application)
    assert chat._handle_command(event, "/connector")  # pylint: disable=W0212
    assert chat.transcript[-1] == (
        "class:notice",
        "No connected connectors found. Please configure connectors using WebUI.\n\n",
    )

    assert chat._handle_command(event, "/connector")  # pylint: disable=W0212
    assert stub.ListConnectors.call_count == 2
    assert chat.input_buffer.text == "/connector "
    chat.input_buffer.start_completion.assert_called_once_with(select_first=False)


def test_chat_rejects_connector_selection_outside_personal_federation() -> None:
    """Connectors should only be selectable in the personal federation."""
    application = Mock()
    federations = [
        Federation(name=_CHAT_FED_ID),
        Federation(name="@flower/other"),
    ]
    with patch.object(ChatApplication, "_create_application", return_value=application):
        chat = ChatApplication(Mock(), federations, Mock())
    chat.connector_refs = ["github"]

    assert chat._handle_command(  # pylint: disable=protected-access
        Mock(app=application), "/federation @flower/other"
    )
    assert not chat.connector_refs

    assert chat._handle_command(  # pylint: disable=protected-access
        Mock(app=application), "/connector"
    )
    assert chat.transcript[-1] == (
        "class:notice",
        "Connectors are only available in the personal federation.\n\n",
    )


def test_start_chat_run_includes_selected_connectors() -> None:
    """Selected connectors should be bound to the chat run."""
    stub = Mock()
    stub.StartRun.return_value = StartRunResponse(run_id=1, series_id=2)

    start_chat_run(
        stub,
        "Hello",
        _CHAT_FED_ID,
        None,
        connector_refs=["github", "attio"],
    )

    request = stub.StartRun.call_args.args[0]
    assert list(request.connector_refs) == ["github", "attio"]
