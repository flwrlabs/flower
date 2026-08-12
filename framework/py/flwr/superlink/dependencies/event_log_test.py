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
"""Tests for the Control API event-log dependency."""

from typing import cast
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from google.protobuf.message import Message
from httpx import Response as HTTPResponse
from pytest import MonkeyPatch

from flwr.common.event_log_plugin import EventLogWriterPlugin
from flwr.proto.control_pb2 import (  # pylint: disable=E0611
    GetLoginDetailsRequest,
    GetLoginDetailsResponse,
)
from flwr.supercore.event_log.typing import LogEntry
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.superlink import main as superlink_main
from flwr.superlink.routers.control import middlewares
from flwr.superlink.servicer.control import control_handlers


def _create_app(
    monkeypatch: MonkeyPatch,
    event_log_plugin: EventLogWriterPlugin,
) -> tuple[FastAPI, TestClient]:
    """Create an app containing the Control event-log dependency."""
    monkeypatch.delenv("FLWR_ENABLE_EVENT_LOG", raising=False)
    monkeypatch.setattr(middlewares, "get_license_plugin", lambda: None)
    app = superlink_main.create_app()
    app.state.control_event_log_plugin = event_log_plugin
    return app, TestClient(app)


def _create_event_log_plugin() -> Mock:
    """Create a mock event-log plugin returning writable entries."""
    plugin = Mock(spec=EventLogWriterPlugin)
    plugin.compose_log_before_event.return_value = Mock(spec=LogEntry)
    plugin.compose_log_after_event.return_value = Mock(spec=LogEntry)
    return plugin


def _post_get_login_details(client: TestClient) -> HTTPResponse:
    """Send a protobuf request to the unauthenticated Control endpoint."""
    return cast(
        HTTPResponse,
        client.post(
            "/v1/control/get-login-details",
            content=GetLoginDetailsRequest().SerializeToString(),
            headers={"content-type": PROTOBUF_MEDIA_TYPE},
        ),
    )


@pytest.mark.parametrize("env_value", [None, "0"])
def test_create_app_disables_event_log_without_enabled_env_var(
    monkeypatch: MonkeyPatch, env_value: str | None
) -> None:
    """Direct FastAPI startup disables event logging unless explicitly enabled."""
    load_plugin = Mock()
    monkeypatch.setattr(superlink_main, "load_control_event_log_plugin", load_plugin)
    if env_value is None:
        monkeypatch.delenv("FLWR_ENABLE_EVENT_LOG", raising=False)
    else:
        monkeypatch.setenv("FLWR_ENABLE_EVENT_LOG", env_value)

    app = superlink_main.create_app()

    assert app.state.control_event_log_plugin is None
    load_plugin.assert_not_called()


def test_create_app_loads_event_log_with_enabled_env_var(
    monkeypatch: MonkeyPatch,
) -> None:
    """Direct FastAPI startup mirrors the CLI event-log flag when enabled."""
    expected_plugin = _create_event_log_plugin()
    load_plugin = Mock(return_value=expected_plugin)
    monkeypatch.setattr(superlink_main, "load_control_event_log_plugin", load_plugin)
    monkeypatch.setenv("FLWR_ENABLE_EVENT_LOG", "1")

    app = superlink_main.create_app()

    assert app.state.control_event_log_plugin is expected_plugin
    load_plugin.assert_called_once_with()


def test_event_log_dependency_writes_before_and_after_events(
    monkeypatch: MonkeyPatch,
) -> None:
    """Write an event before and after a successful unary Control call."""
    event_log_plugin = _create_event_log_plugin()
    expected_response = GetLoginDetailsResponse(authn_type="noop")
    before_entry = event_log_plugin.compose_log_before_event.return_value
    execution_order: list[str] = []
    event_log_plugin.write_log.side_effect = lambda entry: execution_order.append(
        "before" if entry is before_entry else "after"
    )

    def get_login_details(_: Message, __: object) -> GetLoginDetailsResponse:
        execution_order.append("handler")
        return expected_response

    monkeypatch.setattr(
        control_handlers,
        "get_login_details",
        get_login_details,
    )
    _, client = _create_app(monkeypatch, cast(EventLogWriterPlugin, event_log_plugin))

    response = _post_get_login_details(client)

    assert response.status_code == 200
    before_kwargs = event_log_plugin.compose_log_before_event.call_args.kwargs
    assert before_kwargs["request"] == GetLoginDetailsRequest()
    assert isinstance(before_kwargs["context"], Request)
    assert before_kwargs["account_info"] is None
    assert before_kwargs["method_name"] == "/v1/control/get-login-details"
    after_kwargs = event_log_plugin.compose_log_after_event.call_args.kwargs
    assert after_kwargs["response"] == expected_response
    assert execution_order == ["before", "handler", "after"]
    assert event_log_plugin.write_log.call_count == 2


def test_event_log_dependency_writes_handler_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    """Write the handler exception as the after-event response."""
    event_log_plugin = _create_event_log_plugin()

    def fail(_: Message, __: object) -> GetLoginDetailsResponse:
        raise RuntimeError("handler failed")

    monkeypatch.setattr(control_handlers, "get_login_details", fail)
    _, client = _create_app(monkeypatch, cast(EventLogWriterPlugin, event_log_plugin))

    response = _post_get_login_details(client)

    assert response.status_code == 500
    after_result = event_log_plugin.compose_log_after_event.call_args.kwargs["response"]
    assert isinstance(after_result, RuntimeError)
    assert str(after_result) == "handler failed"
    assert event_log_plugin.write_log.call_count == 2
