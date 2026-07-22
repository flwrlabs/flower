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
"""Tests for the Control API middlewares."""


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
from flwr.supercore.error import ApiErrorCode
from flwr.supercore.event_log.typing import LogEntry
from flwr.supercore.license_plugin import LicensePlugin
from flwr.supercore.protobuf.constants import PROTOBUF_MEDIA_TYPE
from flwr.supercore.protobuf.translation import ProtobufTranslationMiddleware
from flwr.superlink import main as superlink_main
from flwr.superlink.servicer.control import control_handlers

from . import middlewares


def _create_app(
    monkeypatch: MonkeyPatch,
    license_plugin: LicensePlugin | None,
    event_log_plugin: EventLogWriterPlugin | None = None,
) -> tuple[FastAPI, TestClient]:
    """Create an app containing the complete Control API middleware stack."""
    monkeypatch.delenv("FLWR_ENABLE_EVENT_LOG", raising=False)
    monkeypatch.setattr(middlewares, "get_license_plugin", lambda: license_plugin)
    app = superlink_main.create_app()
    app.state.control_event_log_plugin = event_log_plugin

    @app.get("/control/get-login-details")
    def control_route() -> dict[str, bool]:
        """Return a successful Control response."""
        return {"ok": True}

    @app.get("/unlicensed")
    def unlicensed_route() -> dict[str, bool]:
        """Return a successful response outside the Control API."""
        return {"ok": True}

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
            "/control/get-login-details",
            content=GetLoginDetailsRequest().SerializeToString(),
            headers={"content-type": PROTOBUF_MEDIA_TYPE},
        ),
    )


def test_license_middleware_passes_through_without_ee_plugin(
    monkeypatch: MonkeyPatch,
) -> None:
    """Control requests pass through when the EE plugin is absent."""
    app, client = _create_app(monkeypatch, None)

    assert middlewares.ControlLicenseMiddleware.__name__ in {
        cast(type[object], middleware.cls).__name__
        for middleware in app.user_middleware
    }
    assert client.get("/control/get-login-details").status_code == 200


def test_license_middleware_allows_valid_license(monkeypatch: MonkeyPatch) -> None:
    """Control requests continue when the EE license is valid."""
    license_plugin = Mock(spec=LicensePlugin)
    license_plugin.check_license.return_value = True
    _, client = _create_app(monkeypatch, license_plugin)

    response = client.get("/control/get-login-details")

    assert response.status_code == 200
    license_plugin.check_license.assert_called_once_with()


def test_license_middleware_rejects_invalid_license(
    monkeypatch: MonkeyPatch,
) -> None:
    """Control requests return permission denied when the EE license is invalid."""
    license_plugin = Mock(spec=LicensePlugin)
    license_plugin.check_license.return_value = False
    _, client = _create_app(monkeypatch, license_plugin)

    response = client.get("/control/get-login-details")

    assert response.status_code == 403
    assert response.json() == {
        "code": ApiErrorCode.LICENSE_CHECK_FAILED,
        "public_message": "License check failed. Please contact the SuperLink "
        "administrator.",
        "public_details": None,
    }
    license_plugin.check_license.assert_called_once_with()


def test_license_middleware_skips_non_control_routes(
    monkeypatch: MonkeyPatch,
) -> None:
    """Routes outside the Control API do not trigger the license check."""
    license_plugin = Mock(spec=LicensePlugin)
    _, client = _create_app(monkeypatch, license_plugin)

    assert client.get("/unlicensed").status_code == 200
    license_plugin.check_license.assert_not_called()


def test_license_middleware_order(monkeypatch: MonkeyPatch) -> None:
    """Run the Control API middleware in interceptor-equivalent order."""
    app, _ = _create_app(monkeypatch, Mock(spec=LicensePlugin))
    middleware_class_names = [
        cast(type[object], middleware.cls).__name__
        for middleware in app.user_middleware
    ]

    assert (
        middleware_class_names.index(
            middlewares.ControlAuthenticationMiddleware.__name__
        )
        < middleware_class_names.index(middlewares.ControlLicenseMiddleware.__name__)
        < middleware_class_names.index(ProtobufTranslationMiddleware.__name__)
        < middleware_class_names.index(middlewares.ControlEventLogMiddleware.__name__)
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


def test_event_log_middleware_writes_before_and_after_events(
    monkeypatch: MonkeyPatch,
) -> None:
    """Write an event before and after a successful unary Control call."""
    event_log_plugin = _create_event_log_plugin()
    expected_response = GetLoginDetailsResponse(authn_type="noop")
    monkeypatch.setattr(
        control_handlers,
        "get_login_details",
        lambda _request, _plugin: expected_response,
    )
    _, client = _create_app(
        monkeypatch, None, cast(EventLogWriterPlugin, event_log_plugin)
    )

    response = _post_get_login_details(client)

    assert response.status_code == 200
    before_kwargs = event_log_plugin.compose_log_before_event.call_args.kwargs
    assert before_kwargs["request"] == GetLoginDetailsRequest()
    assert isinstance(before_kwargs["context"], Request)
    assert before_kwargs["account_info"] is None
    assert before_kwargs["method_name"] == "/control/get-login-details"
    after_kwargs = event_log_plugin.compose_log_after_event.call_args.kwargs
    assert after_kwargs["response"] == expected_response
    assert event_log_plugin.write_log.call_count == 2


def test_event_log_middleware_writes_handler_failure(
    monkeypatch: MonkeyPatch,
) -> None:
    """Write the handler exception as the after-event response."""
    event_log_plugin = _create_event_log_plugin()

    def fail(_: Message, __: object) -> GetLoginDetailsResponse:
        raise RuntimeError("handler failed")

    monkeypatch.setattr(control_handlers, "get_login_details", fail)
    _, client = _create_app(
        monkeypatch, None, cast(EventLogWriterPlugin, event_log_plugin)
    )

    response = _post_get_login_details(client)

    assert response.status_code == 500
    after_result = event_log_plugin.compose_log_after_event.call_args.kwargs["response"]
    assert isinstance(after_result, RuntimeError)
    assert str(after_result) == "handler failed"
    assert event_log_plugin.write_log.call_count == 2
