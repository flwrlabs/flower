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

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from flwr.supercore.error import ApiErrorCode
from flwr.supercore.license_plugin import LicensePlugin
from flwr.supercore.protobuf.translation import ProtobufTranslationMiddleware
from flwr.superlink import main as superlink_main

from . import middlewares


def _create_app(
    monkeypatch: MonkeyPatch, license_plugin: LicensePlugin | None
) -> tuple[FastAPI, TestClient]:
    """Create an app containing the complete Control API middleware stack."""
    monkeypatch.setattr(middlewares, "get_license_plugin", lambda: license_plugin)
    app = superlink_main.create_app()

    @app.get("/control/get-login-details")
    def control_route() -> dict[str, bool]:
        """Return a successful Control response."""
        return {"ok": True}

    @app.get("/unlicensed")
    def unlicensed_route() -> dict[str, bool]:
        """Return a successful response outside the Control API."""
        return {"ok": True}

    return app, TestClient(app)


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
    """Run authentication, license checking, and protobuf translation in order."""
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
    )
