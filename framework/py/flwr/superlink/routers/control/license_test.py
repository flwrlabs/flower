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
"""Tests for Control API FastAPI license checking."""

from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request
from starlette.datastructures import State

from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.license_plugin import LicensePlugin

from .license import ControlLicenseChecker


def _request(license_plugin: LicensePlugin) -> Request[State]:
    """Return a Control request configured with the supplied license plugin."""
    app = FastAPI()
    app.state.control_license_plugin = license_plugin
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/control/list-runs",
            "headers": [],
            "app": app,
        }
    )


def test_check_allows_a_valid_license() -> None:
    """A valid license allows the request to proceed."""
    license_plugin = Mock(spec=LicensePlugin)
    license_plugin.check_license.return_value = True

    ControlLicenseChecker.check(_request(license_plugin))

    license_plugin.check_license.assert_called_once_with()


def test_check_rejects_an_invalid_license() -> None:
    """An invalid license rejects the request."""
    license_plugin = Mock(spec=LicensePlugin)
    license_plugin.check_license.return_value = False

    with pytest.raises(FlowerError) as error:
        ControlLicenseChecker.check(_request(license_plugin))

    assert error.value.code == ApiErrorCode.NO_PERMISSIONS
    assert error.value.message == (
        "License check failed. Please contact the SuperLink administrator."
    )
    license_plugin.check_license.assert_called_once_with()
