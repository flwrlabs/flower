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
"""Control API license checking for FastAPI requests."""

from typing import cast

from fastapi import Request
from starlette.datastructures import State

from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.license_plugin import LicensePlugin


class ControlLicenseChecker:
    """Check the configured Control API license."""

    @staticmethod
    def check(http_request: Request[State]) -> None:
        """Raise when the configured license is invalid."""
        license_plugin = cast(
            LicensePlugin | None,
            getattr(http_request.app.state, "control_license_plugin", None),
        )
        if license_plugin is not None and not license_plugin.check_license():
            raise FlowerError(
                ApiErrorCode.NO_PERMISSIONS,
                "License check failed. Please contact the SuperLink administrator.",
            )
