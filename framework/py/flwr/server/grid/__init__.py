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
"""Deprecated ``flwr.server.grid`` compatibility APIs."""


from flwr.common.logger import warn_deprecated_feature
from flwr.serverapp.grid import Driver as Driver
from flwr.serverapp.grid import Grid as Grid
from flwr.serverapp.grid import GrpcGrid as GrpcGrid
from flwr.serverapp.grid import InMemoryGrid as InMemoryGrid

_WARNING_LOGGED = False


def _warn_deprecated_import() -> None:
    """Warn about the deprecated grid import path."""
    global _WARNING_LOGGED  # pylint: disable=global-statement

    if not _WARNING_LOGGED:
        _WARNING_LOGGED = True
        warn_deprecated_feature(
            "`flwr.server.grid` has moved to `flwr.serverapp.grid`. "
            "Import from `flwr.serverapp.grid` instead."
        )


_warn_deprecated_import()

__all__ = [
    "Driver",
    "Grid",
    "GrpcGrid",
    "InMemoryGrid",
]
