# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Public Flower ServerApp APIs."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

from flwr.supercore.privacy_accounting import (
    GaussianPrivacyEvent as GaussianPrivacyEvent,
)
from flwr.supercore.privacy_accounting import NeighboringRelation as NeighboringRelation
from flwr.supercore.privacy_accounting import PrivacyAccountant as PrivacyAccountant
from flwr.supercore.privacy_accounting import PrivacyConfig as PrivacyConfig
from flwr.supercore.privacy_accounting import PrivacySpent as PrivacySpent
from flwr.supercore.privacy_accounting import SamplingMethod as SamplingMethod

from . import strategy
from .grid import Grid
from .server_app import ServerApp as ServerApp

if TYPE_CHECKING:
    from flwr.supercore.privacy_accounting.rdp_accountant import (
        RdpAccountant as RdpAccountant,
    )

_LAZY_EXPORTS = {
    "RdpAccountant": (
        "flwr.supercore.privacy_accounting.rdp_accountant",
        "RdpAccountant",
    ),
}

__all__ = [
    "GaussianPrivacyEvent",
    "Grid",
    "NeighboringRelation",
    "PrivacyAccountant",
    "PrivacyConfig",
    "PrivacySpent",
    "RdpAccountant",
    "SamplingMethod",
    "ServerApp",
    "strategy",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve privacy-accounting backend exports."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
