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
"""Flower client."""


from importlib import import_module
from typing import TYPE_CHECKING, Any

from flwr.compat.client.client import Client
from flwr.compat.client.numpy_client import NumPyClient
from flwr.compat.client.typing import ClientFn, ClientFnExt

from . import mod

if TYPE_CHECKING:
    from flwr.clientapp.client_app import ClientApp
    from flwr.compat.client.app import start_client, start_numpy_client

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ClientApp": ("flwr.clientapp.client_app", "ClientApp"),
    # Deprecated
    "start_client": ("flwr.compat.client.app", "start_client"),
    # Deprecated
    "start_numpy_client": ("flwr.compat.client.app", "start_numpy_client"),
}

__all__ = [
    "Client",
    "ClientApp",
    "ClientFn",
    "ClientFnExt",
    "NumPyClient",
    "mod",
    "start_client",
    "start_numpy_client",
]


def __getattr__(name: str) -> Any:
    """Get compatibility re-exports lazily."""
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
