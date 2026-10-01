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
"""Flower task process components."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import run_agentapp
    from .connector import run_connector
    from .model import run_model

__all__ = [
    "run_agentapp",
    "run_connector",
    "run_model",
]


def __getattr__(name: str) -> Any:
    """Load only the requested task process stack."""
    value: Any
    if name == "run_agentapp":
        from .agent import run_agentapp  # pylint: disable=import-outside-toplevel

        value = run_agentapp
    elif name == "run_connector":
        from .connector import run_connector  # pylint: disable=import-outside-toplevel

        value = run_connector
    elif name == "run_model":
        from .model import run_model  # pylint: disable=import-outside-toplevel

        value = run_model
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
