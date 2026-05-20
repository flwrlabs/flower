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
"""AgentApp exceptions."""


from __future__ import annotations

from .typing import JSONObject


class AgentAppError(Exception):
    """Base class for AgentApp runtime errors."""


class LoadAgentAppError(AgentAppError):
    """Raised when an AgentApp cannot be loaded."""


class AgentModelError(AgentAppError):
    """Raised when model execution returns a structured error."""

    def __init__(self, error: JSONObject) -> None:
        message = error.get("message", "Model task failed.")
        super().__init__(message if isinstance(message, str) else str(message))
        self.error = error


class AgentModelTimeoutError(AgentModelError):
    """Raised when a model response is not available before the timeout."""
