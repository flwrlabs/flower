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
"""AgentApp abstract base classes and callable type alias."""


from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from flwr.app import Context
from flwr.supercore.typing import JSONObject


class AgentResponses(ABC):
    """Abstract base class for AgentApp model response creation."""

    @abstractmethod
    def create(self, request: JSONObject) -> JSONObject:
        """Create a model response.

        Parameters
        ----------
        request : JSONObject
            Open Responses-compatible create request.
            Built-in connector tools can be requested by name in ``tools``, for
            example ``["web_search"]``. Built-in connector names are reserved
            for runtime-managed tools.

        Returns
        -------
        response : JSONObject
            Open Responses-compatible response.
        """


class AgentSession(ABC):
    """Abstract base class for AgentApp runtime capabilities."""

    @property
    @abstractmethod
    def responses(self) -> AgentResponses:
        """Model response creation API."""


AgentAppCallable = Callable[[AgentSession, Context], None]
