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
"""Flower AgentApp."""


from __future__ import annotations

from collections.abc import Callable

from .exceptions import AgentAppError
from .session import AgentSession
from .typing import JSONObject

_AgentAppCallable = Callable[[AgentSession], JSONObject]


class AgentApp:
    """Flower AgentApp.

    Examples
    --------
    Define an AgentApp with a single main function::

        app = AgentApp()

        @app.main()
        def main(agent: AgentSession) -> JSONObject:
            return {"output_text": "Hello"}
    """

    def __init__(self) -> None:
        self._main: _AgentAppCallable | None = None

    def __call__(self, session: AgentSession) -> JSONObject:
        """Execute the registered main function."""
        if self._main is None:
            raise AgentAppError("AgentApp has no main function.")
        result = self._main(session)
        if not isinstance(result, dict):
            raise AgentAppError("AgentApp main function must return a JSON object.")
        return result

    def main(self) -> Callable[[_AgentAppCallable], _AgentAppCallable]:
        """Return a decorator that registers the AgentApp main function."""

        def main_decorator(main_fn: _AgentAppCallable) -> _AgentAppCallable:
            if self._main is not None:
                raise ValueError("AgentApp main function is already registered.")
            self._main = main_fn
            return main_fn

        return main_decorator
