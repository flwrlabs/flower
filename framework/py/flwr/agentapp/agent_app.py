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

from flwr.app import Context

from .base import AgentAppCallable, AgentSession


class AgentApp:
    """Flower AgentApp.

    Examples
    --------
    Define an AgentApp with a single main function::

        app = AgentApp()

        @app.main()
        def main(agent: AgentSession, context: Context) -> None:
            print("AgentApp running")
    """

    def __init__(self) -> None:
        self._main: AgentAppCallable | None = None

    def __call__(self, agent: AgentSession, context: Context) -> None:
        """Execute `AgentApp`."""
        if self._main is None:
            raise ValueError("AgentApp has no main function.")
        self._main(agent, context)

    def main(self) -> Callable[[AgentAppCallable], AgentAppCallable]:
        """Return a decorator that registers the main fn with the agent app.

        Examples
        --------
        ::

            app = AgentApp()

            @app.main()
            def main(agent: AgentSession, context: Context) -> None:
                print("AgentApp running")
        """

        def main_decorator(main_fn: AgentAppCallable) -> AgentAppCallable:
            """Register the main fn with the AgentApp object."""
            if self._main is not None:
                raise ValueError("AgentApp main function is already registered.")

            # Register provided function with the AgentApp object
            self._main = main_fn

            # Return provided function unmodified
            return main_fn

        return main_decorator
