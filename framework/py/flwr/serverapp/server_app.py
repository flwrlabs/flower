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
"""Flower ServerApp."""


import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from flwr.app import Context
from flwr.common.logger import warn_deprecated_feature_with_example
from flwr.server.client_manager import ClientManager
from flwr.server.compat import start_grid
from flwr.server.server import Server
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import Strategy
from flwr.server.typing import ServerAppCallable, ServerFn

from .grid import Grid

SERVER_FN_USAGE_EXAMPLE = """

        def server_fn(context: Context):
            server_config = ServerConfig(num_rounds=3)
            strategy = FedAvg()
            return ServerAppComponents(
                strategy=strategy,
                server_config=server_config,
        )

        app = ServerApp(server_fn=server_fn)
"""

GRID_USAGE_EXAMPLE = """
                app = ServerApp()

                @app.main()
                def main(grid: Grid, context: Context) -> None:
                    # Your existing ServerApp code ...
"""

BOTH_MAIN_FN_SERVER_FN_PROVIDED_ERROR_MSG = (
    "Use either a custom main function or a `Strategy`, but not both."
    """

Use the `ServerApp` with an existing `Strategy`:

    server_config = ServerConfig(num_rounds=3)
    strategy = FedAvg()

    app = ServerApp(
        server_config=server_config,
        strategy=strategy,
    )

Use the `ServerApp` with a custom main function:

    app = ServerApp()

    @app.main()
    def main(grid: Grid, context: Context) -> None:
        print("ServerApp running")
"""
)

DRIVER_DEPRECATION_MSG = """
            The `Driver` class is deprecated, it will be removed in a future release.
"""
DRIVER_EXAMPLE_MSG = """
            Instead, use `Grid` in the signature of your `ServerApp`. For example:
"""


@contextmanager
def _empty_lifespan(_: Context) -> Iterator[None]:
    yield


class ServerApp:  # pylint: disable=too-many-instance-attributes
    """Flower ServerApp.

    Examples
    --------
    Use the ``ServerApp`` with an existing ``Strategy``::

        def server_fn(context: Context):
            server_config = ServerConfig(num_rounds=3)
            strategy = FedAvg()
            return ServerAppComponents(
                strategy=strategy,
                server_config=server_config,
            )

        app = ServerApp(server_fn=server_fn)

    Use the ``ServerApp`` with a custom main function::

        app = ServerApp()

        @app.main()
        def main(grid: Grid, context: Context) -> None:
           print("ServerApp running")
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        server: Server | None = None,
        config: ServerConfig | None = None,
        strategy: Strategy | None = None,
        client_manager: ClientManager | None = None,
        server_fn: ServerFn | None = None,
    ) -> None:
        if any([server, config, strategy, client_manager]):
            warn_deprecated_feature_with_example(
                deprecation_message="Passing either `server`, `config`, `strategy` or "
                "`client_manager` directly to the ServerApp "
                "constructor is deprecated.",
                example_message="Pass `ServerApp` arguments wrapped "
                "in a `flwr.server.ServerAppComponents` object that gets "
                "returned by a function passed as the `server_fn` argument "
                "to the `ServerApp` constructor. For example: ",
                code_example=SERVER_FN_USAGE_EXAMPLE,
            )

            if server_fn:
                raise ValueError(
                    "Passing `server_fn` is incompatible with passing the "
                    "other arguments (now deprecated) to ServerApp. "
                    "Use `server_fn` exclusively."
                )

        self._server = server
        self._config = config
        self._strategy = strategy
        self._client_manager = client_manager
        self._server_fn = server_fn
        self._main: ServerAppCallable | None = None
        self._lifespan = _empty_lifespan

    def __call__(self, grid: Grid, context: Context) -> None:
        """Execute `ServerApp`."""
        with self._lifespan(context):
            # Compatibility mode
            if not self._main:
                if self._server_fn:
                    # Execute server_fn()
                    components = self._server_fn(context)
                    self._server = components.server
                    self._config = components.config
                    self._strategy = components.strategy
                    self._client_manager = components.client_manager
                start_grid(
                    server=self._server,
                    config=self._config,
                    strategy=self._strategy,
                    client_manager=self._client_manager,
                    grid=grid,
                )
                return

            # New execution mode
            self._main(grid, context)

    def main(self) -> Callable[[ServerAppCallable], ServerAppCallable]:
        """Return a decorator that registers the main fn with the server app.

        Examples
        --------
        ::

            app = ServerApp()

            @app.main()
            def main(grid: Grid, context: Context) -> None:
                print("ServerApp running")
        """

        def main_decorator(main_fn: ServerAppCallable) -> ServerAppCallable:
            """Register the main fn with the ServerApp object."""
            if self._server or self._config or self._strategy or self._client_manager:
                raise ValueError(BOTH_MAIN_FN_SERVER_FN_PROVIDED_ERROR_MSG)

            sig = inspect.signature(main_fn)
            param = list(sig.parameters.values())[0]
            # Check if parameter name or the annotation should be updated
            # pylint: disable=import-outside-toplevel
            from flwr.compat.server.grid import Driver

            # pylint: enable=import-outside-toplevel
            if param.name == "driver" or param.annotation is Driver:
                warn_deprecated_feature_with_example(
                    deprecation_message=DRIVER_DEPRECATION_MSG,
                    example_message=DRIVER_EXAMPLE_MSG,
                    code_example=GRID_USAGE_EXAMPLE,
                )

            # Register provided function with the ServerApp object
            self._main = main_fn

            # Return provided function unmodified
            return main_fn

        return main_decorator

    def lifespan(
        self,
    ) -> Callable[
        [Callable[[Context], Iterator[None]]], Callable[[Context], Iterator[None]]
    ]:
        """Return a decorator that registers the lifespan fn with the server app.

        The decorated function should accept a `Context` object and use `yield`
        to define enter and exit behavior.

        Examples
        --------
        ::

            app = ServerApp()

            @app.lifespan()
            def lifespan(context: Context) -> None:
                # Perform initialization tasks before the app starts
                print("Initializing ServerApp")

                yield  # ServerApp is running

                # Perform cleanup tasks after the app stops
                print("Cleaning up ServerApp")
        """

        def lifespan_decorator(
            lifespan_fn: Callable[[Context], Iterator[None]],
        ) -> Callable[[Context], Iterator[None]]:
            """Register the lifespan fn with the ServerApp object."""

            @contextmanager
            def decorated_lifespan(context: Context) -> Iterator[None]:
                # Execute the code before `yield` in lifespan_fn
                try:
                    if not isinstance(it := lifespan_fn(context), Iterator):
                        raise StopIteration
                    next(it)
                except StopIteration:
                    raise RuntimeError(
                        "lifespan function should yield at least once."
                    ) from None

                try:
                    # Enter the context
                    yield
                finally:
                    try:
                        # Execute the code after `yield` in lifespan_fn
                        next(it)
                    except StopIteration:
                        pass
                    else:
                        raise RuntimeError("lifespan function should only yield once.")

            # Register provided function with the ServerApp object
            # Ignore mypy error because of different argument names (`_` vs `context`)
            self._lifespan = decorated_lifespan  # type: ignore

            # Return provided function unmodified
            return lifespan_fn

        return lifespan_decorator


class LoadServerAppError(Exception):
    """Error when trying to load `ServerApp`."""
