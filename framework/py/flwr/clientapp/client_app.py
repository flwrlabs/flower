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
"""Flower ClientApp."""


import inspect
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from flwr.app.message_type import MessageType
from flwr.app.metadata import validate_message_type
from flwr.client.client import Client
from flwr.client.message_handler.message_handler import (
    handle_legacy_message_from_msgtype,
)
from flwr.client.mod.utils import make_ffn
from flwr.client.typing import ClientFnExt, Mod
from flwr.common import Context, Message
from flwr.common.logger import warn_deprecated_feature

from .typing import ClientAppCallable

DEFAULT_ACTION = "default"


def _alert_erroneous_client_fn() -> None:
    raise ValueError(
        "A `ClientApp` cannot make use of a `client_fn` that does "
        "not have a signature in the form: `def client_fn(context: "
        "Context)`. You can import the `Context` like this: "
        "`from flwr.common import Context`"
    )


def _inspect_maybe_adapt_client_fn_signature(client_fn: ClientFnExt) -> ClientFnExt:
    client_fn_args = inspect.signature(client_fn).parameters

    if len(client_fn_args) != 1:
        _alert_erroneous_client_fn()

    first_arg = list(client_fn_args.keys())[0]
    first_arg_type = client_fn_args[first_arg].annotation

    if first_arg_type is str or first_arg == "cid":
        # Warn previous signature for `client_fn` seems to be used
        warn_deprecated_feature(
            "`client_fn` now expects a signature `def client_fn(context: Context)`."
            "The provided `client_fn` has signature: "
            f"{dict(client_fn_args.items())}. You can import the `Context` like this:"
            " `from flwr.common import Context`"
        )

        # Wrap depcreated client_fn inside a function with the expected signature
        def adaptor_fn(
            context: Context,
        ) -> Client:  # pylint: disable=unused-argument
            # if patition-id is defined, pass it. Else pass node_id that should
            # always be defined during Context init.
            cid = context.node_config.get("partition-id", context.node_id)
            return client_fn(str(cid))  # type: ignore

        return adaptor_fn

    return client_fn


@contextmanager
def _empty_lifespan(_: Context) -> Iterator[None]:
    yield


class ClientAppException(Exception):
    """Exception raised when an exception is raised while executing a ClientApp."""

    def __init__(self, message: str):
        ex_name = self.__class__.__name__
        self.message = f"\nException {ex_name} occurred. Message: " + message
        super().__init__(self.message)


class ClientApp:
    """Flower ClientApp.

    Examples
    --------
    Assuming a typical ``Client`` implementation named ``FlowerClient``, you can wrap
    it in a ``ClientApp`` as follows::

        class FlowerClient(NumPyClient):
            # ...

        def client_fn(context: Context):
           return FlowerClient().to_client()

        app = ClientApp(client_fn)
    """

    def __init__(
        self,
        client_fn: ClientFnExt | None = None,  # Only for backward compatibility
        mods: list[Mod] | None = None,
    ) -> None:
        self._mods: list[Mod] = mods if mods is not None else []
        self._registered_funcs: dict[str, ClientAppCallable] = {}

        # Create wrapper function for `handle`
        self._call: ClientAppCallable | None = None
        if client_fn is not None:

            client_fn = _inspect_maybe_adapt_client_fn_signature(client_fn)

            def ffn(
                message: Message,
                context: Context,
            ) -> Message:  # pylint: disable=invalid-name
                out_message = handle_legacy_message_from_msgtype(
                    client_fn=client_fn, message=message, context=context
                )
                return out_message

            # Wrap mods around the wrapped handle function
            self._call = make_ffn(ffn, mods if mods is not None else [])

        # Lifespan function
        self._lifespan = _empty_lifespan

    def __call__(self, message: Message, context: Context) -> Message:
        """Execute `ClientApp`."""
        with self._lifespan(context):
            # Execute message using `client_fn`
            if self._call:
                return self._call(message, context)

            # Get the category and the action
            # A valid message type is of the form "<category>" or "<category>.<action>",
            # where <category> must be "train"/"evaluate"/"query", and <action> is a
            # valid Python identifier
            if not validate_message_type(message.metadata.message_type):
                raise ValueError(
                    f"Invalid message type: {message.metadata.message_type}"
                )

            category, action = message.metadata.message_type, DEFAULT_ACTION
            if "." in category:
                category, action = category.split(".")

            # Check if the function is registered
            if (full_name := f"{category}.{action}") in self._registered_funcs:
                return self._registered_funcs[full_name](message, context)

            raise ValueError(f"No {category} function registered with name '{action}'")

    def train(
        self, action: str = DEFAULT_ACTION, *, mods: list[Mod] | None = None
    ) -> Callable[[ClientAppCallable], ClientAppCallable]:
        """Register a train function with the ``ClientApp``.

        Parameters
        ----------
        action : str (default: "default")
            The action name used to route messages. Defaults to "default".
        mods : Optional[list[Mod]] (default: None)
            A list of function-specific modifiers.

        Returns
        -------
        Callable[[ClientAppCallable], ClientAppCallable]
            A decorator that registers a train function with the ``ClientApp``.

        Examples
        --------
        Registering a train function::

            app = ClientApp()

            @app.train()
            def train(message: Message, context: Context) -> Message:
                print("Executing default train function")
                # Create and return an echo reply message
                return Message(message.content, reply_to=message)

        Registering a train function with a custom action name::

            app = ClientApp()

            # Messages with `message_type="train.custom_action"` will be
            # routed to this function.
            @app.train("custom_action")
            def custom_action(message: Message, context: Context) -> Message:
                print("Executing train function for custom action")
                return Message(message.content, reply_to=message)

        Registering a train function with a function-specific Flower Mod::

            from flwr.client.mod import message_size_mod

            app = ClientApp()

            # Using the `mods` argument to apply a function-specific mod.
            @app.train(mods=[message_size_mod])
            def train(message: Message, context: Context) -> Message:
                print("Executing train function with message size mod")
                # Create and return an echo reply message
                return Message(message.content, reply_to=message)
        """
        return _get_decorator(self, MessageType.TRAIN, action, mods)

    def evaluate(
        self, action: str = DEFAULT_ACTION, *, mods: list[Mod] | None = None
    ) -> Callable[[ClientAppCallable], ClientAppCallable]:
        """Register an evaluate function with the ``ClientApp``.

        Parameters
        ----------
        action : str (default: "default")
            The action name used to route messages. Defaults to "default".
        mods : Optional[list[Mod]] (default: None)
            A list of function-specific modifiers.

        Returns
        -------
        Callable[[ClientAppCallable], ClientAppCallable]
            A decorator that registers an evaluate function with the ``ClientApp``.

        Examples
        --------
        Registering an evaluate function::

            app = ClientApp()

            @app.evaluate()
            def evaluate(message: Message, context: Context) -> Message:
                print("Executing default evaluate function")
                # Create and return an echo reply message
                return Message(message.content, reply_to=message)

        Registering an evaluate function with a custom action name::

            app = ClientApp()

            # Messages with `message_type="evaluate.custom_action"` will be
            # routed to this function.
            @app.evaluate("custom_action")
            def custom_action(message: Message, context: Context) -> Message:
                print("Executing evaluate function for custom action")
                return Message(message.content, reply_to=message)

        Registering an evaluate function with a function-specific Flower Mod::

            from flwr.client.mod import message_size_mod

            app = ClientApp()

            # Using the `mods` argument to apply a function-specific mod.
            @app.evaluate(mods=[message_size_mod])
            def evaluate(message: Message, context: Context) -> Message:
                print("Executing evaluate function with message size mod")
                # Create and return an echo reply message
                return Message(message.content, reply_to=message)
        """
        return _get_decorator(self, MessageType.EVALUATE, action, mods)

    def query(
        self, action: str = DEFAULT_ACTION, *, mods: list[Mod] | None = None
    ) -> Callable[[ClientAppCallable], ClientAppCallable]:
        """Register a query function with the ``ClientApp``.

        Parameters
        ----------
        action : str (default: "default")
            The action name used to route messages. Defaults to "default".
        mods : Optional[list[Mod]] (default: None)
            A list of function-specific modifiers.

        Returns
        -------
        Callable[[ClientAppCallable], ClientAppCallable]
            A decorator that registers a query function with the ``ClientApp``.

        Examples
        --------
        Registering a query function::

            app = ClientApp()

            @app.query()
            def query(message: Message, context: Context) -> Message:
                print("Executing default query function")
                # Create and return an echo reply message
                return Message(message.content, reply_to=message)

        Registering a query function with a custom action name::

            app = ClientApp()

            # Messages with `message_type="query.custom_action"` will be
            # routed to this function.
            @app.query("custom_action")
            def custom_action(message: Message, context: Context) -> Message:
                print("Executing query function for custom action")
                return Message(message.content, reply_to=message)

        Registering a query function with a function-specific Flower Mod::

            from flwr.client.mod import message_size_mod

            app = ClientApp()

            # Using the `mods` argument to apply a function-specific mod.
            @app.query(mods=[message_size_mod])
            def query(message: Message, context: Context) -> Message:
                print("Executing query function with message size mod")
                # Create and return an echo reply message
                return Message(message.content, reply_to=message)
        """
        return _get_decorator(self, MessageType.QUERY, action, mods)

    def lifespan(
        self,
    ) -> Callable[
        [Callable[[Context], Iterator[None]]], Callable[[Context], Iterator[None]]
    ]:
        """Return a decorator that registers the lifespan fn with the client app.

        The decorated function should accept a `Context` object and use `yield`
        to define enter and exit behavior.

        Examples
        --------
        ::

                app = ClientApp()

                @app.lifespan()
                def lifespan(context: Context) -> None:
                    # Perform initialization tasks before the app starts
                    print("Initializing ClientApp")

                    yield  # ClientApp is running

                    # Perform cleanup tasks after the app stops
                    print("Cleaning up ClientApp")
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

            # Register provided function with the ClientApp object
            # Ignore mypy error because of different argument names (`_` vs `context`)
            self._lifespan = decorated_lifespan  # type: ignore

            # Return provided function unmodified
            return lifespan_fn

        return lifespan_decorator


class LoadClientAppError(Exception):
    """Error when trying to load `ClientApp`."""


def _get_decorator(
    app: ClientApp, category: str, action: str, mods: list[Mod] | None
) -> Callable[[ClientAppCallable], ClientAppCallable]:
    """Get the decorator for the given category and action."""
    # pylint: disable=protected-access
    if app._call:
        raise _registration_error(category)

    def decorator(fn: ClientAppCallable) -> ClientAppCallable:

        # Check if the name is a valid Python identifier
        if not action.isidentifier():
            raise ValueError(
                f"Cannot register {category} function with name '{action}'. "
                "The name must follow Python's function naming rules."
            )

        # Check if the name is already registered
        full_name = f"{category}.{action}"  # Full name of the message type
        if full_name in app._registered_funcs:
            raise ValueError(
                f"Cannot register {category} function with name '{action}'. "
                f"A {category} function with the name '{action}' is already registered."
            )

        # Register provided function with the ClientApp object
        app._registered_funcs[full_name] = make_ffn(fn, app._mods + (mods or []))

        # Return provided function unmodified
        return fn

    # pylint: enable=protected-access
    return decorator


def _registration_error(fn_name: str) -> ValueError:
    return ValueError(
        f"""Use either `@app.{fn_name}()` or `client_fn`, but not both.

        Use the `ClientApp` with an existing `client_fn`:

        class FlowerClient(NumPyClient):
            # ...

            def client_fn(context: Context):
                return FlowerClient().to_client()

            app = ClientApp(
                client_fn=client_fn,
            )

        Use the `ClientApp` with a custom {fn_name} function:

            app = ClientApp()

            @app.{fn_name}()
            def {fn_name}(message: Message, context: Context) -> Message:
                print("ClientApp {fn_name} running")
                # Create and return an echo reply message
                return Message(message.content, reply_to=message)
        """,
    )
