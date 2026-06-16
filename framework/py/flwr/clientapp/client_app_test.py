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
"""Tests for Flower ClientApp."""


from collections.abc import Iterator
from itertools import product
from unittest.mock import Mock

import pytest

from flwr.app.message import Context, Message

from .client_app import ClientApp
from .typing import ClientAppCallable


def test_lifespan_success() -> None:
    """Test the lifespan decorator with success."""
    # Prepare
    app = ClientApp()
    enter_code = Mock()
    exit_code = Mock()

    @app.lifespan()
    def test_fn(_: Context) -> Iterator[None]:
        enter_code()
        yield
        exit_code()

    # Execute
    with app._lifespan(Mock(spec=Context)):  # pylint: disable=W0212
        pass

    # Assert
    enter_code.assert_called_once()
    exit_code.assert_called_once()


def test_lifespan_failure() -> None:
    """Test the lifespan decorator with failure."""
    # Prepare
    app = ClientApp()
    enter_code = Mock()
    exit_code = Mock()

    @app.lifespan()
    def test_fn(_: Context) -> Iterator[None]:
        enter_code()
        yield
        exit_code()

    # Execute
    try:
        with app._lifespan(Mock(spec=Context)):  # pylint: disable=W0212
            raise RuntimeError("Test exception")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")

    # Assert
    enter_code.assert_called_once()
    exit_code.assert_called_once()


def test_lifespan_no_yield() -> None:
    """Test the lifespan decorator with no yield."""
    # Prepare
    app = ClientApp()
    enter_code = Mock()

    @app.lifespan()
    def test_fn(_: Context) -> Iterator[None]:  # type: ignore
        enter_code()

    # Execute
    try:
        with app._lifespan(Mock(spec=Context)):  # pylint: disable=W0212
            pass
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")

    # Assert
    enter_code.assert_called_once()


def test_lifespan_multiple_yields() -> None:
    """Test the lifespan decorator with multiple yields."""
    # Prepare
    app = ClientApp()
    enter_code = Mock()
    middle_code = Mock()
    exit_code = Mock()

    @app.lifespan()
    def test_fn(_: Context) -> Iterator[None]:
        enter_code()
        yield
        middle_code()
        yield
        exit_code()

    # Execute
    try:
        with app._lifespan(Mock(spec=Context)):  # pylint: disable=W0212
            pass
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")

    # Assert
    enter_code.assert_called_once()
    middle_code.assert_called_once()
    exit_code.assert_not_called()


@pytest.mark.parametrize("category", ["train", "evaluate", "query"])
def test_register_func_with_default(category: str) -> None:
    """Test the train/evaluate/query decorators with no args."""
    # Prepare
    app = ClientApp()
    input_message = Mock(metadata=Mock(message_type=category))
    output_message = Mock()
    context = Mock()
    func_code = Mock()
    decorator = getattr(app, category)

    @decorator()  # type: ignore
    def func(_msg: Message, _cxt: Context) -> Message:
        assert _msg is input_message and _cxt is context
        func_code()
        return output_message

    # Execute
    actual_ret = app(input_message, context)

    # Assert
    func_code.assert_called_once()
    assert actual_ret is output_message


@pytest.mark.parametrize("category", ["train", "evaluate", "query"])
def test_register_func_with_mods(category: str) -> None:
    """Test the train/evaluate/query decorators with mods."""
    # Prepare
    app = ClientApp()
    input_message = Mock(metadata=Mock(message_type=category))
    output_message = Mock()
    context = Mock()
    trace: list[str] = []
    decorator = getattr(app, category)

    def mock_mod(_msg: Message, _cxt: Context, call_next: ClientAppCallable) -> Message:
        assert _msg is input_message and _cxt is context
        trace.append("mod_code_before")
        ret = call_next(_msg, _cxt)
        trace.append("mod_code_after")
        return ret

    @decorator(mods=[mock_mod])  # type: ignore
    def func(_msg: Message, _cxt: Context) -> Message:
        assert _msg is input_message and _cxt is context
        trace.append("func_code")
        return output_message

    # Execute
    actual_ret = app(input_message, context)

    # Assert
    assert trace == ["mod_code_before", "func_code", "mod_code_after"]
    assert actual_ret is output_message


@pytest.mark.parametrize("category", ["train", "evaluate", "query"])
def test_register_func_with_custom_action(category: str) -> None:
    """Test the train/evaluate/query decorators with custom action."""
    # Prepare
    app = ClientApp()
    input_message = Mock(metadata=Mock(message_type=f"{category}.custom_action"))
    output_message = Mock()
    context = Mock()
    func_code = Mock()
    decorator = getattr(app, category)

    @decorator()  # type: ignore
    def func1(_msg: Message, _cxt: Context) -> Message:
        raise AssertionError("This function should not be called")

    @decorator("wrong_custom_action")  # type: ignore
    def func2(_msg: Message, _cxt: Context) -> Message:
        raise AssertionError("This function should not be called")

    @decorator("custom_action")  # type: ignore
    def func3(_msg: Message, _cxt: Context) -> Message:
        assert _msg is input_message and _cxt is context
        func_code()
        return output_message

    # Execute
    actual_ret = app(input_message, context)

    # Assert
    func_code.assert_called_once()
    assert actual_ret is output_message


@pytest.mark.parametrize(
    "category, action",
    product(["train", "evaluate", "query"], ["nest.nest", "no-hyphen", "", "123"]),
)
def test_register_func_with_wrong_action_name(category: str, action: str) -> None:
    """Test the train/evaluate/query decorators with wrong action name."""
    # Prepare
    app = ClientApp()
    decorator = getattr(app, category)

    # Execute and assert
    with pytest.raises(ValueError):

        @decorator(action)  # type: ignore
        def func(_msg: Message, _cxt: Context) -> Message:
            raise AssertionError("This function should not be called")


@pytest.mark.parametrize(
    "category, action",
    product(["train", "evaluate", "query"], [None, "dummy_action", "default"]),
)
def test_register_repeated_func(category: str, action: str | None) -> None:
    """Test the train/evaluate/query decorators with repeated functions."""
    # Prepare
    app = ClientApp()
    args = (action,) if action is not None else ()
    decorator = getattr(app, category)

    @decorator(*args)  # type: ignore
    def func1(_msg: Message, _cxt: Context) -> Message:
        raise AssertionError("This function should not be called")

    # Execute and assert
    with pytest.raises(ValueError):

        @decorator(*args)  # type: ignore
        def func2(_msg: Message, _cxt: Context) -> Message:
            raise AssertionError("This function should not be called")
