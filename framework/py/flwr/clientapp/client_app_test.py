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

from flwr.app import Context, Error, Message, RecordDict
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Status,
)
from flwr.common.fl_event import (
    FL_NODE_EVALUATE_COMPLETED,
    FL_NODE_EVALUATE_FAILED,
    FL_NODE_EVALUATE_STARTED,
    FL_NODE_FIT_COMPLETED,
    FL_NODE_FIT_FAILED,
    FL_NODE_FIT_STARTED,
)
from flwr.compat.client.client import Client
from flwr.compat.common.recorddict_compat import (
    evaluateins_to_recorddict,
    fitins_to_recorddict,
)
from flwr.supercore.utils import strict_json_loads

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


class _MockClient(Client):
    """A minimal ``Client`` implementation for event tests."""

    def fit(self, ins: FitIns) -> FitRes:
        """Return a successful fit result."""
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=ins.parameters,
            num_examples=1,
            metrics={"accuracy": 0.9},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Return a successful evaluate result."""
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=0.5,
            num_examples=1,
            metrics={"accuracy": 0.8},
        )


class _FailingFitClient(Client):
    """A ``Client`` whose ``fit`` raises for failure-event tests."""

    def fit(self, ins: FitIns) -> FitRes:
        """Raise an exception to simulate a fit failure."""
        raise RuntimeError("fit failed")


class _FailingEvaluateClient(Client):
    """A ``Client`` whose ``evaluate`` raises for failure-event tests."""

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Raise an exception to simulate an evaluate failure."""
        raise RuntimeError("evaluate failed")


def _make_message(message_type: str) -> Message:
    """Create a message with the given message type."""
    parameters = Parameters(tensors=[], tensor_type="")
    if message_type == "train":
        content = fitins_to_recorddict(FitIns(parameters, {}), keep_input=True)
    elif message_type == "evaluate":
        content = evaluateins_to_recorddict(
            EvaluateIns(parameters, {}), keep_input=True
        )
    else:
        content = RecordDict()
    return Message(
        content,
        dst_node_id=123,
        message_type=message_type,
        group_id="7",
    )


def test_client_app_emits_fit_events() -> None:
    """Test that a legacy ``ClientApp`` emits fit lifecycle events."""
    # Prepare
    callback = Mock()
    app = ClientApp(
        client_fn=lambda _: _MockClient(),
        event_callback=callback,
    )
    message = _make_message("train")
    context = Mock(spec=Context)

    # Execute
    app(message, context)

    # Assert
    events = [call.args[0].event for call in callback.call_args_list]
    assert events == [
        FL_NODE_FIT_STARTED,
        FL_NODE_FIT_COMPLETED,
    ]
    started_event = callback.call_args_list[0].args[0]
    assert (
        started_event.data
        == '{"type":"fl.node.fit.started","node_id":123,"server_round":7}'
    )
    completed_event = strict_json_loads(callback.call_args_list[1].args[0].data)
    assert completed_event.pop("elapsed_time") >= 0
    assert completed_event == {
        "type": FL_NODE_FIT_COMPLETED,
        "node_id": 123,
        "server_round": 7,
        "num_examples": 1,
        "accuracy": 0.9,
    }


def test_client_app_emits_evaluate_events() -> None:
    """Test that a legacy ``ClientApp`` emits evaluate lifecycle events."""
    # Prepare
    callback = Mock()
    app = ClientApp(
        client_fn=lambda _: _MockClient(),
        event_callback=callback,
    )
    message = _make_message("evaluate")
    context = Mock(spec=Context)

    # Execute
    app(message, context)

    # Assert
    events = [call.args[0].event for call in callback.call_args_list]
    assert events == [
        FL_NODE_EVALUATE_STARTED,
        FL_NODE_EVALUATE_COMPLETED,
    ]
    completed_event = strict_json_loads(callback.call_args_list[1].args[0].data)
    assert completed_event.pop("elapsed_time") >= 0
    assert completed_event == {
        "type": FL_NODE_EVALUATE_COMPLETED,
        "node_id": 123,
        "server_round": 7,
        "loss": 0.5,
        "num_examples": 1,
        "accuracy": 0.8,
    }


def test_client_app_emits_fit_failed_event() -> None:
    """Test that a legacy ``ClientApp`` emits a fit-failed event on exceptions."""
    # Prepare
    callback = Mock()
    app = ClientApp(
        client_fn=lambda _: _FailingFitClient(),
        event_callback=callback,
    )
    message = _make_message("train")
    context = Mock(spec=Context)

    # Execute and assert
    with pytest.raises(RuntimeError, match="fit failed"):
        app(message, context)

    events = [call.args[0].event for call in callback.call_args_list]
    assert events == [
        FL_NODE_FIT_STARTED,
        FL_NODE_FIT_FAILED,
    ]
    failed_event = strict_json_loads(callback.call_args_list[-1].args[0].data)
    assert failed_event["elapsed_time"] >= 0


def test_client_app_emits_evaluate_failed_event() -> None:
    """Test that a legacy ``ClientApp`` emits an evaluate-failed event on exceptions."""
    # Prepare
    callback = Mock()
    app = ClientApp(
        client_fn=lambda _: _FailingEvaluateClient(),
        event_callback=callback,
    )
    message = _make_message("evaluate")
    context = Mock(spec=Context)

    # Execute and assert
    with pytest.raises(RuntimeError, match="evaluate failed"):
        app(message, context)

    events = [call.args[0].event for call in callback.call_args_list]
    assert events == [
        FL_NODE_EVALUATE_STARTED,
        FL_NODE_EVALUATE_FAILED,
    ]


def test_client_app_emits_events_for_registered_train_handler() -> None:
    """Test lifecycle events are emitted for registered train handlers."""
    callback = Mock()
    app = ClientApp(event_callback=callback)

    @app.train()
    def train(message: Message, _: Context) -> Message:
        return Message(message.content, reply_to=message)

    app(_make_message("train"), Mock(spec=Context))

    assert [call.args[0].event for call in callback.call_args_list] == [
        FL_NODE_FIT_STARTED,
        FL_NODE_FIT_COMPLETED,
    ]


def test_client_app_emits_failed_event_for_error_reply() -> None:
    """Error replies are terminal node failures, not completed executions."""
    callback = Mock()
    app = ClientApp(event_callback=callback)

    @app.train()
    def train(message: Message, _: Context) -> Message:
        return Message(Error(code=1), reply_to=message)

    app(_make_message("train"), Mock(spec=Context))

    failed_event = strict_json_loads(callback.call_args_list[-1].args[0].data)
    assert [call.args[0].event for call in callback.call_args_list] == [
        FL_NODE_FIT_STARTED,
        FL_NODE_FIT_FAILED,
    ]
    assert failed_event.pop("elapsed_time") >= 0
    assert failed_event["error"] == "execution_failed"


def test_client_app_failure_event_does_not_include_exception_details() -> None:
    """Test failure events only contain a stable, safe error classification."""
    callback = Mock()
    app = ClientApp(event_callback=callback)

    @app.train()
    def train(_: Message, __: Context) -> Message:
        raise RuntimeError("secret-token-should-not-be-persisted")

    with pytest.raises(RuntimeError, match="secret-token-should-not-be-persisted"):
        app(_make_message("train"), Mock(spec=Context))

    failed_event = strict_json_loads(callback.call_args_list[-1].args[0].data)
    assert failed_event.pop("elapsed_time") >= 0
    assert failed_event == {
        "type": FL_NODE_FIT_FAILED,
        "node_id": 123,
        "server_round": 7,
        "error": "execution_failed",
    }


def test_client_app_continues_when_event_delivery_fails() -> None:
    """Test lifecycle event delivery failures do not affect ClientApp execution."""
    app = ClientApp(
        client_fn=lambda _: _MockClient(),
        event_callback=Mock(side_effect=RuntimeError("event delivery failed")),
    )

    app(_make_message("train"), Mock(spec=Context))


def test_client_app_emits_failed_event_when_lifespan_teardown_fails() -> None:
    """Test a lifespan teardown failure does not emit a completed event."""
    callback = Mock()
    app = ClientApp(event_callback=callback)

    @app.lifespan()
    def lifespan(_: Context) -> Iterator[None]:
        yield
        raise RuntimeError("lifespan teardown failed")

    @app.train()
    def train(message: Message, _: Context) -> Message:
        return Message(message.content, reply_to=message)

    with pytest.raises(RuntimeError, match="lifespan teardown failed"):
        app(_make_message("train"), Mock(spec=Context))

    assert [call.args[0].event for call in callback.call_args_list] == [
        FL_NODE_FIT_STARTED,
        FL_NODE_FIT_FAILED,
    ]
