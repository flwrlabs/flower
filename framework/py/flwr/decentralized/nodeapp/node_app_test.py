# Copyright 2026 Inria (cyrille kenfack & davide frey). All Rights Reserved.
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
"""Unit tests for decentralized NodeApp."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from flwr.decentralized.nodeapp import (
    NodeApp,
    create_nodeapps_from_pyproject,
    load_nodeapp_configs_from_pyproject,
)
from flwr.decentralized.common.message import AggregateRequest
from flwr.decentralized.common.typing import Action
from flwr.decentralized.nodeapp.node_app import _load_handler, _load_object

TRAIN_CALLS: list[str] = []
EVAL_CALLS: list[str] = []


COMPONENT_APP1 = NodeApp(subject="component_app1", run_config={"local-epochs": 3})
COMPONENT_APP2 = NodeApp(subject="component_app2", run_config={"batch-size": 16})


def mapped_train_handler(message: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
    del kwargs
    TRAIN_CALLS.append(message)


def mapped_evaluate_handler(message: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
    del kwargs
    EVAL_CALLS.append(message)


def test_train_decorator_and_handle_message() -> None:
    """Register train callback and ensure dispatch calls it."""
    app = NodeApp(subject="trainer", run_config={"local-epochs": 2})
    calls: list[tuple[str, str | None, dict]] = []

    @app.train()
    def train(
        message: str,
        node_id: str | None,
        run_config: dict,
        subject: str,
        app: NodeApp,
    ) -> None:
        del subject, app
        calls.append((message, node_id, run_config))

    app.handle_message("hello", node_id="node-1")

    assert len(calls) == 1
    assert calls[0][0] == "hello"
    assert calls[0][1] == "node-1"
    assert calls[0][2]["local-epochs"] == 2


def test_evaluate_decorator_dispatch_from_json_event() -> None:
    """Route evaluate event to evaluate callback when using JSON envelope."""
    app = NodeApp(subject="eval")
    received: list[str] = []

    @app.train()
    def train(**kwargs) -> None:  # type: ignore[no-untyped-def]
        del kwargs

    @app.evaluate()
    def evaluate(message: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
        del kwargs
        received.append(message)

    app.handle_message('{"event": "evaluate", "payload": "metric"}')

    assert received == ["metric"]


def test_missing_train_callback_raises_runtime_error() -> None:
    """Raise RuntimeError if train event is received but no train callback exists."""
    app = NodeApp(subject="trainer")

    try:
        app.handle_message("hello")
        raised = False
    except RuntimeError:
        raised = True

    assert raised


def test_periodic_run_invokes_train_callback_when_registered() -> None:
    """Periodic run should call train callback with periodic marker."""
    app = NodeApp(subject="periodic")
    called = {"value": False}

    @app.train()
    def train(message: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
        del message
        del kwargs
        called["value"] = True

    app.periodic_run(view=["a", "b"], node_id="node-1")

    assert called["value"] is True


def test_load_nodeapp_configs_from_pyproject(tmp_path: Path) -> None:
    """Load multiple NodeApp subject configs from pyproject TOML."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[tool.flwr.nodeapp]",
                "default-timeout = 33",
                "",
                "[tool.flwr.nodeapp.apps.trainer]",
                'subject = "trainer"',
                "timeout = 20",
                "",
                "[tool.flwr.nodeapp.apps.trainer.config]",
                "local-epochs = 2",
                "lr = 0.1",
                "",
                "[tool.flwr.nodeapp.apps.evaluator]",
                'subject = "evaluator"',
                "",
                "[tool.flwr.nodeapp.apps.evaluator.config]",
                "batch-size = 64",
            ]
        )
    )

    configs = load_nodeapp_configs_from_pyproject(pyproject)

    assert set(configs.keys()) == {"trainer", "evaluator"}
    assert configs["trainer"]["timeout"] == 20
    assert configs["trainer"]["run_config"]["local-epochs"] == 2
    assert configs["evaluator"]["timeout"] == 33
    assert configs["evaluator"]["run_config"]["batch-size"] == 64


def test_create_nodeapps_from_pyproject(tmp_path: Path) -> None:
    """Instantiate NodeApp objects from pyproject configuration."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[tool.flwr.nodeapp]",
                "default-timeout = 30",
                "",
                "[tool.flwr.nodeapp.apps.subject_a]",
                'subject = "subject_a"',
                "",
                "[tool.flwr.nodeapp.apps.subject_a.config]",
                "rounds = 5",
            ]
        )
    )

    apps = create_nodeapps_from_pyproject(pyproject)

    assert list(apps.keys()) == ["subject_a"]
    assert isinstance(apps["subject_a"], NodeApp)
    assert apps["subject_a"].run_config.rounds == 5


def test_create_nodeapps_from_pyproject_with_handler_mapping(tmp_path: Path) -> None:
    """Attach mapped train/evaluate handlers from pyproject import specs."""
    TRAIN_CALLS.clear()
    EVAL_CALLS.clear()

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[tool.flwr.nodeapp]",
                "default-timeout = 30",
                "",
                "[tool.flwr.nodeapp.apps.subject_a]",
                'subject = "subject_a"',
                'train = "flwr.decentralized.nodeapp.node_app_test:mapped_train_handler"',
                'evaluate = "flwr.decentralized.nodeapp.node_app_test:mapped_evaluate_handler"',
            ]
        )
    )

    apps = create_nodeapps_from_pyproject(pyproject)
    app = apps["subject_a"]

    app.handle_message("hello")
    app.handle_message('{"event": "evaluate", "payload": "metric"}')

    assert TRAIN_CALLS == ["hello"]
    assert EVAL_CALLS == ["metric"]


def test_create_nodeapps_from_pyproject_components_style(tmp_path: Path) -> None:
    """Load pre-built NodeApp objects from [tool.flwr.app.components]."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[tool.flwr.app.components]",
                'nodeapp1 = "flwr.decentralized.nodeapp.node_app_test:COMPONENT_APP1"',
                'nodeapp2 = "flwr.decentralized.nodeapp.node_app_test:COMPONENT_APP2"',
            ]
        )
    )

    apps = create_nodeapps_from_pyproject(pyproject)

    assert set(apps.keys()) == {"nodeapp1", "nodeapp2"}
    assert apps["nodeapp1"] is COMPONENT_APP1
    assert apps["nodeapp2"] is COMPONENT_APP2


def test_create_nodeapps_from_pyproject_components_style_invalid_type(
    tmp_path: Path,
) -> None:
    """Raise ValueError when a nodeapp component does not point to NodeApp."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "\n".join(
            [
                "[tool.flwr.app.components]",
                'nodeapp1 = "pathlib:Path"',
            ]
        )
    )

    try:
        create_nodeapps_from_pyproject(pyproject)
        raised = False
    except ValueError:
        raised = True

    assert raised


def test_push_protocol_ignores_non_push_actions() -> None:
    """Ignore aggregate actions that are incompatible with PUSH protocol."""
    app = NodeApp(subject="trainer", run_config={"rounds": 2, "protocol": "push"})
    app.current_round = 1
    seen: list[str] = []

    app._update_own_parameters = cast(  # pylint: disable=protected-access
        object,
        lambda request: seen.append(request.action.value),
    )

    req = app.create_aggregate_request(
        action=Action.CANCEL,
        source_node_id="peer-a",
        round_number=1,
        msg=None,
    )

    app.handle_message(app.aggregate_request_to_str(req))

    assert seen == []


def test_aggregate_request_deduplication_in_same_round() -> None:
    """Process at most once per (source, action, round) request tuple."""
    app = NodeApp(subject="trainer", run_config={"rounds": 2, "protocol": "push"})
    app.current_round = 1
    seen: list[str] = []

    app._update_own_parameters = cast(  # pylint: disable=protected-access
        object,
        lambda request: seen.append(request.source_node_id),
    )

    req = app.create_aggregate_request(
        action=Action.PUSH,
        source_node_id="peer-a",
        round_number=1,
        msg=None,
    )
    encoded = app.aggregate_request_to_str(req)

    app.handle_message(encoded)
    app.handle_message(encoded)

    assert seen == ["peer-a"]


def test_aggregate_request_deduplication_allows_distinct_payloads() -> None:
    """Allow multiple requests when payload message differs within same round."""
    app = NodeApp(subject="trainer", run_config={"rounds": 2, "protocol": "push"})
    app.current_round = 1
    seen: list[str] = []

    app._update_own_parameters = cast(  # pylint: disable=protected-access
        object,
        lambda request: seen.append(request.msg.object_id if request.msg else ""),
    )

    msg_a = app.create_message(message_type="train", config=app.train_config)
    msg_b = app.create_message(message_type="train", config=app.train_config)

    req_a = app.create_aggregate_request(
        action=Action.PUSH,
        source_node_id="peer-a",
        round_number=1,
        msg=msg_a,
    )
    req_b = app.create_aggregate_request(
        action=Action.PUSH,
        source_node_id="peer-a",
        round_number=1,
        msg=msg_b,
    )

    app.handle_message(app.aggregate_request_to_str(req_a))
    app.handle_message(app.aggregate_request_to_str(req_b))

    assert len(seen) == 2


def test_aggregate_handler_error_does_not_fallback_to_train_callback() -> None:
    """Aggregate handler errors should not trigger legacy event fallback."""
    app = NodeApp(subject="trainer", run_config={"rounds": 2})
    app.current_round = 1
    train_calls: list[str] = []

    @app.train()
    def train(message: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
        del kwargs
        train_calls.append(message)

    app._handle_push_request = cast(  # pylint: disable=protected-access
        object,
        lambda request: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    req = app.create_aggregate_request(
        action=Action.PUSH,
        source_node_id="peer-b",
        round_number=1,
        msg=None,
    )

    with pytest.raises(RuntimeError, match="boom"):
        app.handle_message(app.aggregate_request_to_str(req))

    assert train_calls == []


def test_invalid_aggregate_payload_falls_back_to_train_event() -> None:
    """Fallback to train/evaluate parsing when AggregateRequest decoding fails."""
    app = NodeApp(subject="trainer")
    seen: list[str] = []

    @app.train()
    def train(message: str, **kwargs) -> None:  # type: ignore[no-untyped-def]
        del kwargs
        seen.append(message)

    invalid_aggregate_like_payload = '{"foo": "bar"}'
    app.handle_message(invalid_aggregate_like_payload)

    assert seen == [invalid_aggregate_like_payload]


def test_aggregate_request_with_empty_source_is_ignored() -> None:
    """Ignore aggregate requests that do not carry a source node id."""
    app = NodeApp(subject="trainer", run_config={"rounds": 2, "protocol": "push"})
    app.current_round = 1
    seen: list[str] = []

    app._update_own_parameters = cast(  # pylint: disable=protected-access
        object,
        lambda request: seen.append("called"),
    )

    req = app.create_aggregate_request(
        action=Action.PUSH,
        source_node_id="",
        round_number=1,
        msg=None,
    )
    app.handle_message(app.aggregate_request_to_str(req))

    assert seen == []


@pytest.mark.parametrize("spec", ["invalidspec", "pathlib:"])
def test_load_object_rejects_invalid_spec_format(spec: str) -> None:
    """`_load_object` should reject malformed `<module>:<symbol>` specs."""
    with pytest.raises(ValueError):
        _load_object(spec)


@pytest.mark.parametrize("spec", ["invalidspec", "pathlib:"])
def test_load_handler_rejects_invalid_spec_format(spec: str) -> None:
    """`_load_handler` should reject malformed `<module>:<symbol>` specs."""
    with pytest.raises(ValueError):
        _load_handler(spec)


@pytest.mark.parametrize("round_number", [0, 1, 2, 3])
def test_aggregate_request_round_validation(round_number: int) -> None:
    """Only handle aggregate requests for the current active round."""
    app = NodeApp(subject="trainer", run_config={"rounds": 3})
    app.current_round = 2
    seen: list[AggregateRequest] = []

    app._handle_push_request = cast(  # pylint: disable=protected-access
        object,
        lambda request: seen.append(request),
    )

    req = app.create_aggregate_request(
        action=Action.PUSH,
        source_node_id="peer-b",
        round_number=round_number,
        msg=None,
    )

    app.handle_message(app.aggregate_request_to_str(req))

    expected = 1 if round_number == 2 else 0
    assert len(seen) == expected
