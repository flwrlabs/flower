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
"""Source-coupled regression tests for SecAgg+ stage-specific share ownership."""

import copy

import numpy as np
import pytest

from flwr.app import ConfigRecord, Context, Error, Message, RecordDict
from flwr.app.user_config import UserConfig
from flwr.client.mod.secure_aggregation.secaggplus_mod import secaggplus_mod
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.secure_aggregation.secaggplus_constants import (
    RECORD_KEY_CONFIGS,
    Key,
)
from flwr.compat.common import recorddict_compat as compat
from flwr.server.workflow.constant import MAIN_CONFIGS_RECORD, MAIN_PARAMS_RECORD
from flwr.server.workflow.constant import Key as WorkflowKey
from flwr.server.workflow.secure_aggregation import secaggplus_workflow
from flwr.supercore.task_identity import TaskIdentity


class _Proxy:
    def __init__(self, node_id: int) -> None:
        self.node_id = node_id


class _ClientManager:
    def __init__(self, count: int) -> None:
        self.count = count

    def num_available(self) -> int:
        return self.count


class _Strategy:
    def __init__(self, node_ids: list[int]) -> None:
        self.node_ids = node_ids
        self.aggregate: np.ndarray | None = None

    def configure_fit(self, server_round, parameters, client_manager):
        return [
            (_Proxy(node_id), FitIns(parameters, {}))
            for node_id in self.node_ids
        ]

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        arrays = parameters_to_ndarrays(results[0][1].parameters)
        self.aggregate = np.concatenate(
            [np.asarray(array).ravel() for array in arrays]
        )
        return results[0][1].parameters, {}


class _ServerState:
    def __init__(self) -> None:
        self.config_records = {
            MAIN_CONFIGS_RECORD: ConfigRecord({WorkflowKey.CURRENT_ROUND: 1})
        }
        parameters = ndarrays_to_parameters([np.zeros(1)])
        self.array_records = {
            MAIN_PARAMS_RECORD: compat.parameters_to_arrayrecord(
                parameters, keep_input=True
            )
        }


class _History:
    def add_metrics_distributed_fit(self, **kwargs) -> None:
        pass


class _ServerContext:
    def __init__(self, strategy: _Strategy, count: int) -> None:
        self.strategy = strategy
        self.client_manager = _ClientManager(count)
        self.state = _ServerState()
        self.history = _History()


class _Grid:
    """The real client SecAgg+ modifier with a synchronous, crashable transport."""

    def __init__(self, contexts: dict[int, Context], down: dict[int, int]) -> None:
        self.contexts = contexts
        self.down = down
        self.stage_index = 0
        self.errors: list[tuple[int, type[Exception]]] = []
        self.last_requests: list[Message] = []

    def send_and_receive(self, messages, *, timeout=None):
        self.last_requests = list(messages)
        replies = []
        for message in self.last_requests:
            node_id = message.metadata.dst_node_id
            if self.down.get(node_id, 100) <= self.stage_index:
                continue
            try:
                replies.append(
                    secaggplus_mod(
                        copy.deepcopy(message),
                        self.contexts[node_id],
                        self._call_next,
                    )
                )
            except Exception as exc:  # Transport returns client exceptions as errors.
                self.errors.append((node_id, type(exc)))
                replies.append(
                    Message(
                        Error(code=1, reason=f"{type(exc).__name__}: {exc}"),
                        reply_to=message,
                    )
                )
        return replies

    @staticmethod
    def _call_next(message: Message, context: Context) -> Message:
        value = float(context.node_id + 1)
        fitres = FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters(
                [np.array([value, value / 2], dtype=float)]
            ),
            num_examples=4,
            metrics={},
        )
        return Message(
            compat.fitres_to_recorddict(fitres, keep_input=True),
            reply_to=message,
        )


def _run_round(monkeypatch, *, n: int, k: int, t: int, down: dict[int, int]):
    """Run real Flower server stages and client cryptographic logic in-process."""
    TaskIdentity.run_id = 1
    TaskIdentity.node_id = 0
    TaskIdentity.task_id = 0

    np.random.seed(74)
    monkeypatch.setattr(
        secaggplus_workflow.random,
        "shuffle",
        lambda ids: ids.sort(),
    )
    node_ids = list(range(n))
    contexts = {
        node_id: Context(
            run_id=1,
            node_id=node_id,
            node_config=UserConfig(),
            state=RecordDict(),
            run_config=UserConfig(),
        )
        for node_id in node_ids
    }
    grid = _Grid(contexts, down)
    strategy = _Strategy(node_ids)
    context = _ServerContext(strategy, n)
    workflow = secaggplus_workflow.SecAggPlusWorkflow(
        num_shares=k, reconstruction_threshold=t
    )
    state = secaggplus_workflow.WorkflowState()
    results = {}
    shares = frozenset()
    contributors = frozenset()
    requests = []
    for index, name in enumerate(
        ("setup", "share_keys", "collect_masked_vectors", "unmask")
    ):
        grid.stage_index = index
        results[name] = getattr(workflow, name + "_stage")(grid, context, state)
        if name == "share_keys":
            shares = frozenset(state.active_node_ids)
        if name == "collect_masked_vectors":
            contributors = frozenset(state.active_node_ids)
        if name == "unmask":
            requests = list(grid.last_requests)
        if not results[name]:
            break
    return results, shares, contributors, strategy.aggregate, requests, grid, workflow, state


def _assert_correct(result, *, n: int, expected_contributors: set[int]):
    stages, owners, contributors, aggregate, requests, grid, workflow, state = result
    assert stages == {
        "setup": True,
        "share_keys": True,
        "collect_masked_vectors": True,
        "unmask": True,
    }
    assert contributors == expected_contributors
    assert not grid.errors
    assert aggregate is not None
    expected_mean = sum(nid + 1 for nid in contributors) / len(contributors)
    reference = np.array([expected_mean, expected_mean / 2])
    q = round(4 / workflow.max_weight * workflow.quantization_range)
    assert np.max(np.abs(aggregate - reference)) < 2 * workflow.clipping_range / q

    # Never request both the random mask seed and private-key shares for an owner.
    # Never ask for private-key shares of sampled nodes that never generated shares.
    for request in requests:
        configs = request.content.config_records[RECORD_KEY_CONFIGS]
        active = set(configs[Key.ACTIVE_NODE_ID_LIST])
        dead = set(configs[Key.DEAD_NODE_ID_LIST])
        assert active.isdisjoint(dead)
        assert active | dead <= owners
    assert owners == state.share_owner_node_ids


@pytest.mark.parametrize("stage", [0, 1])
def test_dropout_before_share_ownership_is_ignored_at_unmask(
    monkeypatch, stage: int
) -> None:
    """Setup or share-key departure must not create nonexistent recovery shares."""
    result = _run_round(monkeypatch, n=4, k=4, t=3, down={1: stage})
    _assert_correct(result, n=4, expected_contributors={0, 2, 3})
    assert result[1] == {0, 2, 3}
    assert all(
        not request.content.config_records[RECORD_KEY_CONFIGS][Key.DEAD_NODE_ID_LIST]
        for request in result[4]
    )


@pytest.mark.parametrize("topology", [(4, 4, 3), (8, 5, 3)])
def test_post_share_dropout_preserves_reconstructable_owner(
    monkeypatch, topology: tuple[int, int, int]
) -> None:
    """A dropped contributor's existing key shares remain needed."""
    n, k, t = topology
    result = _run_round(monkeypatch, n=n, k=k, t=t, down={1: 2})
    _assert_correct(result, n=n, expected_contributors=set(range(n)) - {1})
    assert result[1] == set(range(n))
    assert any(
        1 in request.content.config_records[RECORD_KEY_CONFIGS][Key.DEAD_NODE_ID_LIST]
        for request in result[4]
    )


@pytest.mark.parametrize("topology", [(4, 4, 3), (8, 5, 3)])
@pytest.mark.parametrize("stage", [0, 1])
def test_sparse_and_dense_early_dropout(
    monkeypatch, topology: tuple[int, int, int], stage: int
) -> None:
    n, k, t = topology
    result = _run_round(monkeypatch, n=n, k=k, t=t, down={1: stage})
    _assert_correct(result, n=n, expected_contributors=set(range(n)) - {1})


def test_mixed_setup_and_post_share_dropout(monkeypatch) -> None:
    """Exclude never-generated shares while retaining a later dropped owner."""
    result = _run_round(monkeypatch, n=5, k=5, t=3, down={1: 0, 2: 2})
    _assert_correct(result, n=5, expected_contributors={0, 3, 4})
    assert result[1] == {0, 2, 3, 4}
    for request in result[4]:
        dead = set(
            request.content.config_records[RECORD_KEY_CONFIGS][Key.DEAD_NODE_ID_LIST]
        )
        assert 1 not in dead
        assert 2 in dead


def test_healthy_round_unchanged(monkeypatch) -> None:
    result = _run_round(monkeypatch, n=4, k=4, t=3, down={})
    _assert_correct(result, n=4, expected_contributors=set(range(4)))


def test_insufficient_setup_responses_halt(monkeypatch) -> None:
    result = _run_round(monkeypatch, n=4, k=4, t=3, down={1: 0, 2: 0})
    assert result[0] == {"setup": False}
    assert result[3] is None


def test_insufficient_unmask_responses_halt(monkeypatch) -> None:
    result = _run_round(monkeypatch, n=4, k=4, t=3, down={1: 3, 2: 3})
    assert result[0]["collect_masked_vectors"] is True
    assert result[0]["unmask"] is False
    assert result[3] is None
