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
"""Tests for the adaptive-clipping strategy wrappers."""

import math
from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
import pytest

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.supercore.differential_privacy import (
    KEY_CLIPPING_NORM,
    KEY_NORM_BIT,
    compute_adaptive_clip_model_update,
)
from flwr.supercore.task_identity import TaskIdentity

from ..grid import Grid
from .dp_adaptive_clipping import (
    DifferentialPrivacyAdaptiveBase,
    DifferentialPrivacyClientSideAdaptiveClipping,
    DifferentialPrivacyServerSideAdaptiveClipping,
)
from .fedavg import FedAvg

NUM_CLIENTS = 20
DIM = 8
CLIP_NORM_LR = 0.2
TARGET_QUANTILE = 0.5


@pytest.fixture(autouse=True)
def task_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the task identity used by strategy message tests."""
    monkeypatch.setattr(TaskIdentity, "_task_id", 123)
    monkeypatch.setattr(TaskIdentity, "_run_id", 456)
    monkeypatch.setattr(TaskIdentity, "_node_id", 789)


def _grid() -> Mock:
    grid = Mock(spec=Grid)
    grid.get_node_ids.return_value = list(range(NUM_CLIENTS))
    return grid


def _updates(norms: list[float]) -> list[np.ndarray]:
    """Return one update per client with exactly the requested L2 norm."""
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(len(norms), DIM))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return [direction * norm for direction, norm in zip(directions, norms, strict=True)]


def _reply(update: np.ndarray, norm_bit: int | None = None) -> Mock:
    metrics = MetricRecord({"num-examples": 1.0})
    if norm_bit is not None:
        metrics[KEY_NORM_BIT] = norm_bit
    reply = Mock(spec=Message)
    reply.content = RecordDict({"arrays": ArrayRecord([update]), "metrics": metrics})
    reply.has_error.return_value = False
    return reply


def _server_side(initial_clipping_norm: float) -> DifferentialPrivacyAdaptiveBase:
    return DifferentialPrivacyServerSideAdaptiveClipping(
        strategy=FedAvg(min_train_nodes=NUM_CLIENTS, min_available_nodes=NUM_CLIENTS),
        noise_multiplier=0.0,  # also sets the clipped-count noise to 0: deterministic
        num_sampled_clients=NUM_CLIENTS,
        initial_clipping_norm=initial_clipping_norm,
        target_clipped_quantile=TARGET_QUANTILE,
        clip_norm_lr=CLIP_NORM_LR,
    )


def _client_side(initial_clipping_norm: float) -> DifferentialPrivacyAdaptiveBase:
    return DifferentialPrivacyClientSideAdaptiveClipping(
        strategy=FedAvg(min_train_nodes=NUM_CLIENTS, min_available_nodes=NUM_CLIENTS),
        noise_multiplier=0.0,
        num_sampled_clients=NUM_CLIENTS,
        initial_clipping_norm=initial_clipping_norm,
        target_clipped_quantile=TARGET_QUANTILE,
        clip_norm_lr=CLIP_NORM_LR,
    )


def _run_round(wrapper: DifferentialPrivacyAdaptiveBase, norms: list[float]) -> None:
    """Run one round in which client i sends an update of norm `norms[i]`."""
    config = ConfigRecord()
    arrays = ArrayRecord([np.zeros(DIM)])
    wrapper.configure_train(1, arrays, config, _grid())
    updates = _updates(norms)
    if isinstance(wrapper, DifferentialPrivacyClientSideAdaptiveClipping):
        # Clients clip with the norm the server sent and report the bit that
        # `adaptiveclipping_mod` reports.
        clipping_norm = config[KEY_CLIPPING_NORM]
        assert isinstance(clipping_norm, float)
        replies = []
        for update in updates:
            params = [update.copy()]
            norm_bit = compute_adaptive_clip_model_update(
                params, [np.zeros(DIM)], clipping_norm
            )
            replies.append(_reply(params[0], int(norm_bit)))
    else:
        replies = [_reply(update) for update in updates]
    wrapper.aggregate_train(1, replies)


WRAPPERS = [
    pytest.param(_server_side, id="server-side"),
    pytest.param(_client_side, id="client-side"),
]


@pytest.mark.parametrize("make_wrapper", WRAPPERS)
def test_clipping_norm_grows_when_every_update_is_clipped(
    make_wrapper: Callable[[float], DifferentialPrivacyAdaptiveBase],
) -> None:
    """With nothing under the bound the norm must go up, by exp(lr * target)."""
    wrapper = make_wrapper(1.0)

    _run_round(wrapper, [10.0] * NUM_CLIENTS)

    assert wrapper.clipping_norm == pytest.approx(
        math.exp(CLIP_NORM_LR * TARGET_QUANTILE)
    )


@pytest.mark.parametrize("make_wrapper", WRAPPERS)
def test_clipping_norm_shrinks_when_no_update_is_clipped(
    make_wrapper: Callable[[float], DifferentialPrivacyAdaptiveBase],
) -> None:
    """With everything under the bound the norm must go down."""
    wrapper = make_wrapper(10.0)

    _run_round(wrapper, [1.0] * NUM_CLIENTS)

    assert wrapper.clipping_norm == pytest.approx(
        10.0 * math.exp(-CLIP_NORM_LR * (1.0 - TARGET_QUANTILE))
    )


@pytest.mark.parametrize("make_wrapper", WRAPPERS)
def test_clipping_norm_tracks_target_quantile(
    make_wrapper: Callable[[float], DifferentialPrivacyAdaptiveBase],
) -> None:
    """The norm should settle at the target quantile of the update norms."""
    norms = [float(n) for n in np.linspace(1.0, 3.0, NUM_CLIENTS)]
    wrapper = make_wrapper(0.1)

    for _ in range(120):
        _run_round(wrapper, norms)

    median = float(np.quantile(norms, TARGET_QUANTILE))
    assert wrapper.clipping_norm == pytest.approx(median, rel=0.1)
