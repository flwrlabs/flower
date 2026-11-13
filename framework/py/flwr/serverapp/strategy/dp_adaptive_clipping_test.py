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
"""Tests for adaptive-clipping central DP strategies."""

import math
from unittest.mock import Mock

import numpy as np
import pytest

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.supercore.differential_privacy import KEY_CLIPPING_NORM, KEY_NORM_BIT
from flwr.supercore.task_identity import TaskIdentity

from ..grid import Grid
from .dp_adaptive_clipping import (
    DifferentialPrivacyAdaptiveBase,
    DifferentialPrivacyClientSideAdaptiveClipping,
    DifferentialPrivacyServerSideAdaptiveClipping,
)
from .fedavg import FedAvg


@pytest.fixture(autouse=True)
def task_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the task identity used by strategy message tests."""
    monkeypatch.setattr(TaskIdentity, "_task_id", 123)
    monkeypatch.setattr(TaskIdentity, "_run_id", 456)
    monkeypatch.setattr(TaskIdentity, "_node_id", 789)


def _grid(num_nodes: int) -> Mock:
    grid = Mock(spec=Grid)
    grid.get_node_ids.return_value = list(range(1, num_nodes + 1))
    return grid


def _reply(update: np.ndarray, norm_bit: int | None = None) -> Mock:
    metrics: dict[str, int | float] = {"num-examples": 1.0}
    if norm_bit is not None:
        metrics[KEY_NORM_BIT] = norm_bit
    reply = Mock(spec=Message)
    reply.content = RecordDict(
        {
            "arrays": ArrayRecord([update]),
            "metrics": MetricRecord(metrics),
        }
    )
    reply.has_error.return_value = False
    return reply


def _make_wrapper(
    wrapper_cls: type[DifferentialPrivacyAdaptiveBase],
    num_clients: int,
    initial_clipping_norm: float = 0.1,
    target_clipped_quantile: float = 0.5,
    clip_norm_lr: float = 0.2,
) -> DifferentialPrivacyAdaptiveBase:
    return wrapper_cls(
        strategy=FedAvg(
            min_train_nodes=num_clients,
            min_available_nodes=num_clients,
        ),
        noise_multiplier=0.0,
        num_sampled_clients=num_clients,
        initial_clipping_norm=initial_clipping_norm,
        target_clipped_quantile=target_clipped_quantile,
        clip_norm_lr=clip_norm_lr,
        clipped_count_stddev=0.0,
    )


@pytest.mark.parametrize(
    "wrapper_cls",
    [
        DifferentialPrivacyServerSideAdaptiveClipping,
        DifferentialPrivacyClientSideAdaptiveClipping,
    ],
)
def test_adaptive_clipping_converges_to_median_norm(
    wrapper_cls: type[DifferentialPrivacyAdaptiveBase],
) -> None:
    """Clipping norm should grow from a small initial value toward the median norm."""
    norms = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
    median_norm = float(np.median(norms))
    num_clients = len(norms)
    grid = _grid(num_clients)
    wrapper = _make_wrapper(
        wrapper_cls,
        num_clients=num_clients,
        initial_clipping_norm=0.1,
        target_clipped_quantile=0.5,
        clip_norm_lr=0.2,
    )

    is_client_side = issubclass(
        wrapper_cls, DifferentialPrivacyClientSideAdaptiveClipping
    )
    initial_arrays = ArrayRecord([np.array([0.0])])

    for rnd in range(1, 61):
        config = ConfigRecord()
        list(wrapper.configure_train(rnd, initial_arrays, config, grid))
        threshold = wrapper.clipping_norm
        if is_client_side:
            assert config[KEY_CLIPPING_NORM] == pytest.approx(threshold)

        replies = [
            _reply(
                np.array([float(min(n, threshold)) if is_client_side else float(n)]),
                norm_bit=int(n > threshold) if is_client_side else None,
            )
            for n in norms
        ]
        wrapper.aggregate_train(rnd, replies)
        if rnd == 1:
            # All 10 updates exceed 0.1, so clipping_norm must grow by exp(0.2 * 0.5).
            assert wrapper.clipping_norm == pytest.approx(0.1 * math.exp(0.1))

    assert wrapper.clipping_norm == pytest.approx(median_norm, rel=0.1)


@pytest.mark.parametrize(
    "wrapper_cls",
    [
        DifferentialPrivacyServerSideAdaptiveClipping,
        DifferentialPrivacyClientSideAdaptiveClipping,
    ],
)
@pytest.mark.parametrize(
    ("target_clipped_quantile", "expected_low", "expected_high"),
    [
        # 20% clipped means 80% stay unclipped, placing norm between 8.0 and 9.0.
        (0.2, 8.0, 9.0),
        # 80% clipped means 20% stay unclipped, placing norm between 2.0 and 3.0.
        (0.8, 2.0, 3.0),
    ],
)
def test_adaptive_clipping_respects_non_half_target_quantile(
    wrapper_cls: type[DifferentialPrivacyAdaptiveBase],
    target_clipped_quantile: float,
    expected_low: float,
    expected_high: float,
) -> None:
    """Clipping norm should stabilize where clipped fraction matches target quantile."""
    norms = np.arange(1.0, 11.0)
    num_clients = len(norms)
    grid = _grid(num_clients)
    wrapper = _make_wrapper(
        wrapper_cls,
        num_clients=num_clients,
        initial_clipping_norm=5.0,
        target_clipped_quantile=target_clipped_quantile,
        clip_norm_lr=0.2,
    )

    is_client_side = issubclass(
        wrapper_cls, DifferentialPrivacyClientSideAdaptiveClipping
    )
    initial_arrays = ArrayRecord([np.array([0.0])])

    for rnd in range(1, 51):
        list(wrapper.configure_train(rnd, initial_arrays, ConfigRecord(), grid))
        threshold = wrapper.clipping_norm
        replies = [
            _reply(
                np.array([float(min(n, threshold)) if is_client_side else float(n)]),
                norm_bit=int(n > threshold) if is_client_side else None,
            )
            for n in norms
        ]
        wrapper.aggregate_train(rnd, replies)

    assert expected_low <= wrapper.clipping_norm <= expected_high


@pytest.mark.parametrize(
    "wrapper_cls",
    [
        DifferentialPrivacyServerSideAdaptiveClipping,
        DifferentialPrivacyClientSideAdaptiveClipping,
    ],
)
def test_adaptive_clipping_shrinks_when_no_updates_clipped(
    wrapper_cls: type[DifferentialPrivacyAdaptiveBase],
) -> None:
    """Clipping norm should decrease when fewer updates are clipped than targeted."""
    num_clients = 4
    grid = _grid(num_clients)
    wrapper = _make_wrapper(
        wrapper_cls,
        num_clients=num_clients,
        initial_clipping_norm=5.0,
        target_clipped_quantile=0.4,
        clip_norm_lr=0.5,
    )

    is_client_side = issubclass(
        wrapper_cls, DifferentialPrivacyClientSideAdaptiveClipping
    )
    list(
        wrapper.configure_train(1, ArrayRecord([np.array([0.0])]), ConfigRecord(), grid)
    )
    replies = [
        _reply(np.array([0.5]), norm_bit=0 if is_client_side else None)
        for _ in range(num_clients)
    ]
    wrapper.aggregate_train(1, replies)

    assert wrapper.clipping_norm == pytest.approx(5.0 * math.exp(-0.5 * 0.4))
