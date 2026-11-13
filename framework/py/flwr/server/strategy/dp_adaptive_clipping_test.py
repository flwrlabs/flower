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
"""Tests for legacy central DP adaptive clipping strategies."""

import math
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest

from flwr.common import (
    Code,
    FitRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.supercore.differential_privacy import KEY_CLIPPING_NORM, KEY_NORM_BIT

from .dp_adaptive_clipping import (
    DifferentialPrivacyClientSideAdaptiveClipping,
    DifferentialPrivacyServerSideAdaptiveClipping,
)
from .fedavg import FedAvg


def _client_proxy(cid: int) -> Mock:
    proxy = Mock(spec=ClientProxy)
    proxy.cid = str(cid)
    return proxy


def _client_manager(num_clients: int) -> MagicMock:
    manager = MagicMock(spec=ClientManager)
    manager.sample.return_value = [_client_proxy(i) for i in range(num_clients)]
    manager.num_available.return_value = num_clients
    return manager


def _fit_res(update: float, norm_bit: bool | None = None) -> FitRes:
    metrics = {} if norm_bit is None else {KEY_NORM_BIT: norm_bit}
    return FitRes(
        status=Status(code=Code.OK, message="Ok"),
        parameters=ndarrays_to_parameters([np.array([update])]),
        num_examples=1,
        metrics=metrics,
    )


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
        (0.2, 8.0, 9.0),
        (0.5, 5.0, 6.0),
        (0.8, 2.0, 3.0),
    ],
)
def test_legacy_adaptive_clipping_tracks_target_clipped_quantile(
    wrapper_cls: (
        type[DifferentialPrivacyServerSideAdaptiveClipping]
        | type[DifferentialPrivacyClientSideAdaptiveClipping]
    ),
    target_clipped_quantile: float,
    expected_low: float,
    expected_high: float,
) -> None:
    """Legacy wrappers should adapt clipping_norm toward the target clipped quantile."""
    norms = np.arange(1.0, 11.0)
    num_clients = len(norms)
    strategy = wrapper_cls(
        strategy=FedAvg(
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
        ),
        noise_multiplier=0.0,
        num_sampled_clients=num_clients,
        initial_clipping_norm=0.5,
        target_clipped_quantile=target_clipped_quantile,
        clip_norm_lr=0.25,
        clipped_count_stddev=0.0,
    )
    is_client_side = issubclass(
        wrapper_cls, DifferentialPrivacyClientSideAdaptiveClipping
    )
    manager = _client_manager(num_clients)
    base_params: Parameters = ndarrays_to_parameters([np.array([0.0])])

    for rnd in range(1, 51):
        instructions = strategy.configure_fit(rnd, base_params, manager)
        threshold = strategy.clipping_norm
        if is_client_side:
            for _, fit_ins in instructions:
                assert fit_ins.config[KEY_CLIPPING_NORM] == pytest.approx(threshold)

        results = [
            (
                proxy,
                _fit_res(
                    float(min(n, threshold)) if is_client_side else float(n),
                    norm_bit=bool(n > threshold) if is_client_side else None,
                ),
            )
            for (proxy, _), n in zip(instructions, norms, strict=True)
        ]
        strategy.aggregate_fit(rnd, results, [])
        if rnd == 1:
            assert strategy.clipping_norm == pytest.approx(
                0.5 * math.exp(0.25 * (1.0 - target_clipped_quantile))
            )

    assert expected_low <= strategy.clipping_norm <= expected_high


@pytest.mark.parametrize(
    "wrapper_cls",
    [
        DifferentialPrivacyServerSideAdaptiveClipping,
        DifferentialPrivacyClientSideAdaptiveClipping,
    ],
)
def test_legacy_adaptive_clipping_shrinks_when_all_unclipped(
    wrapper_cls: (
        type[DifferentialPrivacyServerSideAdaptiveClipping]
        | type[DifferentialPrivacyClientSideAdaptiveClipping]
    ),
) -> None:
    """Legacy wrappers should decrease clipping_norm when no updates are clipped."""
    num_clients = 4
    strategy = wrapper_cls(
        strategy=FedAvg(
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
        ),
        noise_multiplier=0.0,
        num_sampled_clients=num_clients,
        initial_clipping_norm=5.0,
        target_clipped_quantile=0.4,
        clip_norm_lr=0.5,
        clipped_count_stddev=0.0,
    )
    is_client_side = issubclass(
        wrapper_cls, DifferentialPrivacyClientSideAdaptiveClipping
    )
    manager = _client_manager(num_clients)
    base_params = ndarrays_to_parameters([np.array([0.0])])
    instructions = strategy.configure_fit(1, base_params, manager)
    results = [
        (proxy, _fit_res(0.5, norm_bit=False if is_client_side else None))
        for proxy, _ in instructions
    ]
    strategy.aggregate_fit(1, results, [])

    assert strategy.clipping_norm == pytest.approx(5.0 * math.exp(-0.5 * 0.4))
