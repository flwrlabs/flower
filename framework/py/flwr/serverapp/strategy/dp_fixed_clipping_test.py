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
"""Tests for fixed-clipping privacy accounting."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.supercore.privacy_accounting import (
    GaussianPrivacyEvent,
    NeighboringRelation,
    PrivacyAccountant,
    PrivacyConfig,
    PrivacySpent,
    SamplingMethod,
)

from ..exception import PrivacyBudgetExhausted
from ..grid import Grid
from .dp_fixed_clipping import (
    DifferentialPrivacyClientSideFixedClipping,
    DifferentialPrivacyFixedClippingBase,
    DifferentialPrivacyServerSideFixedClipping,
)
from .fedavg import FedAvg


def _accountant(*, would_exceed: bool = False) -> Mock:
    accountant = Mock(spec=PrivacyAccountant)
    accountant.config = PrivacyConfig(
        target_delta=1e-6,
        population_size=2,
        neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
        sampling_method=SamplingMethod.NO_AMPLIFICATION,
        max_epsilon=1.0 if would_exceed else None,
    )
    accountant.would_exceed.return_value = would_exceed
    accountant.get_privacy_spent.return_value = PrivacySpent(
        epsilon=0.5,
        delta=1e-6,
        num_releases=1,
        accounting_method="test",
    )
    return accountant


def _grid() -> Mock:
    grid = Mock(spec=Grid)
    grid.get_node_ids.return_value = [1, 2]
    return grid


def _reply(value: float, weight: float) -> Mock:
    reply = Mock(spec=Message)
    reply.content = RecordDict(
        {
            "arrays": ArrayRecord([np.array([value])]),
            "metrics": MetricRecord({"num-examples": weight}),
        }
    )
    reply.has_error.return_value = False
    return reply


def _wrapper(
    wrapper_cls: type[DifferentialPrivacyFixedClippingBase], accountant: Mock
) -> DifferentialPrivacyFixedClippingBase:
    return wrapper_cls(
        strategy=FedAvg(min_train_nodes=2, min_available_nodes=2),
        noise_multiplier=1.0,
        clipping_norm=10.0,
        num_sampled_clients=2,
        accountant=accountant,
    )


@pytest.mark.parametrize(
    "wrapper_cls",
    [
        DifferentialPrivacyClientSideFixedClipping,
        DifferentialPrivacyServerSideFixedClipping,
    ],
)
def test_successful_release_is_accounted_once(
    wrapper_cls: type[DifferentialPrivacyFixedClippingBase],
) -> None:
    """A successful uniformly aggregated release should consume one event."""
    accountant = _accountant()
    wrapper = _wrapper(wrapper_cls, accountant)
    initial_arrays = ArrayRecord([np.array([0.0])])
    messages = list(wrapper.configure_train(1, initial_arrays, ConfigRecord(), _grid()))
    assert len(messages) == 2

    replies = [_reply(1.0, 1.0), _reply(3.0, 1.0)]
    with patch("flwr.serverapp.strategy.dp_fixed_clipping.add_gaussian_noise_inplace"):
        arrays, metrics = wrapper.aggregate_train(1, replies)

    assert arrays is not None
    assert metrics is not None
    np.testing.assert_allclose(arrays.to_numpy_ndarrays()[0], np.array([2.0]))
    event = accountant.compose.call_args.args[0]
    assert event == GaussianPrivacyEvent(1.0, 2, 2)
    assert wrapper.privacy_spent() == accountant.get_privacy_spent.return_value
    with pytest.raises(ValueError, match="no pending privacy event"):
        wrapper.aggregate_train(1, replies)


def test_failed_round_does_not_consume_privacy() -> None:
    """Unsafe aggregation should abort without composition."""
    accountant = _accountant()
    wrapper = _wrapper(DifferentialPrivacyClientSideFixedClipping, accountant)
    list(wrapper.configure_train(1, ArrayRecord(), ConfigRecord(), _grid()))

    assert wrapper.aggregate_train(1, [_reply(1.0, 1.0)]) == (None, None)

    list(wrapper.configure_train(2, ArrayRecord(), ConfigRecord(), _grid()))
    with pytest.raises(ValueError, match="equal client weights"):
        wrapper.aggregate_train(2, [_reply(1.0, 1.0), _reply(3.0, 2.0)])

    empty_wrapper = DifferentialPrivacyClientSideFixedClipping(
        FedAvg(fraction_train=0.0),
        noise_multiplier=1.0,
        clipping_norm=1.0,
        num_sampled_clients=2,
        accountant=accountant,
    )
    assert not list(
        empty_wrapper.configure_train(1, ArrayRecord(), ConfigRecord(), _grid())
    )
    assert empty_wrapper.aggregate_train(1, []) == (None, None)
    accountant.compose.assert_not_called()


def test_budget_is_checked_before_training() -> None:
    """A release exceeding max_epsilon should not be configured."""
    accountant = _accountant(would_exceed=True)
    wrapper = _wrapper(DifferentialPrivacyClientSideFixedClipping, accountant)

    with pytest.raises(PrivacyBudgetExhausted):
        list(wrapper.configure_train(1, ArrayRecord(), ConfigRecord(), _grid()))
    accountant.compose.assert_not_called()
