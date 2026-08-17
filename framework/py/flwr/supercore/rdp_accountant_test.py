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
"""Tests for the Renyi differential privacy accountant."""

import json

import pytest

from .privacy_accounting import (
    GaussianPrivacyEvent,
    NeighboringRelation,
    PrivacyConfig,
    SamplingMethod,
)
from .rdp_accountant import RdpAccountant

TARGET_DELTA = 1e-6


def _config(
    sampling_method: SamplingMethod = SamplingMethod.POISSON,
    neighboring_relation: NeighboringRelation = (NeighboringRelation.ADD_OR_REMOVE_ONE),
    population_size: int = 1_000,
    max_epsilon: float | None = None,
) -> PrivacyConfig:
    return PrivacyConfig(
        target_delta=TARGET_DELTA,
        population_size=population_size,
        neighboring_relation=neighboring_relation,
        sampling_method=sampling_method,
        max_epsilon=max_epsilon,
    )


def _event(
    noise_multiplier: float = 1.2,
    sample_size: int = 10,
    population_size: int = 1_000,
) -> GaussianPrivacyEvent:
    return GaussianPrivacyEvent(
        noise_multiplier=noise_multiplier,
        sample_size=sample_size,
        population_size=population_size,
    )


@pytest.mark.parametrize(
    ("config", "event", "count", "expected_epsilon", "expected_delta"),
    [
        (
            _config(
                sampling_method=SamplingMethod.NO_AMPLIFICATION,
                population_size=100,
            ),
            _event(sample_size=100, population_size=100),
            1,
            4.253499721177566,
            0.0005189705118041026,
        ),
        (
            _config(),
            _event(),
            100,
            0.975892494240961,
            2.8268017638996466e-17,
        ),
        (
            _config(
                sampling_method=SamplingMethod.WITHOUT_REPLACEMENT,
                neighboring_relation=NeighboringRelation.REPLACE_ONE,
            ),
            _event(noise_multiplier=1.5, sample_size=100),
            10,
            2.7582097098379643,
            1.684726385938029e-7,
        ),
    ],
)
def test_supported_models_match_reference_values(
    config: PrivacyConfig,
    event: GaussianPrivacyEvent,
    count: int,
    expected_epsilon: float,
    expected_delta: float,
) -> None:
    """All supported accounting models should match reference values."""
    accountant = RdpAccountant(config)

    accountant.compose(event, count)

    assert accountant.get_epsilon(TARGET_DELTA) == pytest.approx(expected_epsilon)
    assert accountant.get_delta(3.0) == pytest.approx(expected_delta)
    assert accountant.num_releases == count


def test_budget_projection_does_not_mutate_composition() -> None:
    """Prospective accounting should enforce a budget without consuming it."""
    accountant = RdpAccountant(_config(max_epsilon=0.1))

    assert accountant.would_exceed(_event())
    assert accountant.num_releases == 0

    accountant.compose(_event())
    spent = accountant.get_privacy_spent()
    assert spent.num_releases == 1
    assert spent.delta == TARGET_DELTA
    assert spent.accounting_method == "rdp"


def test_state_dict_round_trip_and_config_validation() -> None:
    """Serialized state should restore exactly under the same configuration."""
    accountant = RdpAccountant(_config())
    accountant.compose(_event(), count=2)
    state = json.loads(json.dumps(accountant.state_dict()))

    restored = RdpAccountant(_config())
    restored.load_state_dict(state)
    assert restored.get_privacy_spent() == accountant.get_privacy_spent()

    with pytest.raises(ValueError, match="does not match"):
        RdpAccountant(_config(population_size=2_000)).load_state_dict(state)


def test_rejects_unsafe_accounting_inputs() -> None:
    """Reject inputs that would invalidate the reported guarantee."""
    with pytest.raises(ValueError, match="target_delta"):
        PrivacyConfig(
            target_delta=0.0,
            population_size=1_000,
            neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
            sampling_method=SamplingMethod.POISSON,
        )
    with pytest.raises(ValueError, match="Unsupported privacy accounting model"):
        _config(neighboring_relation=NeighboringRelation.REPLACE_ONE)
    with pytest.raises(ValueError, match="noise_multiplier"):
        _event(noise_multiplier=0.0)
    with pytest.raises(ValueError, match="population_size"):
        RdpAccountant(_config()).compose(_event(population_size=2_000))
    for orders in ((), (float("nan"),), (float("inf"),), (1.0,)):
        with pytest.raises(ValueError, match="RDP order"):
            RdpAccountant(_config(), orders)
