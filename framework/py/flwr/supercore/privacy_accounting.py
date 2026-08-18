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
"""Backend-independent differential privacy accounting types."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol


class NeighboringRelation(str, Enum):
    """Define which client-level datasets are considered neighbors."""

    ADD_OR_REMOVE_ONE = "add-or-remove-one"
    REPLACE_ONE = "replace-one"


class SamplingMethod(str, Enum):
    """Define how clients are sampled for a private release."""

    POISSON = "poisson"
    WITHOUT_REPLACEMENT = "without-replacement"
    NO_AMPLIFICATION = "no-amplification"


@dataclass(frozen=True)
class PrivacyConfig:
    """Configure cumulative client-level privacy accounting.

    Parameters
    ----------
    target_delta : float
        Default delta used when reporting the cumulative epsilon.
    population_size : int
        Size of the stable population from which clients are sampled.
    neighboring_relation : NeighboringRelation
        Relation used to define neighboring client populations.
    sampling_method : SamplingMethod
        Sampling model used by every composed privacy event.
    max_epsilon : float or None
        Optional privacy-budget limit used for prospective checks.
    """

    target_delta: float
    population_size: int
    neighboring_relation: NeighboringRelation
    sampling_method: SamplingMethod
    max_epsilon: float | None = None

    def __post_init__(self) -> None:
        """Validate the privacy configuration."""
        _validate_probability("target_delta", self.target_delta)
        _validate_positive_int("population_size", self.population_size)
        if self.max_epsilon is not None:
            _validate_positive_finite("max_epsilon", self.max_epsilon)
        _validate_accounting_model(
            self.sampling_method,
            self.neighboring_relation,
        )


@dataclass(frozen=True)
class GaussianPrivacyEvent:
    """Describe one or more identical sampled Gaussian releases.

    For Poisson sampling, ``sample_size / population_size`` is the expected
    sampling probability. For sampling without replacement, ``sample_size`` is
    the exact sample size. With no amplification, the sampling metadata is
    retained for reporting but does not reduce the accounted privacy cost.
    """

    noise_multiplier: float
    sample_size: int
    population_size: int

    def __post_init__(self) -> None:
        """Validate the Gaussian privacy event."""
        _validate_positive_finite("noise_multiplier", self.noise_multiplier)
        _validate_positive_int("sample_size", self.sample_size)
        _validate_positive_int("population_size", self.population_size)
        if self.sample_size > self.population_size:
            raise ValueError("sample_size must not exceed population_size.")


@dataclass(frozen=True)
class PrivacySpent:
    """Represent cumulative privacy expenditure."""

    epsilon: float
    delta: float
    num_releases: int
    accounting_method: str
    optimal_order: float | None = None


class PrivacyAccountant(Protocol):
    """Protocol implemented by privacy-accounting backends."""

    @property
    def config(self) -> PrivacyConfig:
        """Return the privacy configuration."""

    @property
    def num_releases(self) -> int:
        """Return the number of composed private releases."""

    def compose(self, event: GaussianPrivacyEvent, count: int = 1) -> None:
        """Compose one or more identical privacy events."""

    def get_epsilon(self, delta: float) -> float:
        """Return cumulative epsilon at the supplied delta."""

    def get_delta(self, epsilon: float) -> float:
        """Return cumulative delta at the supplied epsilon."""

    def get_privacy_spent(self, delta: float | None = None) -> PrivacySpent:
        """Return cumulative privacy expenditure."""

    def would_exceed(
        self,
        event: GaussianPrivacyEvent,
        max_epsilon: float | None = None,
        count: int = 1,
    ) -> bool:
        """Return whether composing an event would exceed an epsilon limit."""

    def state_dict(self) -> dict[str, object]:
        """Return serializable accountant state."""

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore serializable accountant state."""


def _validate_accounting_model(
    sampling_method: SamplingMethod,
    neighboring_relation: NeighboringRelation,
) -> None:
    """Validate sampling and neighboring-relation compatibility."""
    supported_models = {
        (
            SamplingMethod.POISSON,
            NeighboringRelation.ADD_OR_REMOVE_ONE,
        ),
        (
            SamplingMethod.WITHOUT_REPLACEMENT,
            NeighboringRelation.REPLACE_ONE,
        ),
        (
            SamplingMethod.NO_AMPLIFICATION,
            NeighboringRelation.ADD_OR_REMOVE_ONE,
        ),
    }
    if (sampling_method, neighboring_relation) not in supported_models:
        raise ValueError(
            "Unsupported privacy accounting model: "
            f"sampling_method='{sampling_method.value}' with "
            f"neighboring_relation='{neighboring_relation.value}'."
        )


def _validate_probability(name: str, value: float) -> None:
    """Validate that a value is a probability strictly between zero and one."""
    if isinstance(value, bool) or not isfinite(value) or not 0 < value < 1:
        raise ValueError(f"{name} must be finite and strictly between 0 and 1.")


def _validate_positive_finite(name: str, value: float) -> None:
    """Validate that a value is positive and finite."""
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")


def _validate_positive_int(name: str, value: int) -> None:
    """Validate that a value is a positive integer and not a boolean."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
