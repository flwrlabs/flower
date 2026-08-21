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
"""Renyi differential privacy accountant."""

from math import isfinite
from typing import Any, cast

from . import (
    GaussianPrivacyEvent,
    NeighboringRelation,
    PrivacyConfig,
    PrivacySpent,
    SamplingMethod,
)

_STATE_SCHEMA_VERSION = 1
_ACCOUNTING_METHOD = "rdp"


class RdpAccountant:
    """Track cumulative privacy loss using Renyi differential privacy.

    This class wraps Google's ``dp-accounting`` package while exposing only
    Flower-owned configuration, event, result, and serialization types.
    """

    def __init__(
        self,
        config: PrivacyConfig,
        orders: tuple[float, ...] | None = None,
    ) -> None:
        self._config = config
        if orders is not None:
            orders = _validate_orders(orders)
        (
            self._dp_event,
            self._privacy_accountant,
            self._rdp_module,
        ) = _load_dp_accounting()
        self._backend = self._new_backend(orders)
        self._orders = tuple(float(order) for order in self._backend.orders)
        self._compositions: list[tuple[GaussianPrivacyEvent, int]] = []

    @property
    def config(self) -> PrivacyConfig:
        """Return the privacy configuration."""
        return self._config

    @property
    def num_releases(self) -> int:
        """Return the number of composed private releases."""
        return sum(count for _, count in self._compositions)

    def compose(self, event: GaussianPrivacyEvent, count: int = 1) -> None:
        """Compose one or more identical sampled Gaussian releases."""
        self._validate_event(event)
        _validate_count(count)
        backend_event = self._to_backend_event(event)
        if not self._backend.supports(backend_event):
            raise ValueError(
                "The configured RDP backend does not support this privacy event."
            )
        self._backend.compose(backend_event, count)
        self._compositions.append((event, count))

    def get_epsilon(self, delta: float) -> float:
        """Return cumulative epsilon at the supplied delta."""
        _validate_delta(delta)
        return float(self._backend.get_epsilon(delta))

    def get_delta(self, epsilon: float) -> float:
        """Return cumulative delta at the supplied epsilon."""
        _validate_epsilon(epsilon)
        return float(self._backend.get_delta(epsilon))

    def get_privacy_spent(self, delta: float | None = None) -> PrivacySpent:
        """Return cumulative privacy expenditure at delta."""
        target_delta = self._config.target_delta if delta is None else delta
        _validate_delta(target_delta)
        epsilon, optimal_order = self._backend.get_epsilon_and_optimal_order(
            target_delta
        )
        return PrivacySpent(
            epsilon=float(epsilon),
            delta=target_delta,
            num_releases=self.num_releases,
            accounting_method=_ACCOUNTING_METHOD,
            optimal_order=float(optimal_order),
        )

    def would_exceed(
        self,
        event: GaussianPrivacyEvent,
        max_epsilon: float | None = None,
        count: int = 1,
    ) -> bool:
        """Return whether composing an event would exceed an epsilon limit.

        This method does not mutate committed accountant state.
        """
        self._validate_event(event)
        _validate_count(count)
        epsilon_limit = self._config.max_epsilon
        if max_epsilon is not None:
            epsilon_limit = max_epsilon
        if epsilon_limit is None:
            raise ValueError(
                "max_epsilon must be supplied either in PrivacyConfig or to "
                "would_exceed()."
            )
        _validate_positive_epsilon("max_epsilon", epsilon_limit)

        projected = self._new_backend(self._orders)
        self._replay(projected)
        backend_event = self._to_backend_event(event)
        if not projected.supports(backend_event):
            raise ValueError(
                "The configured RDP backend does not support this privacy event."
            )
        projected.compose(backend_event, count)
        return bool(projected.get_epsilon(self._config.target_delta) > epsilon_limit)

    def state_dict(self) -> dict[str, object]:
        """Return JSON-serializable accountant state."""
        return {
            "schema_version": _STATE_SCHEMA_VERSION,
            "backend": _ACCOUNTING_METHOD,
            "config": {
                "target_delta": self._config.target_delta,
                "population_size": self._config.population_size,
                "neighboring_relation": self._config.neighboring_relation.value,
                "sampling_method": self._config.sampling_method.value,
                "max_epsilon": self._config.max_epsilon,
            },
            "orders": list(self._orders),
            "compositions": [
                {
                    "event": {
                        "noise_multiplier": event.noise_multiplier,
                        "sample_size": event.sample_size,
                        "population_size": event.population_size,
                    },
                    "count": count,
                }
                for event, count in self._compositions
            ],
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore state produced by :meth:`state_dict`.

        The restored state must use the same privacy configuration as this
        accountant. Backend numeric state is rebuilt from Flower's event ledger.
        """
        schema_version = state.get("schema_version")
        if schema_version != _STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported accountant state schema version: {schema_version!r}."
            )
        if state.get("backend") != _ACCOUNTING_METHOD:
            raise ValueError("Accountant state was produced by a different backend.")

        restored_config = _parse_config(state.get("config"))
        if restored_config != self._config:
            raise ValueError(
                "Accountant state privacy configuration does not match the current "
                "configuration."
            )

        orders = _parse_orders(state.get("orders"))
        compositions = _parse_compositions(state.get("compositions"))

        restored_backend = self._new_backend(orders)
        for event, count in compositions:
            self._validate_event(event)
            restored_backend.compose(self._to_backend_event(event), count)

        self._backend = restored_backend
        self._orders = orders
        self._compositions = compositions

    def _new_backend(self, orders: tuple[float, ...] | None) -> Any:
        relation = getattr(
            self._privacy_accountant.NeighboringRelation,
            self._config.neighboring_relation.name,
        )
        return self._rdp_module.RdpAccountant(
            orders=orders,
            neighboring_relation=relation,
        )

    def _replay(self, backend: Any) -> None:
        for event, count in self._compositions:
            backend.compose(self._to_backend_event(event), count)

    def _to_backend_event(self, event: GaussianPrivacyEvent) -> Any:
        gaussian_event = self._dp_event.GaussianDpEvent(event.noise_multiplier)
        if self._config.sampling_method is SamplingMethod.POISSON:
            return self._dp_event.PoissonSampledDpEvent(
                event.sample_size / event.population_size,
                gaussian_event,
            )
        if self._config.sampling_method is SamplingMethod.WITHOUT_REPLACEMENT:
            return self._dp_event.SampledWithoutReplacementDpEvent(
                event.population_size,
                event.sample_size,
                gaussian_event,
            )
        if self._config.sampling_method is SamplingMethod.NO_AMPLIFICATION:
            return gaussian_event
        raise ValueError(
            f"Unsupported sampling method: {self._config.sampling_method}."
        )

    def _validate_event(self, event: GaussianPrivacyEvent) -> None:
        if event.population_size != self._config.population_size:
            raise ValueError(
                "Privacy event population_size does not match PrivacyConfig: "
                f"{event.population_size} != {self._config.population_size}."
            )


def _load_dp_accounting() -> tuple[Any, Any, Any]:
    """Load the optional privacy-accounting backend."""
    # pylint: disable=import-outside-toplevel
    try:
        from dp_accounting import dp_event, privacy_accountant
        from dp_accounting import rdp as rdp_module
    except ImportError as exc:
        raise ImportError(
            'RDP accounting requires the optional dependency: pip install "flwr[dp]".'
        ) from exc
    # pylint: enable=import-outside-toplevel
    return dp_event, privacy_accountant, rdp_module


def _parse_config(value: object) -> PrivacyConfig:
    config = _require_dict(value, "config")
    try:
        max_epsilon_value = config["max_epsilon"]
        return PrivacyConfig(
            target_delta=_require_float(config["target_delta"], "target_delta"),
            population_size=_require_int(config["population_size"], "population_size"),
            neighboring_relation=NeighboringRelation(
                _require_str(config["neighboring_relation"], "neighboring_relation")
            ),
            sampling_method=SamplingMethod(
                _require_str(config["sampling_method"], "sampling_method")
            ),
            max_epsilon=(
                None
                if max_epsilon_value is None
                else _require_float(max_epsilon_value, "max_epsilon")
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid privacy configuration in accountant state.") from exc


def _parse_orders(value: object) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise ValueError("Accountant state orders must be a list.")
    try:
        orders = tuple(float(order) for order in value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Accountant state contains invalid RDP orders.") from exc
    return _validate_orders(orders)


def _validate_orders(orders: tuple[float, ...]) -> tuple[float, ...]:
    if not orders:
        raise ValueError("RDP orders must be non-empty.")
    if any(not isfinite(order) or order <= 1 for order in orders):
        raise ValueError("Every RDP order must be finite and greater than 1.")
    return orders


def _parse_compositions(value: object) -> list[tuple[GaussianPrivacyEvent, int]]:
    if not isinstance(value, list):
        raise ValueError("Accountant state compositions must be a list.")
    compositions: list[tuple[GaussianPrivacyEvent, int]] = []
    try:
        for composition_value in value:
            composition = _require_dict(composition_value, "composition")
            event_value = _require_dict(composition["event"], "event")
            event = GaussianPrivacyEvent(
                noise_multiplier=_require_float(
                    event_value["noise_multiplier"], "noise_multiplier"
                ),
                sample_size=_require_int(event_value["sample_size"], "sample_size"),
                population_size=_require_int(
                    event_value["population_size"], "population_size"
                ),
            )
            count = _require_int(composition["count"], "count")
            _validate_count(count)
            compositions.append((event, count))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid composition in accountant state.") from exc
    return compositions


def _require_dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Accountant state {name} must be a dictionary.")
    return cast(dict[str, object], value)


def _require_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Accountant state {name} must be an integer.")
    return value


def _require_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"Accountant state {name} must be numeric.")
    return float(value)


def _require_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Accountant state {name} must be a string.")
    return value


def _validate_count(count: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("count must be a positive integer.")


def _validate_delta(delta: float) -> None:
    if not isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must be finite and strictly between 0 and 1.")


def _validate_epsilon(epsilon: float) -> None:
    if isinstance(epsilon, bool) or not isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be non-negative and finite.")


def _validate_positive_epsilon(name: str, epsilon: float) -> None:
    if isinstance(epsilon, bool) or not isfinite(epsilon) or epsilon <= 0:
        raise ValueError(f"{name} must be positive and finite.")
