"""Unit tests for decentralized run config helpers."""

from flwr.decentralized.common.run_config import DLRunConfig


def test_get_cycles_with_positive_aggregation_steps() -> None:
    """Compute finite cycle count when aggregation steps are enabled."""
    cfg = DLRunConfig(rounds=2, n_aggregation_steps=2)

    assert cfg.get_steps_per_round() == 3
    assert cfg.get_cycles() == 7


def test_get_cycles_disabled_aggregation_returns_infinite_marker() -> None:
    """Return -1 marker when no aggregation steps are configured."""
    cfg = DLRunConfig(rounds=5, n_aggregation_steps=0)

    assert cfg.get_steps_per_round() == 1
    assert cfg.get_cycles() == -1


def test_none_aggregation_steps_defaults_to_one() -> None:
    """Treat `None` aggregation steps as one effective step."""
    cfg = DLRunConfig(rounds=1, n_aggregation_steps=None)  # type: ignore[arg-type]

    assert cfg.get_steps_per_round() == 2
    assert cfg.get_cycles() == 3
