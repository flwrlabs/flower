"""Unit tests for the FedAgain trust computation and aggregation."""

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, ConfigRecord, MetricRecord, RecordDict

from fedagain.strategy import (
    FedAgain,
    floating_keys,
    parameter_divergence,
    trust_weights,
    weighted_average,
)


def test_trust_weights_match_equations():
    errors, divs, eps = [0.25, 0.5], [0.8, 0.6], 1e-3
    weights, raw = trust_weights(errors, divs, eps)
    expected_raw = 1.0 / (np.array(errors) * np.array(divs) + eps)
    np.testing.assert_allclose(raw, expected_raw)
    np.testing.assert_allclose(weights, expected_raw / expected_raw.sum())
    assert weights.sum() == pytest.approx(1.0)
    # Example of Sec. 3.4 of the paper: client k weighs ~1.5x client k'
    assert weights[0] / weights[1] == pytest.approx(0.3 / 0.2, rel=1e-2)


def test_trust_weights_are_scale_invariant():
    errors, divs = np.array([0.3, 1.2, 0.7]), np.array([2.0, 5.0, 3.0])
    w1, _ = trust_weights(errors, divs, 1e-12)
    w2, _ = trust_weights(10 * errors, 0.1 * divs, 1e-12)
    np.testing.assert_allclose(w1, w2, rtol=1e-9)


def test_trust_weights_uniform_fallback():
    weights, _ = trust_weights([np.nan, 1.0], [1.0, 1.0], 1e-3)
    np.testing.assert_allclose(weights, [0.5, 0.5])


def test_divergence_ignores_integer_buffers():
    global_arrays = {
        "w": np.zeros(3, dtype=np.float32),
        "bn.num_batches_tracked": np.array(0, dtype=np.int64),
    }
    client = {
        "w": np.array([3.0, 4.0, 0.0], dtype=np.float32),
        "bn.num_batches_tracked": np.array(1000, dtype=np.int64),
    }
    keys = floating_keys(global_arrays)
    assert keys == ["w"]
    assert parameter_divergence(client, global_arrays, keys) == pytest.approx(5.0)


def test_weighted_average_keeps_dtypes():
    ref = {"w": np.zeros(2, np.float32), "n": np.array(0, np.int64)}
    clients = [
        {"w": np.array([1.0, 2.0], np.float32), "n": np.array(10, np.int64)},
        {"w": np.array([3.0, 4.0], np.float32), "n": np.array(13, np.int64)},
    ]
    out = weighted_average(clients, [0.25, 0.75], ref)
    np.testing.assert_allclose(out["w"], [2.5, 3.5])
    assert out["w"].dtype == np.float32
    assert out["n"].dtype == np.int64 and int(out["n"]) == 12


class _FakeReply:
    """Minimal stand-in for a reply ``Message`` (content + source node)."""

    def __init__(self, arrays: dict, metrics: dict, node_id: int) -> None:
        self.content = RecordDict(
            {
                "arrays": ArrayRecord({k: Array(v) for k, v in arrays.items()}),
                "metrics": MetricRecord(metrics),
            }
        )
        self.metadata = type("Metadata", (), {"src_node_id": node_id})()

    def has_error(self) -> bool:
        return False


def _reply(arrays: dict, metrics: dict, node_id: int) -> _FakeReply:
    return _FakeReply(arrays, metrics, node_id)


class _FakeGrid:
    def get_node_ids(self):
        return [1, 2, 3]


def test_fedagain_downweights_high_error_client():
    strategy = FedAgain(epsilon=1e-6, min_available_nodes=2)
    global_arrays = {"w": np.zeros(4, np.float32)}
    strategy._global_arrays = global_arrays  # as set by configure_train

    # Two clients agree (low E, small update); one has a high benchmark error.
    replies = [
        _reply(
            {"w": np.full(4, 0.1, np.float32)},
            {"benchmark-error": 0.2, "num-examples": 100, "partition-id": 0},
            1,
        ),
        _reply(
            {"w": np.full(4, 0.1, np.float32)},
            {"benchmark-error": 0.2, "num-examples": 100, "partition-id": 1},
            2,
        ),
        _reply(
            {"w": np.full(4, 0.1, np.float32)},
            {"benchmark-error": 2.0, "num-examples": 1000, "partition-id": 2},
            3,
        ),
    ]
    arrays, metrics = strategy.aggregate_train(1, replies)
    weights = strategy.trust_history[-1]["weight"]
    # Same divergence -> weights proportional to 1/E: 10:10:1
    np.testing.assert_allclose(weights, np.array([10, 10, 1]) / 21, rtol=1e-4)
    assert arrays is not None and metrics is not None
    np.testing.assert_allclose(arrays["w"].numpy(), np.full(4, 0.1), rtol=1e-6)
    assert metrics["trust-weight-min"] == pytest.approx(1 / 21, rel=1e-4)


def test_fedagain_requires_benchmark_error():
    strategy = FedAgain(min_available_nodes=2)
    strategy._global_arrays = {"w": np.zeros(2, np.float32)}
    replies = [
        _reply({"w": np.ones(2, np.float32)}, {"num-examples": 10}, 1),
        _reply({"w": np.ones(2, np.float32)}, {"num-examples": 10}, 2),
    ]
    with pytest.raises(ValueError, match="benchmark-error"):
        strategy.aggregate_train(1, replies)


def test_configure_train_requests_benchmark_error(monkeypatch):
    strategy = FedAgain(min_available_nodes=2)
    sent = {}

    def fake_construct(record, node_ids, message_type):
        sent.update(record=record, node_ids=node_ids, message_type=message_type)
        return []

    monkeypatch.setattr(strategy, "_construct_messages", fake_construct)
    arrays = ArrayRecord({"w": Array(np.ones(2, np.float32))})
    strategy.configure_train(1, arrays, ConfigRecord(), _FakeGrid())
    assert sorted(sent["node_ids"]) == [1, 2, 3]
    assert sent["record"].config_records["config"]["compute-benchmark-error"]
    assert strategy._global_arrays is not None
    np.testing.assert_array_equal(strategy._global_arrays["w"], np.ones(2))
