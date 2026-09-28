"""Metric aggregation helpers for SSFLStrategy."""

from __future__ import annotations

from flwr.app import MetricRecord, RecordDict


def aggregate_train_metrics(
    records: list[RecordDict], weighting_key: str
) -> MetricRecord:
    """
    Weighted mean for loss/LR; sum for traffic, FLOPs, and counts.

    Flower's default aggregator applies sample weighting to every metric.
    SSFL needs totals for communication and compute accounting.
    """
    if not records:
        return MetricRecord()

    weighted_keys = {"train_loss", "learning_rate", "sparsity_percent"}
    sum_keys = {
        "num-examples",
        "comm_params",
        "arrayrecord_payload_bytes",
        "message_bytes",
        "training_flops",
        "packed_numel",
    }

    totals: dict[str, float] = {}
    weighted_sums: dict[str, float] = {}
    weight_denom = 0.0

    for record in records:
        metrics = record["metrics"]
        if not isinstance(metrics, MetricRecord):
            raise TypeError("Expected a MetricRecord under the 'metrics' key")
        weight_value = metrics[weighting_key]
        if not isinstance(weight_value, (int, float)):
            raise TypeError(f"Weighting metric {weighting_key!r} must be scalar")
        weight = float(weight_value)
        weight_denom += weight
        for key, value in metrics.items():
            if not isinstance(value, (int, float)):
                raise TypeError(f"Metric {key!r} must be scalar")
            value_f = float(value)
            if key in sum_keys or key == weighting_key:
                totals[key] = totals.get(key, 0.0) + value_f
            elif key in weighted_keys:
                weighted_sums[key] = weighted_sums.get(key, 0.0) + value_f * weight
            else:
                # Default: sample-weighted mean for unknown scalar metrics
                weighted_sums[key] = weighted_sums.get(key, 0.0) + value_f * weight

    out: dict[str, int | float | list[int] | list[float]] = {}
    out.update(totals)
    if weight_denom > 0:
        for key, value in weighted_sums.items():
            out[key] = value / weight_denom
    return MetricRecord(out)
