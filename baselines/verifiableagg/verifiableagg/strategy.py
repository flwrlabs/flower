"""Custom FedAvg strategy with optional aggregation verification hooks."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from flwr.app import ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords


@dataclass
class VerificationRound:
    """Verification metadata for one aggregation round."""

    server_round: int
    num_replies: int
    max_abs_diff: float
    passed: bool
    aggregate_hash: str


def _update_with_length_prefix(digest: Any, data: bytes) -> None:
    """Update a hash digest using unambiguous length-prefixed framing."""
    digest.update(len(data).to_bytes(8, byteorder="little", signed=False))
    digest.update(data)


def _hash_arrayrecord(arrays: ArrayRecord) -> str:
    """Compute a deterministic SHA256 hash of an ArrayRecord."""
    digest = hashlib.sha256()
    for key in sorted(arrays.keys()):
        arr = np.ascontiguousarray(arrays[key].numpy())
        _update_with_length_prefix(digest, key.encode("utf-8"))
        _update_with_length_prefix(digest, str(arr.dtype).encode("utf-8"))
        _update_with_length_prefix(
            digest,
            np.asarray(arr.shape, dtype=np.int64).tobytes(),
        )
        _update_with_length_prefix(digest, arr.tobytes())
    return digest.hexdigest()


def _recompute_weighted_average(
    replies: list[RecordDict],
    weighted_by_key: str,
    arrayrecord_key: str,
    metricrecord_key: str,
) -> dict[str, np.ndarray]:
    """Recompute weighted average from client replies."""
    first_array_records = replies[0].array_records
    if arrayrecord_key in first_array_records:
        first_arrays = first_array_records[arrayrecord_key]
    else:
        first_arrays = next(iter(first_array_records.values()))
    keys = list(first_arrays.keys())

    sums: dict[str, np.ndarray] = {}
    total_weight = 0.0

    for reply in replies:
        metric_records = reply.metric_records
        if metricrecord_key in metric_records:
            metrics = metric_records[metricrecord_key]
        else:
            metrics = next(iter(metric_records.values()))

        array_records = reply.array_records
        if arrayrecord_key in array_records:
            arrays = array_records[arrayrecord_key]
        else:
            arrays = next(iter(array_records.values()))
        weight = float(cast(int | float, metrics[weighted_by_key]))
        total_weight += weight

        for key in keys:
            current = arrays[key].numpy().astype(np.float64)
            if key not in sums:
                sums[key] = np.zeros_like(current)
            sums[key] += weight * current

    if total_weight <= 0.0:
        return {key: np.zeros_like(value) for key, value in sums.items()}

    return {key: value / total_weight for key, value in sums.items()}


class VerifiableFedAvg(FedAvg):
    """FedAvg with optional deterministic post-aggregation verification."""

    def __init__(
        self,
        verify_aggregation: bool = True,
        verification_tolerance: float = 1e-6,
        metricrecord_key: str = "metrics",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.verify_aggregation = verify_aggregation
        self.verification_tolerance = verification_tolerance
        self.metricrecord_key = metricrecord_key
        self.verification_rounds: list[VerificationRound] = []

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate train replies and optionally verify deterministic replay."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            self.verification_rounds.append(
                VerificationRound(
                    server_round=server_round,
                    num_replies=0,
                    max_abs_diff=0.0,
                    passed=True,
                    aggregate_hash="",
                )
            )
            return None, None

        reply_contents = [msg.content for msg in valid_replies]

        arrays = aggregate_arrayrecords(
            reply_contents,
            self.weighted_by_key,
        )
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)

        max_abs_diff = 0.0
        verification_passed = True

        if self.verify_aggregation:
            replay_avg = _recompute_weighted_average(
                replies=reply_contents,
                weighted_by_key=self.weighted_by_key,
                arrayrecord_key=self.arrayrecord_key,
                metricrecord_key=self.metricrecord_key,
            )
            for key in replay_avg:
                agg_arr = arrays[key].numpy().astype(np.float64)
                diff = np.max(np.abs(replay_avg[key] - agg_arr))
                max_abs_diff = max(max_abs_diff, float(diff))
            verification_passed = max_abs_diff <= self.verification_tolerance

        aggregate_hash = _hash_arrayrecord(arrays)

        if metrics is None:
            metrics = MetricRecord()
        metrics["verification_passed"] = int(verification_passed)
        metrics["verification_max_abs_diff"] = float(max_abs_diff)
        metrics["verification_num_replies"] = int(len(valid_replies))

        self.verification_rounds.append(
            VerificationRound(
                server_round=server_round,
                num_replies=len(valid_replies),
                max_abs_diff=max_abs_diff,
                passed=verification_passed,
                aggregate_hash=aggregate_hash,
            )
        )

        return arrays, metrics
