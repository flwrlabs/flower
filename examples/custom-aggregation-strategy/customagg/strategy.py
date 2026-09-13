"""customagg: A minimal custom-aggregation Strategy for the Flower Message API.

TrustAwareFedAvg subclasses FedAvg and overrides ONLY `aggregate_train`. It computes
each client's update norm ||client_arrays - global_arrays||_2, then down-weights
clients whose update norm is far above the median (likely outliers / poisoned). The
trust score is folded into FedAvg's existing weighting key so the parent's weighted
aggregation is reused unchanged.
"""

from collections.abc import Iterable
from logging import INFO

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
)


class TrustAwareFedAvg(FedAvg):
    """FedAvg with norm-based trust weighting of client updates."""

    def __init__(
        self, *args, trust_beta: float = 5.0, trust_z: float = 3.5, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        # trust_beta: decay rate applied past the dead zone.
        # trust_z: width of the "normal" dead zone, in robust std units (MAD).
        #   Default 3.5 follows the Iglewicz-Hoaglin modified z-score outlier rule.
        self.trust_beta = trust_beta
        self.trust_z = trust_z
        self._global: ArrayRecord | None = None
        # Aggregation weight derived from trust; kept separate so the client-reported
        # `num-examples` metric is never overwritten.
        self._effective_weight_key = "trust-weighted-num-examples"

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        # Stash current global model so aggregate_train can compute the *update* norm
        # (the base API only passes replies, not the global model, to aggregate_train).
        self._global = arrays
        return super().configure_train(server_round, arrays, config, grid)

    def _update_norm(self, content) -> float:
        """L2 norm of (client_arrays - global_arrays)."""
        client_ar = next(iter(content.array_records.values()))
        sq = 0.0
        for key, val in client_ar.items():
            diff = val.numpy() - self._global[key].numpy()
            sq += float(np.square(diff).sum())
        return float(np.sqrt(sq))

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None
        contents = [m.content for m in valid_replies]

        # 1) per-client update norm
        norms = np.array([self._update_norm(c) for c in contents])

        # 2) robust outlier score via median + MAD (assumes an honest majority).
        #    A "dead zone" of `trust_z` robust-std keeps normal clients at trust=1.0;
        #    only updates beyond it are down-weighted, decaying with `trust_beta`.
        median = float(np.median(norms))
        scaled_mad = 1.4826 * float(np.median(np.abs(norms - median)))
        if scaled_mad < 1e-9:
            # Degenerate spread (norms nearly identical): trust everyone equally.
            trust = np.ones_like(norms)
        else:
            z = (norms - median) / scaled_mad
            excess = np.maximum(0.0, z - self.trust_z)
            trust = np.exp(-self.trust_beta * excess)

        # 3) Aggregate metrics first using the original (data-size) weights, then store a
        #    SEPARATE effective weight for model aggregation. This never overwrites the
        #    client-reported `num-examples`, which stays intact for downstream use.
        metrics = self.train_metrics_aggr_fn(contents, self.weighted_by_key)
        for c, t in zip(contents, trust):
            mr = next(iter(c.metric_records.values()))
            mr[self._effective_weight_key] = float(mr[self.weighted_by_key]) * float(t)

        # Transparency: show what got down-weighted
        flagged = [
            (i, float(n), float(tr))
            for i, (n, tr) in enumerate(zip(norms, trust))
            if tr < 0.5
        ]
        log(
            INFO,
            "\t└──> TrustAware: median_norm=%.3f mad=%.3f | down-weighted %d/%d clients %s",
            median,
            scaled_mad,
            len(flagged),
            len(contents),
            [f"#{i}(norm={n:.1f},trust={tr:.2f})" for i, n, tr in flagged],
        )

        arrays = aggregate_arrayrecords(contents, self._effective_weight_key)
        return arrays, metrics
