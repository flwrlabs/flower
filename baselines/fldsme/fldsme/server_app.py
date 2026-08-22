"""fldsme: ServerApp - PAN Coordinator with energy-aware aggregation."""

import json
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fldsme.model import Net

app = ServerApp()


def save_result_json(path, strategy, run_config=None) -> None:
    """Write per-round metrics to JSON so a sweep can be plotted later.

    Writes the strategy's own recorded history rather than introspecting the
    Flower Result object: field names move between releases, and the weight
    ArrayRecord is large enough to make the dump useless as a metrics file.
    """
    if not path:
        return

    payload = {
        "rounds": {
            phase: {str(r): m for r, m in sorted(rounds.items())}
            for phase, rounds in strategy.history.items()
        }
    }
    if run_config is not None:
        payload["run_config"] = {
            str(k): (v if isinstance(v, (bool, int, float, str)) else str(v))
            for k, v in run_config.items()
        }

    out = Path(str(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    n_train = len(strategy.history.get("train", {}))
    n_eval = len(strategy.history.get("evaluate", {}))
    size_kb = out.stat().st_size / 1024
    print(
        f"Wrote results to {out} "
        f"({n_train} train rounds, {n_eval} evaluate rounds, {size_kb:.1f} KB)"
    )


def dsme_metrics_aggregation(
    reply_contents: list,
    weighted_by_key: str,
) -> MetricRecord:
    """Custom metric aggregation that correctly accounts for skipped clients.

    Standard FedAvg weights metrics by num-examples, meaning skipped clients
    (num-examples=0) contribute zero weight and are invisible in aggregated
    metrics. This function separately tracks:
      - active clients: weighted average of train_loss, energy_used_mj,
        bandwidth_frac, gts_slots, residual_mj by num-examples
      - skipped clients: simple count reported as skipped_clients
      - total sampled: active + skipped

    This gives accurate per-round reporting of DSME MAC-layer behaviour.
    """
    active, skipped = [], []
    for rc in reply_contents:
        n = int(rc.metric_records["metrics"].get("num-examples", 0))
        if n > 0:
            active.append(rc)
        else:
            skipped.append(rc)

    n_active  = len(active)
    n_skipped = len(skipped)
    n_total   = n_active + n_skipped

    if n_active == 0:
        # All depleted - return summary metrics only
        return MetricRecord({
            "train_loss":      0.0,
            "energy_used_mj":  0.0,
            "bandwidth_frac":  0.0,
            "gts_slots":       0.0,
            "residual_mj":     0.0,
            "ncr_fraction":    0.0,
            "active_clients":  float(n_active),
            "skipped_clients": float(n_skipped),
            "total_sampled":   float(n_total),
        })

    # Weighted average over active clients only
    total_examples = sum(
        int(rc.metric_records["metrics"].get("num-examples", 0))
        for rc in active
    )

    def weighted_avg(key: str) -> float:
        return sum(
            float(rc.metric_records["metrics"].get(key, 0.0))
            * int(rc.metric_records["metrics"].get("num-examples", 0))
            for rc in active
        ) / total_examples

    return MetricRecord({
        "train_loss":      weighted_avg("train_loss"),
        "energy_used_mj":  weighted_avg("energy_used_mj"),
        "bandwidth_frac":  weighted_avg("bandwidth_frac"),
        "gts_slots":       weighted_avg("gts_slots"),
        "residual_mj":     weighted_avg("residual_mj"),
        # Fraction of active clients running in SeCAP NCR mode this round.
        # Once budgets fall below the CR-mode cost this is the only reason any
        # client remains eligible at all, so it belongs in the round record.
        "ncr_fraction":    sum(
            float(rc.metric_records["metrics"].get("cap_ncr", 0.0))
            for rc in active
        ) / n_active,
        "active_clients":  float(n_active),
        "skipped_clients": float(n_skipped),
        "total_sampled":   float(n_total),
    })


class DSMEFedAvg(FedAvg):
    """FedAvg extended for DSME MAC-layer aware aggregation.

    Two key additions over standard FedAvg:

    1. Handles all-depleted rounds: when every sampled client returns
       num-examples=0, standard FedAvg raises ZeroDivisionError.
       DSMEFedAvg detects this and holds the global model unchanged,
       modelling a DSME beacon interval with no successful transmissions.

    2. Correct metric aggregation: uses dsme_metrics_aggregation to
       separately count skipped vs active clients per round, so the
       server logs accurate DSME participation statistics.
    """

    def __init__(self, **kwargs):
        super().__init__(
            train_metrics_aggr_fn=dsme_metrics_aggregation,
            **kwargs,
        )
        # Per-round history recorded as we go. Introspecting the Result object
        # afterwards is fragile (fields move between Flower releases, and the
        # weight ArrayRecord is easy to drag in by accident, which produced a
        # 1.7 MB dump). Recording here is deterministic and small.
        self.history: dict[str, dict[int, dict[str, float]]] = {
            "train": {},
            "evaluate": {},
        }

    def _record(self, phase: str, server_round: int, metrics) -> None:
        """Store one round's aggregated metrics as plain floats."""
        if metrics is None:
            return
        try:
            self.history[phase][int(server_round)] = {
                str(k): float(v) for k, v in metrics.items()
            }
        except Exception:  # pragma: no cover - never break a run over logging
            pass

    def aggregate_evaluate(self, server_round: int, replies):
        """Record evaluate metrics, then return exactly what FedAvg returned.

        The return value is passed through untouched. In Flower 1.31 this hook
        returns a single MetricRecord, not a (loss, metrics) tuple; unpacking it
        blindly raises ValueError. Handle both shapes and change neither.
        """
        result = super().aggregate_evaluate(server_round, replies)
        metrics = result[-1] if isinstance(result, tuple) else result
        self._record("evaluate", server_round, metrics)
        return result

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate train replies, skipping all-depleted rounds."""
        replies_list = list(replies)

        # Separate valid replies from error replies first.
        # Only valid replies (no error, metric_records accessible) count
        # toward the DSME energy-depletion check. Error replies are passed
        # through to FedAvg's _check_and_log_replies for proper logging.
        valid_replies = []
        error_replies = []
        for msg in replies_list:
            if msg.has_error():
                error_replies.append(msg)
                continue
            try:
                _ = msg.content.metric_records["metrics"]
                valid_replies.append(msg)
            except (KeyError, AttributeError):
                error_replies.append(msg)

        # Only enter the DSME all-depleted path when there is at least one
        # valid reply (i.e. a client successfully reported num-examples=0).
        # If all replies are errors, fall through to FedAvg which will log
        # the failures correctly via _check_and_log_replies.
        if valid_replies:
            total_examples = sum(
                int(msg.content.metric_records["metrics"].get("num-examples", 0))
                for msg in valid_replies
            )

            if total_examples == 0:
                # All valid clients energy-depleted: hold global model unchanged.
                # Models a DSME beacon interval with no successful transmissions.
                # Log any error replies explicitly before returning so that
                # failures are never silently swallowed by the depleted path.
                n_valid = len(valid_replies)
                if error_replies:
                    from flwr.common.logger import log
                    from logging import WARNING
                    log(
                        WARNING,
                        "[Round %d] %d error reply/replies received alongside "
                        "%d energy-depleted reply/replies. Logging failures "
                        "before recording as all-depleted DSME round.",
                        server_round,
                        len(error_replies),
                        n_valid,
                    )
                    for err_msg in error_replies:
                        log(WARNING, "  Failed reply: %s", err_msg.error)
                print(
                    f"\n[Round {server_round}] All {n_valid} valid clients "
                    "energy-depleted. Keeping current global model "
                    "(DSME beacon interval skipped).\n"
                )
                depleted_metrics = MetricRecord({
                    "train_loss":      0.0,
                    "energy_used_mj":  0.0,
                    "bandwidth_frac":  0.0,
                    "gts_slots":       0.0,
                    "residual_mj":     0.0,
                    "ncr_fraction":    0.0,
                    "active_clients":  0.0,
                    "skipped_clients": float(n_valid),
                    "total_sampled":   float(len(replies_list)),
                    "error_replies":   float(len(error_replies)),
                })
                self._record("train", server_round, depleted_metrics)
                return None, depleted_metrics

        arrays, metrics = super().aggregate_train(server_round, iter(replies_list))
        self._record("train", server_round, metrics)
        return arrays, metrics


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run entry point for the ServerApp."""
    num_rounds: int = int(context.run_config["num-server-rounds"])
    fraction_train: float = float(context.run_config["fraction-train"])
    seed: int = int(context.run_config.get("seed", 0))

    # Seed the global model initialisation so a multi-seed sweep actually
    # varies, and so a fixed seed is reproducible across arms.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    global_model = Net()
    initial_state_dict = global_model.state_dict()   # keep reference for fallback
    arrays = ArrayRecord(initial_state_dict)

    strategy = DSMEFedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=1.0,
        min_available_nodes=2,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    print("\nSaving final model to disk...")
    # result.arrays may be empty if every train round was skipped
    # (all clients energy-depleted every round). Fall back to the
    # initial global model so the saved file is always a valid model.
    if result.arrays is not None and len(result.arrays) > 0:
        state_dict = result.arrays.to_torch_state_dict()
    else:
        print(
            "Warning: all training rounds were skipped. "
            "Saving initial global model as final_model.pt."
        )
        state_dict = initial_state_dict
    torch.save(state_dict, "final_model.pt")

    save_result_json(
        context.run_config.get("results-path", ""),
        strategy,
        context.run_config,
    )
