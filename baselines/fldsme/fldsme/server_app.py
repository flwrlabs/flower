"""fldsme: ServerApp - PAN Coordinator with energy-aware aggregation."""

import torch
from collections.abc import Iterable

from flwr.app import ArrayRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fldsme.model import Net

app = ServerApp()


def dsme_metrics_aggregation(
    reply_contents: list,
    weighted_by_key: str,
) -> MetricRecord:
    """Custom metric aggregation that correctly accounts for skipped clients.

    Standard FedAvg weights metrics by num-examples, meaning skipped clients
    (num-examples=0) contribute zero weight and are invisible in aggregated
    metrics. This function separately tracks:
      - active clients: weighted average of train_loss, energy_used_mj,
        bandwidth_frac, gts_slots by num-examples
      - skipped clients: simple count reported as skipped_count
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
        # All depleted — return summary metrics only
        return MetricRecord({
            "train_loss":      0.0,
            "energy_used_mj":  0.0,
            "bandwidth_frac":  0.0,
            "gts_slots":       0.0,
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

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate train replies, skipping all-depleted rounds."""
        replies_list = list(replies)

        # Count total examples across all non-error replies
        total_examples = 0
        for msg in replies_list:
            if not msg.has_error():
                try:
                    n = msg.content.metric_records["metrics"].get(
                        "num-examples", 0
                    )
                    total_examples += int(n)
                except (KeyError, AttributeError):
                    pass

        if total_examples == 0:
            # All clients energy-depleted: hold global model unchanged.
            # Models a DSME beacon interval with no successful transmissions.
            # Return None arrays (keep current model) but a MetricRecord so
            # skipped rounds still appear in experiment logs with correct
            # DSME participation statistics (fixes P2 bot review).
            n_sampled = sum(1 for msg in replies_list if not msg.has_error())
            print(
                f"\n[Round {server_round}] All sampled clients energy-depleted."
                " Keeping current global model (DSME beacon interval skipped).\n"
            )
            return None, MetricRecord({
                "train_loss":      0.0,
                "energy_used_mj":  0.0,
                "bandwidth_frac":  0.0,
                "gts_slots":       0.0,
                "active_clients":  0.0,
                "skipped_clients": float(n_sampled),
                "total_sampled":   float(n_sampled),
            })

        return super().aggregate_train(server_round, iter(replies_list))


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run entry point for the ServerApp."""
    num_rounds: int = int(context.run_config["num-server-rounds"])
    fraction_train: float = float(context.run_config["fraction-train"])

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

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
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
