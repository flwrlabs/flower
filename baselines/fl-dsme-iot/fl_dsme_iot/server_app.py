"""fl_dsme_iot: ServerApp - PAN Coordinator with energy-aware aggregation."""

import torch
from collections.abc import Iterable
from flwr.app import ArrayRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fl_dsme_iot.model import Net

app = ServerApp()


class DSMEFedAvg(FedAvg):
    """FedAvg extended to handle energy-depleted rounds gracefully.

    When all sampled clients return num-examples=0 (all energy-depleted),
    standard FedAvg raises ZeroDivisionError in aggregate_arrayrecords.
    DSMEFedAvg detects this and keeps the previous global model unchanged,
    correctly modelling DSME behaviour: if no device has enough energy to
    transmit in this beacon interval, the PAN Coordinator waits for the next.
    """

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate train replies, skipping all-depleted rounds."""
        replies_list = list(replies)

        # Check total examples across all non-error replies
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
            # All clients energy-depleted: keep global model, skip aggregation.
            # This models a DSME beacon interval with no successful transmissions.
            print(
                f"\n[Round {server_round}] All sampled clients energy-depleted. "
                "Keeping current global model (DSME beacon interval skipped).\n"
            )
            return None, None

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
