"""customagg: ServerApp. Selects FedAvg or TrustAwareFedAvg via run-config."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from customagg.strategy import TrustAwareFedAvg
from customagg.task import Net, load_centralized_dataset, test

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    strategy_name: str = context.run_config["strategy"]

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    if strategy_name == "trust":
        strategy = TrustAwareFedAvg(
            fraction_evaluate=fraction_evaluate,
            trust_beta=context.run_config["trust-beta"],
            trust_z=context.run_config["trust-z"],
        )
    else:
        strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    _ = result


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})
