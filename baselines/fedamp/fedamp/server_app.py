"""FedAMP: Personalized Cross-Silo Federated Learning on Non-IID Data."""

import json
import os

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

from fedamp.model import FashionNet, Net
from fedamp.utils import get_strategy

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run entry point for the ServerApp."""
    # Read from config
    num_rounds: int = int(context.run_config["num-server-rounds"])
    fraction_train: float = float(context.run_config["fraction-train"])

    # Load global model
    dataset = context.run_config["dataset"]
    if dataset == "cifar10":
        global_model = Net(n_channels=3)
    elif dataset == "fashion":
        global_model = FashionNet(n_channels=1)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    arrays = ArrayRecord(global_model.state_dict())

    run_configurations = "\n".join(f"{k}: {v}" for k, v in context.run_config.items())
    print(f"Run Configuration:\n{run_configurations}")

    strategy = get_strategy(context=context)

    # Start strategy, run FedAMP for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # fedamp is PFL algorithm, so there is no global model to save

    train_rounds = sorted(result.train_metrics_clientapp.keys())
    train_losses = [
        result.train_metrics_clientapp[r]["train_loss"] for r in train_rounds
    ]

    eval_rounds = sorted(result.evaluate_metrics_clientapp.keys())
    eval_losses = [
        result.evaluate_metrics_clientapp[r]["eval_loss"] for r in eval_rounds
    ]
    eval_accs = [result.evaluate_metrics_clientapp[r]["eval_acc"] for r in eval_rounds]

    history = {
        "algorithm": context.run_config["algorithm"],
        "label": context.run_config["algorithm"],
        "train_rounds": train_rounds,
        "train_losses": train_losses,
        "eval_rounds": eval_rounds,
        "eval_accs": eval_accs,
        "eval_losses": eval_losses,
    }

    os.makedirs(f"{context.run_config['save-dir']}", exist_ok=True)

    with open(
        f"{context.run_config['save-dir']}/{context.run_config['algorithm']}_m_{fraction_train}.json",
        "w",
    ) as f:
        json.dump(history, f, indent=4)

    return
