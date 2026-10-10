"""FedSaSync: Semi-asynchronous Federated Learning in Flower."""

import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

from fedsasync.model import Net

from .strategy import FedSaSync
from .utils import train_metrics_aggr_fn

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run entry point for the ServerApp."""
    # Read from config
    num_rounds: int = int(context.run_config["num-server-rounds"])
    fraction_train: float = float(context.run_config["fraction-train"])
    fraction_evaluate: float = float(context.run_config["fraction-evaluate"])
    strategy_name: str = str(context.run_config["name"])
    semiasync_deg: int = int(context.run_config["semiasync-deg"])
    dataset_name: str = str(context.run_config["dataset-name"])
    data_distribution: str = str(context.run_config["data-distribution"])
    number_slow: int = int(context.run_config["number-slow"])
    polling_interval: float = float(context.run_config["polling-interval"])
    run_id: int = int(context.run_config["run-id"])
    proximal_mu: float = float(context.run_config["proximal-mu"])

    # Load global model
    torch.manual_seed(42 + run_id)
    if dataset_name == "uoft-cs/cifar10":
        global_model = Net()
    elif dataset_name == "ylecun/mnist":
        global_model = Net(1, 4)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedSaSync strategy
    strategy = FedSaSync(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_available_nodes=2,
        strategy_name=strategy_name,
        semiasync_deg=semiasync_deg,
        number_slow=number_slow,
        dataset_name=dataset_name,
        data_distribution=data_distribution,
        train_metrics_aggr_fn=train_metrics_aggr_fn,
        run_id=run_id,
        proximal_mu=proximal_mu
    )

    # Start strategy, run FedSaSync for `num_rounds`
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
        polling_interval=float(polling_interval)
    )
