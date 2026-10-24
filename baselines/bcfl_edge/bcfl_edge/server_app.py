"""bcfl_edge: Federated training on the devices selected for the blockchain."""

import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from bcfl_edge.dataset import load_test
from bcfl_edge.model import Net, test
from bcfl_edge.strategy import BCFL
from bcfl_edge.utils import save_results

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run entry point for the ServerApp."""
    cfg = context.run_config
    num_devices = int(cfg["num-devices"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testloader = load_test(num_devices, int(cfg["batch-size"]), str(cfg["dataset"]))

    def evaluate(_server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate the global model on the centralized test set."""
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        loss, accuracy, f1 = test(model, testloader, device)
        return MetricRecord({"loss": loss, "accuracy": accuracy, "f1": f1})

    strategy = BCFL(
        num_devices=num_devices,
        num_miners=int(cfg["num-miners"]),
        selected_devices=int(cfg["selected-devices"]),
        beta=float(cfg["beta"]),
        selection=str(cfg["selection"]),
        seed=int(cfg["seed"]),
        fraction_evaluate=0.0,  # the global model is evaluated centrally
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(Net().state_dict()),
        num_rounds=int(cfg["num-server-rounds"]),
        evaluate_fn=evaluate,
    )
    save_results(dict(cfg), strategy.round_cost, result.evaluate_metrics_serverapp)
