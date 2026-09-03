"""floco: A Flower Baseline."""

import json
import os
import time

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from .dataset import get_testloader
from .model import SimplexModel, create_model, test
from .strategy import Floco

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run entry point for the ServerApp."""
    num_rounds = int(context.run_config["num-server-rounds"])
    strategy = get_strategy(context)
    initial_arrays = ArrayRecord(create_model(context).state_dict())
    evaluate_fn = make_evaluate_fn(context, strategy)

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        evaluate_fn=evaluate_fn,
    )

    save_results(context, result)


def get_strategy(context: Context) -> FedAvg | Floco:
    """Get the strategy based on run config."""
    fraction_train = float(context.run_config["fraction-train"])

    if context.run_config["algorithm"] == "FedAvg":
        return FedAvg(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=2,
        )
    if context.run_config["algorithm"] == "Floco":
        tau = int(context.run_config["tau"])
        rho = float(context.run_config["rho"])
        endpoints = int(context.run_config["endpoints"])
        return Floco(
            fraction_train=fraction_train,
            fraction_evaluate=1.0,
            min_available_nodes=2,
            tau=tau,
            rho=rho,
            endpoints=endpoints,
        )
    raise ValueError("Algorithm not implemented")


def make_evaluate_fn(context: Context, strategy):
    """Create a centralized evaluation closure."""
    dataset = str(context.run_config["dataset"])

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
        server_model = create_model(context)
        server_model.load_state_dict(arrays.to_torch_state_dict())
        if isinstance(server_model, SimplexModel):
            server_model.training = False
            center = np.array(
                [1 / strategy.endpoints for _ in range(strategy.endpoints)]
            )
            server_model.subregion_parameters = (center, strategy.rho)

        testloader = get_testloader(dataset)
        loss, accuracy = test(server_model, testloader, DEVICE)
        return MetricRecord(
            {"centralized_loss": loss, "centralized_accuracy": accuracy}
        )

    return evaluate_fn


def save_results(context: Context, result) -> None:
    """Write all results from the Result object to a single JSON file."""
    res_save_path = "./results"
    if not os.path.exists(res_save_path):
        os.makedirs(res_save_path)
    res_save_name = time.strftime("%Y-%m-%d-%H-%M-%S")
    filepath = f"{res_save_path}/result-{res_save_name}.json"

    round_res = []
    all_rounds = sorted(
        set(result.evaluate_metrics_clientapp.keys())
        | set(result.evaluate_metrics_serverapp.keys())
    )
    for rnd in all_rounds:
        entry = {}
        if rnd in result.evaluate_metrics_serverapp:
            for k, v in result.evaluate_metrics_serverapp[rnd].items():
                entry[k] = v
        if rnd in result.evaluate_metrics_clientapp:
            for k, v in result.evaluate_metrics_clientapp[rnd].items():
                entry[k] = v
        round_res.append(entry)

    data = {
        "run_config": dict(context.run_config.items()),
        "round_res": round_res,
    }
    with open(filepath, "w", encoding="UTF-8") as fout:
        json.dump(data, fout, indent=4)
