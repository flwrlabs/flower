"""Flower ServerApp for the C++ quickstart example."""

import flwr as fl
import numpy as np
from flwr.common import Context

from fedavg_cpp import FedAvgCpp, ndarrays_to_parameters


MODEL_SIZE = 2


def fit_config(server_round: int) -> dict[str, fl.common.Scalar]:
    return {
        "server_round": server_round,
        "local_epochs": 1,
        "learning_rate": 0.01,
    }


def server_fn(context: Context) -> fl.server.ServerAppComponents:
    initial_weights = [
        np.zeros(MODEL_SIZE, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    ]
    strategy = FedAvgCpp(
        initial_parameters=ndarrays_to_parameters(initial_weights),
        on_fit_config_fn=fit_config,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    return fl.server.ServerAppComponents(
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )


app = fl.server.ServerApp(server_fn=server_fn)
