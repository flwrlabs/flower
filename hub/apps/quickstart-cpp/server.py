import flwr as fl
import numpy as np

from fedavg_cpp import FedAvgCpp, weights_to_parameters

model_size = 2
initial_weights = [
    np.array([1.0, 2.0], dtype=np.float64),
    np.array([3.0], dtype=np.float64),
]
initial_parameters = weights_to_parameters(initial_weights)
strategy = FedAvgCpp(initial_parameters=initial_parameters)

app = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
