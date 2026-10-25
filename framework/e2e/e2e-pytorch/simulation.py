from typing import List, Tuple

import numpy as np
from e2e_pytorch.client_app import client_fn

import flwr as fl
from flwr.common import Metrics

STATE_VAR = "timestamp"


# Define metric aggregation function
def record_state_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Ensure that timestamps are monotonically increasing."""
    states = []
    for _, m in metrics:
        # split string and covert timestamps to float
        states.append([float(tt) for tt in m[STATE_VAR].split(",")])

    for client_state in states:
        if len(client_state) == 1:
            continue
        deltas = np.diff(client_state)
        assert np.all(
            deltas > 0
        ), f"Timestamps are not monotonically increasing: {client_state}"

    return {STATE_VAR: states}


strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=record_state_metrics
)

hist = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=2,
    config=fl.server.ServerConfig(num_rounds=2),
    strategy=strategy,
)

assert (
    hist.losses_distributed[-1][1] == 0
    or (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) >= 0.98
)

# The checks in record_state_metrics don't do anything if client's state has a single entry
state_metrics_last_round = hist.metrics_distributed[STATE_VAR][-1]
assert (
    len(state_metrics_last_round[1][0]) == 2 * state_metrics_last_round[0]
), "There should be twice as many entries in the client state as rounds"
