import numpy as np

import flwr as fl

STATE_VAR = "timestamp"


# Define metric aggregation function
def record_state_metrics(metrics):
    """Ensure that timestamps are monotonically increasing."""
    if not metrics:
        return {}

    if STATE_VAR not in metrics[0][1]:
        # Do nothing if keyword is not present
        return {}

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


app = fl.serverapp.ServerApp()


@app.main()
def main(grid, context):
    # Construct the LegacyContext
    context = fl.server.LegacyContext(
        context=context,
        config=fl.server.ServerConfig(num_rounds=2),
    )

    # Create the workflow
    workflow = fl.server.workflow.DefaultWorkflow()

    # Execute
    workflow(grid, context)

    hist = context.history
    assert (
        hist.losses_distributed[-1][1] == 0
        or (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) >= 0.98
    )


if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=record_state_metrics
    )

    hist = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
    )

    assert (
        hist.losses_distributed[-1][1] == 0
        or (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) >= 0.98
    )

    if STATE_VAR in hist.metrics_distributed:
        # The checks in record_state_metrics don't do anythinng if client's state has a single entry
        state_metrics_last_round = hist.metrics_distributed[STATE_VAR][-1]
        assert (
            len(state_metrics_last_round[1][0]) == 2 * state_metrics_last_round[0]
        ), "There should be twice as many entries in the client state as rounds"
