"""sklearnexample: A Flower / sklearn app."""

from typing import Dict, List

from flwr.app import ArrayRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from sklearnexample.task import (
    create_log_reg_and_instantiate_parameters,
    get_model_parameters,
)


def weighted_average(replies: List[RecordDict], weighted_by_key: str) -> MetricRecord:
    """Compute a weighted average of metrics returned by clients."""
    total_weight = 0.0
    metrics_sums: Dict[str, float] = {}

    # Accumulate weighted metrics from each reply
    for record in replies:
        metrics: MetricRecord = record["metrics"]  
        weight = float(metrics[weighted_by_key])  
        total_weight += weight
        for name, value in metrics.items():
            if name == weighted_by_key:  # skip num-examples
                continue
            if isinstance(value, (float, int)):
                metrics_sums[name] = metrics_sums.get(name, 0.0) + float(value) * weight

    # Compute averages
    averaged_metrics = {
        name: metrics_sums[name] / total_weight for name in metrics_sums
    }

    return MetricRecord(averaged_metrics)


# Instantiate the ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Entry point for the server."""

    # Build the initial model based on the run configuration
    penalty = context.run_config["penalty"]
    model = create_log_reg_and_instantiate_parameters(penalty)
    ndarrays = get_model_parameters(model)
    initial_arrays = ArrayRecord.from_numpy_ndarrays(ndarrays)  

    # Configure the strategy. 
    min_available_nodes = context.run_config["min-available-clients"]
    strategy = FedAvg(
        min_available_nodes=min_available_nodes,
        train_metrics_aggr_fn=weighted_average,
        evaluate_metrics_aggr_fn=weighted_average,
    )

    # Number of federated learning rounds
    num_rounds = context.run_config["num-server-rounds"]

    # Start federated learning. The result includes the final model and
    # aggregated metrics across rounds.
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
    )

    # Print result for debugging or downstream use
    print(result)
