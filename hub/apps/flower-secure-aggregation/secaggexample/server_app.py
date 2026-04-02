"""secaggexample: A Flower with SecAgg+ app."""

from logging import DEBUG
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.logger import update_console_handler
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow

from secaggexample.task import get_weights, make_net
from secaggexample.workflow_with_log import SecAggPlusWorkflowWithLogs


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Flower ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:

    is_demo = context.run_config["is-demo"]

    # Get initial parameters
    ndarrays = get_weights(make_net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvg(
        # Select all available clients
        fraction_fit=1.0,
        min_fit_clients=5,
        # Disable evaluation in demo
        fraction_evaluate=(0.0 if is_demo else context.run_config["fraction-evaluate"]),
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

    # Construct the LegacyContext
    num_rounds = context.run_config["num-server-rounds"]
    context = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Create fit workflow
    # For further information, please see:
    # https://flower.ai/docs/framework/ref-api/flwr.server.workflow.SecAggPlusWorkflow.html
    if is_demo:
        update_console_handler(DEBUG, True, True)
        fit_workflow = SecAggPlusWorkflowWithLogs(
            num_shares=context.run_config["num-shares"],
            reconstruction_threshold=context.run_config["reconstruction-threshold"],
            max_weight=1,
            timeout=context.run_config["timeout"],
        )
    else:
        fit_workflow = SecAggPlusWorkflow(
            num_shares=context.run_config["num-shares"],
            reconstruction_threshold=context.run_config["reconstruction-threshold"],
            max_weight=context.run_config["max-weight"],
        )

    # Create the workflow
    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    # Execute
    workflow(grid, context)
