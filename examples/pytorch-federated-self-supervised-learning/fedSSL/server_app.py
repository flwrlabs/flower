import os
import torch
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fedSSL.model import SimClr, get_parameters, set_parameters


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m['loss'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"Loss": sum(accuracies) / sum(examples)}


class SaveModelStrategy(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            gb_simclr = SimClr()
            aggregated_ndarrays = parameters_to_ndarrays(aggregated_parameters)

            set_parameters(gb_simclr, aggregated_ndarrays)

            save_dir = './fedSSL/model_weights/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(gb_simclr.state_dict(), save_dir + f"model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    # Initialize model parameters
    ndarrays = get_parameters(SimClr())
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = SaveModelStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        fit_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)
