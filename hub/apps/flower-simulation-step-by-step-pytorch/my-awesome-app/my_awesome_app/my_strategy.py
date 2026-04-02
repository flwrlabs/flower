import json
from datetime import datetime

import torch
import wandb
from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .task import Net, set_weights


class CustomFedAvg(FedAvg):
    """A strategy that keeps the core functionality of FedAvg unchanged but enables
    additional features such as: Saving global checkpoints, saving metrics to the local
    file system as a JSON, pushing metrics to Weight & Biases.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # A dictionary that will store the metrics generated on each round
        self.results_to_save = {}

        # Log those same metrics to W&B
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        wandb.init(project="flower-simulation-tutorial", name=f"custom-strategy-{name}")

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        """Aggregate received model updates and metrics, ave global model checkpoint."""

        # Call the default aggregate_fit method from FedAvg
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        ## Save new Global Model as a PyTorch checkpoint
        # Convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)
        # Instantiate model
        model = Net()
        # Apply paramters to model
        set_weights(model, ndarrays)
        # Save global model in the standard PyTorch way
        torch.save(model.state_dict(), f"global_model_round_{server_round}")

        # Return the expected outputs for `aggregate_fit`
        return parameters_aggregated, metrics_aggregated

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        """Evaluate global model, then save metrics to local JSON and to W&B."""
        # Call the default behaviour from FedAvg
        loss, metrics = super().evaluate(server_round, parameters)

        # Store metrics as dictionary
        my_results = {"loss": loss, **metrics}
        # Insert into local dictionary
        self.results_to_save[server_round] = my_results

        # Save metrics as json
        with open("results.json", "w") as json_file:
            json.dump(self.results_to_save, json_file, indent=4)

        # Log metrics to W&B
        wandb.log(my_results, step=server_round)

        # Return the expected outputs for `evaluate`
        return loss, metrics
