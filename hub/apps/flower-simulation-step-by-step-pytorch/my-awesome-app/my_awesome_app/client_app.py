"""my-awesome-app: A Flower / PyTorch app."""

import json
from random import random

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import ConfigRecord, Context

from my_awesome_app.task import Net, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, context: Context):
        self.client_state = context.state
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        if "fit_metrics" not in self.client_state.config_records:
            self.client_state.config_records["fit_metrics"] = ConfigRecord()

    def fit(self, parameters, config):
        """Train a model using as starting point the parameters sent by the ServerApp.

        Then, communicate the weights of the locally-updated model back to the
        ServerApp.
        """
        # Apply parameters to local model
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            config["lr"],
            self.device,
        )

        # Append to persistent state the `train_loss` just obtained
        fit_metrics = self.client_state.config_records["fit_metrics"]
        if "train_loss_hist" not in fit_metrics:
            # If first entry, create the list
            fit_metrics["train_loss_hist"] = [train_loss]
        else:
            # If it's not the first entry, append to the existing list
            fit_metrics["train_loss_hist"].append(train_loss)

        # A complex metric strcuture can be returned by a ClientApp if it is first
        # converted to a supported type by `flwr.common.Scalar`. Here we serialize it with
        # JSON and therefore representing it as a string (one of the supported types)
        complex_metric = {"a": 123, "b": random(), "mylist": [1, 2, 3, 4]}
        complex_metric_str = json.dumps(complex_metric)

        return (
            get_weights(self.net),  # Return parameters of the locally-updated model
            len(
                self.trainloader.dataset
            ),  # Training examples used (needed sometimes for aggregation)
            {
                "train_loss": train_loss,
                "my_metric": complex_metric_str,
            },  # Communicate metrics
        )

    def evaluate(self, parameters, config):
        """Evaluate the global model weights using the local validation set."""
        # Apply weights from global model
        set_weights(self.net, parameters)
        # Run the test evaluation function
        loss, accuracy = test(self.net, self.valloader, self.device)
        # Report results. Note the last argument is of type `Metrics` so you could communicate
        # other values that are relevant to your use case.
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """A function that returns a Client."""

    # Instantiate the model
    net = Net()
    # Read node config and fetch data for the ClientApp that is being constructed
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)

    # Read the run config (defined in the `pyproject.toml`)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, context).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
