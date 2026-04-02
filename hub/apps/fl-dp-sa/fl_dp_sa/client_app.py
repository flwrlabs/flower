"""fl_dp_sa: Flower Example using Differential Privacy and Secure Aggregation."""

import torch
from flwr.client import NumPyClient
from flwr.client.mod import fixedclipping_mod, secaggplus_mod
from flwr.clientapp import ClientApp
from flwr.common import Context

from fl_dp_sa.task import Net, get_weights, load_data, set_weights, test, train


class FlowerClient(NumPyClient):
    def __init__(self, trainloader, testloader) -> None:
        self.net = Net()
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.testloader,
            epochs=1,
            device=self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    trainloader, testloader = load_data(
        partition_id=partition_id, num_partitions=context.node_config["num-partitions"]
    )
    return FlowerClient(trainloader, testloader).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
        fixedclipping_mod,
    ],
)
