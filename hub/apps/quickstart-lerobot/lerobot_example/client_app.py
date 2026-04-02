"""lerobot_example: A Flower / Hugging Face LeRobot app."""

import warnings
from logging import INFO
from pathlib import Path

import torch
from flwr.client import Client, NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context
from flwr.common.logger import log
from transformers import logging

from lerobot_example.task import (
    get_model,
    get_params,
    load_data,
    set_params,
    test,
    train,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# To mute warnings reminding that we need to train the model to a downstream task
# This is something this example does.
logging.set_verbosity_error()


# Flower client
class LeRobotClient(NumPyClient):
    def __init__(self, partition_id, local_epochs, trainloader, nn_device=None) -> None:
        self.partition_id = partition_id
        self.trainloader = trainloader
        self.net = get_model(dataset_stats=trainloader.dataset.stats)
        self.local_epochs = local_epochs
        policy = self.net
        self.device = nn_device
        if self.device == torch.device("cpu"):
            # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
            policy.diffusion.num_inference_steps = 10
        policy.to(self.device)

    def fit(self, parameters, config) -> tuple[list, int, dict]:
        set_params(self.net, parameters)
        train(
            net=self.net,
            trainloader=self.trainloader,
            epochs=self.local_epochs,
            device=self.device,
        )
        return get_params(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, float]]:
        set_params(self.net, parameters)
        round_save_path = Path(config["save_path"])
        if config["skip"]:
            log(INFO, "Skipping evaluation")
            accuracy, loss = 0.0, 0.0
        else:
            loss, accuracy = test(
                partition_id=self.partition_id,
                net=self.net,
                device=self.device,
                output_dir=round_save_path,
            )

        testset_len = 1  # we test on one gym generated task
        return float(loss), testset_len, {"accuracy": accuracy}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    # Discover device
    nn_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Read the run config to get settings to configure the Client
    model_name = context.run_config["model-name"]
    local_epochs = int(context.run_config["local-epochs"])
    trainloader = load_data(partition_id, num_partitions, model_name, device=nn_device)

    return LeRobotClient(
        partition_id=partition_id,
        local_epochs=local_epochs,
        trainloader=trainloader,
        nn_device=nn_device,
    ).to_client()


app = ClientApp(client_fn=client_fn)
