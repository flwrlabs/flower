import argparse
import os.path
from collections import OrderedDict
from typing import Dict, Tuple, List
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import Scalar

from datasets.utils.logging import disable_progress_bar
from substrafl.index_generator import NpIndexGenerator

from dataset_manager import fetch_camelyon, reset_data_folder, creates_data_folder
from data_managers import Data, CamelyonDataset
from model import Weldon
from utils import train, test


parser = argparse.ArgumentParser(description="Flower Simulation for weldon on camelyon")

parser.add_argument(
    "--num-cpus",
    type=int,
    default=1,
    help="Number of CPUs to assign to a virtual client",
)
parser.add_argument(
    "--num-gpus",
    type=float,
    default=0.0,
    help="Ratio of GPU memory to assign to a virtual client",
)
parser.add_argument(
    "--nb-train-data-samples",
    type=int,
    default=5,
    help="Number of data sample of 400 Mb to use for each train task on each client",
)
parser.add_argument(
    "--nb-test-data-samples",
    type=int,
    default=2,
    help="Number of data sample of 400 Mb to use for each test task",
)
parser.add_argument(
    "--pool-size", type=int, default=2, help="Number of clients in total"
)
parser.add_argument("--num-rounds", type=int, default=10, help="Number of FL rounds.")
parser.add_argument(
    "--n-local-steps",
    type=int,
    default=5,
    help="Number of batch to learn from at each step of the strategy",
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=8,
    help="Number of sample to use learn from for each local step",
)
parser.add_argument(
    "--data-path",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "data",
    help="Path to the data",
)


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, index_generator):
        self.trainset = trainset
        self.valset = valset
        self.index_generator = index_generator

        # Instantiate model
        self.model = Weldon(
            in_features=2048,
            out_features=1,
            n_top=10,
            n_bottom=10,
        )

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        batch_sampler = deepcopy(self.index_generator)
        batch_sampler.n_samples = len(self.trainset)

        # Construct dataloader
        train_dataloader = DataLoader(
            self.trainset,
            batch_sampler=batch_sampler,
        )

        # Define optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        # Train
        train(self.model, train_dataloader, optimizer, self.device)

        # Reset the iterator
        batch_sampler.reset_counter()

        # Return local model and statistics
        return self.get_parameters({}), len(train_dataloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)

        batch_sampler = deepcopy(self.index_generator)
        batch_sampler.n_samples = len(self.valset)

        # Construct dataloader
        valloader = DataLoader(
            self.valset,
            batch_sampler=batch_sampler,
        )
        # Evaluate
        loss, auc, accuracy = test(self.model, valloader, self.device)

        # Return statistics
        return (
            float(loss),
            len(valloader.dataset),
            {"auc": float(auc), "accuracy": float(accuracy)},
        )


def get_client_fn(train_data, test_data, index_generator):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str):
        """Construct a FlowerClient with its own dataset partition."""

        # Let's get the partition corresponding to the i-th client
        # Here we use the same data for simulation
        train_dataset = CamelyonDataset(datasamples=train_data)
        test_dataset = CamelyonDataset(datasamples=test_data)

        # Create and return client
        return FlowerClient(train_dataset, test_dataset, index_generator).to_client()

    return client_fn


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for (federated) evaluation metrics, i.e. those returned by
    the client's evaluate() method."""
    # Multiply accuracy of each client by number of examples used
    aucs = [num_examples * m["auc"] for num_examples, m in metrics]
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "auc": sum(aucs) / sum(examples),
        "accuracy": sum(accuracies) / sum(examples),
    }


def get_evaluate_fn(
    centralized_testdata,
    index_generator,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Weldon(
            in_features=2048,
            out_features=1,
            n_top=10,
            n_bottom=10,
        )
        set_params(model, parameters)
        model.to(device)

        test_dataset = CamelyonDataset(datasamples=centralized_testdata)
        batch_sampler = deepcopy(index_generator)
        batch_sampler.n_samples = len(test_dataset)

        # Construct dataloader
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=batch_sampler,
        )

        # Evaluate
        loss, auc, accuracy = test(model, test_loader, device)
        return loss, {"auc": auc, "accuracy": accuracy}

    return evaluate


def main():
    # Parse input arguments
    args = parser.parse_args()

    # Data pre-processing
    index_generator = NpIndexGenerator(
        batch_size=args.batch_size,
        num_updates=args.n_local_steps,
        drop_last=True,
        shuffle=False,
    )
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    exp_data_path = args.data_path / "tmp"
    fetch_camelyon(args.data_path)
    reset_data_folder(exp_data_path)
    train_folder = creates_data_folder(
        img_dir_path=args.data_path / "tiles_0.5mpp",
        dest_folder=exp_data_path / "train",
    )
    test_folder = creates_data_folder(
        img_dir_path=args.data_path / "tiles_0.5mpp",
        dest_folder=exp_data_path / "test",
    )

    train_camelyon = Data(paths=[train_folder] * args.nb_train_data_samples)
    test_camelyon = Data(paths=[test_folder] * args.nb_test_data_samples)

    # Configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=2,  # Never sample less than 2 clients for training
        min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
        min_available_clients=args.pool_size,  # Wait until all clients are available
        evaluate_metrics_aggregation_fn=weighted_average,  # Aggregate federated metrics
        evaluate_fn=get_evaluate_fn(
            test_camelyon, index_generator
        ),  # Global evaluation function
    )

    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(train_camelyon, test_camelyon, index_generator),
        num_clients=args.pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
    )


if __name__ == "__main__":
    main()
