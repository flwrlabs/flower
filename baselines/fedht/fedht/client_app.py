"""FedHT client application."""

import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

from .dataset import (
    generate_simulation_I,
    generate_simulation_II,
    make_loaders,
    load_mnist_partition,
)
from .model import (
    SparseLinearRegression,
    SparseLogisticRegression,
    SparseSoftmaxRegression,
    get_weights,
    set_weights,
    train_linear,
    train_logistic,
    train_softmax,
    eval_linear,
    eval_logistic,
    eval_softmax,
)

TASK_LINEAR = "simulation_I"
TASK_LOGISTIC = "simulation_II"
TASK_SOFTMAX = "mnist"


class FedHTClient(NumPyClient):
    """Flower client for Fed-HT and FedIter-HT across all three experimental tasks."""

    def __init__(self, net, train_loader, val_loader, task, local_steps, lr, device):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task = task
        self.local_steps = local_steps
        self.lr = lr
        self.device = device
        self.net.to(self.device)

    def fit(self, parameters, config):
        """Run K local SGD steps and return updated parameters."""
        set_weights(self.net, parameters)

        use_local_ht = bool(config.get("use_local_ht", False))
        tau = int(config.get("tau", 0))

        train_kwargs = {
            "loader": self.train_loader,
            "device": self.device,
            "lr": self.lr,
            "local_steps": self.local_steps,
            "use_local_ht": use_local_ht,
            "tau": tau,
        }

        if self.task == TASK_LINEAR:
            train_linear(self.net, **train_kwargs)
        elif self.task == TASK_LOGISTIC:
            train_logistic(self.net, **train_kwargs)
        else:
            train_softmax(self.net, **train_kwargs)

        num_samples = len(self.train_loader.dataset)
        return get_weights(self.net), num_samples, {}

    def evaluate(self, parameters, config):
        """Evaluate the global model on the local validation set."""
        set_weights(self.net, parameters)
        num_samples = len(self.val_loader.dataset)

        if self.task == TASK_LINEAR:
            loss, _ = eval_linear(self.net, self.val_loader, self.device)
            return float(loss), num_samples, {"loss": float(loss)}

        if self.task == TASK_LOGISTIC:
            loss, acc, _ = eval_logistic(self.net, self.val_loader, self.device)
            return (
                float(loss),
                num_samples,
                {"loss": float(loss), "accuracy": float(acc)},
            )

        if self.task == TASK_SOFTMAX:
            loss, acc, _ = eval_softmax(self.net, self.val_loader, self.device)
            return (
                float(loss),
                num_samples,
                {"loss": float(loss), "accuracy": float(acc)},
            )

        raise ValueError(f"Unknown task: {self.task!r}")


def client_fn(context: Context) -> Client:
    """Construct a FedHTClient for a given partition."""
    run_cfg = context.run_config
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])

    task = str(run_cfg.get("dataset.name", TASK_LINEAR))
    local_steps = int(run_cfg.get("algorithm.local-steps", 5))
    lr = float(run_cfg.get("algorithm.learning-rate", 0.1))
    batch_size = int(run_cfg.get("dataset.batch-size", 20))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net: torch.nn.Module
    if task == TASK_LINEAR:
        d = int(run_cfg.get("model.input-dim", 1000))
        all_data = generate_simulation_I(num_clients=num_partitions, seed=42)
        X, y = all_data[partition_id]
        train_loader, val_loader = make_loaders(
            X, y, batch_size=batch_size, seed=partition_id
        )
        net = SparseLinearRegression(input_dim=d)

    elif task == TASK_LOGISTIC:
        d = int(run_cfg.get("model.input-dim", 1000))
        all_data = generate_simulation_II(num_clients=num_partitions, seed=42)
        X, y = all_data[partition_id]
        train_loader, val_loader = make_loaders(
            X, y, batch_size=batch_size, seed=partition_id
        )
        net = SparseLogisticRegression(input_dim=d)

    elif task == TASK_SOFTMAX:
        d = 784
        num_classes = int(run_cfg.get("model.num-classes", 10))
        train_loader, val_loader = load_mnist_partition(
            partition_id=partition_id,
            num_partitions=num_partitions,
            batch_size=batch_size,
        )
        net = SparseSoftmaxRegression(input_dim=d, num_classes=num_classes)

    else:
        raise ValueError(f"Unknown dataset: {task!r}")

    # Initialize to zero as specified in the paper (x_0 = 0).
    for param in net.parameters():
        torch.nn.init.zeros_(param)

    return FedHTClient(
        net=net,
        train_loader=train_loader,
        val_loader=val_loader,
        task=task,
        local_steps=local_steps,
        lr=lr,
        device=device,
    ).to_client()


app = ClientApp(client_fn=client_fn)
