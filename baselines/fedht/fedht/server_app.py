"""FedHT server application."""

import numpy as np
import torch
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader, TensorDataset

from .dataset import generate_simulation_I, generate_simulation_II, make_loaders
from .model import (
    SparseLinearRegression,
    SparseLogisticRegression,
    SparseSoftmaxRegression,
    get_weights,
    set_weights,
    eval_linear,
    eval_logistic,
    eval_softmax,
)
from .strategy import FedHT, FedIterHT

TASK_LINEAR = "simulation_I"
TASK_LOGISTIC = "simulation_II"
TASK_SOFTMAX = "mnist"


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Compute sample-weighted average of client metrics."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    aggregated: Metrics = {}
    for key in metrics[0][1]:
        aggregated[key] = sum(n * float(m[key]) for n, m in metrics) / total
    return aggregated


def _build_val_loader(task: str, num_clients: int, batch_size: int) -> DataLoader:
    """Build a centralized validation loader for server-side evaluation."""
    if task == TASK_LINEAR:
        all_data = generate_simulation_I(num_clients=num_clients, seed=42)
    elif task == TASK_LOGISTIC:
        all_data = generate_simulation_II(num_clients=num_clients, seed=42)
    elif task == TASK_SOFTMAX:
        from flwr_datasets import FederatedDataset
        from flwr_datasets.partitioner import IidPartitioner

        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": IidPartitioner(num_partitions=num_clients)},
        )
        test_data = fds.load_split("test").with_format("numpy")
        X_test = (
            test_data["image"].reshape(len(test_data["image"]), -1).astype(np.float32)
            / 255.0
        )
        y_test = test_data["label"].astype(np.int64)
        return DataLoader(
            TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
            batch_size=64,
        )
    else:
        raise ValueError(f"Unknown dataset: {task!r}")

    X_all = np.concatenate([X for X, _ in all_data], axis=0)
    y_all = np.concatenate([y for _, y in all_data], axis=0)
    split = max(1, int(len(X_all) * 0.1))
    _, val_loader = make_loaders(X_all[-split:], y_all[-split:], batch_size=batch_size)
    return val_loader


def make_eval_fn(net, task, val_loader, device):
    """Return a centralized evaluation function for the given task."""

    def evaluate(server_round, parameters_ndarrays, config):
        set_weights(net, parameters_ndarrays)

        if task == TASK_LINEAR:
            loss, _ = eval_linear(net, val_loader, device)
            return float(loss), {"objective": float(loss)}

        if task == TASK_LOGISTIC:
            loss, acc, _ = eval_logistic(net, val_loader, device)
            return float(loss), {"objective": float(loss), "accuracy": float(acc)}

        if task == TASK_SOFTMAX:
            loss, acc, _ = eval_softmax(net, val_loader, device)
            return float(loss), {"objective": float(loss), "accuracy": float(acc)}

        raise ValueError(f"Unknown task: {task!r}")

    return evaluate


def server_fn(context: Context) -> ServerAppComponents:
    """Build FedHT or FedIter-HT server components from the run config."""
    run_cfg = context.run_config

    task = str(run_cfg.get("dataset.name", TASK_LINEAR))
    strategy_name = str(run_cfg.get("algorithm.name", "FedHT"))
    num_rounds = int(run_cfg.get("algorithm.num-server-rounds", 100))
    tau = int(run_cfg.get("algorithm.tau", 200))
    fraction_fit = float(run_cfg.get("algorithm.fraction-fit", 1.0))
    min_clients = int(run_cfg.get("algorithm.min-available-clients", 2))
    num_clients = int(run_cfg.get("algorithm.num-clients", 100))
    batch_size = int(run_cfg.get("dataset.batch-size", 20))
    device = torch.device("cpu")

    if task == TASK_LINEAR:
        net: torch.nn.Module = SparseLinearRegression(
            input_dim=int(run_cfg.get("model.input-dim", 1000))
        )
    elif task == TASK_LOGISTIC:
        net = SparseLogisticRegression(
            input_dim=int(run_cfg.get("model.input-dim", 1000))
        )
    elif task == TASK_SOFTMAX:
        net = SparseSoftmaxRegression(
            input_dim=784,
            num_classes=int(run_cfg.get("model.num-classes", 10)),
        )
    else:
        raise ValueError(f"Unknown dataset: {task!r}")

    for param in net.parameters():
        torch.nn.init.zeros_(param)

    initial_parameters = ndarrays_to_parameters(get_weights(net))
    val_loader = _build_val_loader(task, num_clients, batch_size)
    evaluate_fn = make_eval_fn(net, task, val_loader, device)

    def fit_config_fn(server_round: int) -> dict:
        return {"current_round": server_round}

    strategy_kwargs = {
        "tau": tau,
        "fraction_fit": fraction_fit,
        "fraction_evaluate": 0.0,
        "min_available_clients": min_clients,
        "initial_parameters": initial_parameters,
        "on_fit_config_fn": fit_config_fn,
        "evaluate_fn": evaluate_fn,
        "fit_metrics_aggregation_fn": weighted_average,
    }

    strategy: FedHT
    if strategy_name == "FedIterHT":
        strategy = FedIterHT(**strategy_kwargs)
    else:
        strategy = FedHT(**strategy_kwargs)

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
