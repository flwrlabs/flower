"""fedagain: Flower ServerApp."""

import time
from logging import INFO
from pathlib import Path
from typing import Any, cast

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedMedian
from torch.utils.data import DataLoader

from fedagain.dataset import (
    corrupted_clients,
    get_dataset_spec,
    load_centralized_testset,
)
from fedagain.model import build_model
from fedagain.model import test as test_fn
from fedagain.strategy import FedAgain
from fedagain.utils import (
    aggregate_train_metrics,
    corrupted_weight_share,
    get_device,
    save_results,
    set_seed,
)

app = ServerApp()


def _build_strategy(run_config: dict, num_nodes: int) -> FedAvg:
    """Build the strategy named in the run config with shared sampling options."""
    # All nodes must be connected so that every run uses the same clients
    name = str(run_config["strategy"]).lower()
    kwargs: dict[str, Any] = {
        "fraction_train": float(run_config["fraction-train"]),
        "fraction_evaluate": float(run_config["fraction-evaluate"]),
        "min_available_nodes": num_nodes,
        "train_metrics_aggr_fn": aggregate_train_metrics,
    }
    if name == "fedagain":
        return FedAgain(epsilon=float(run_config["epsilon"]), **kwargs)
    if name == "fedavg":
        return FedAvg(**kwargs)
    if name == "fedmedian":
        return FedMedian(**kwargs)
    raise ValueError(
        f"Unknown strategy '{name}'. Use 'fedagain', 'fedavg' or 'fedmedian'."
    )


def run_name(run_config: dict, num_clients: int) -> str:
    """Return the name of the result file of a run (read by ``fedagain.plot``)."""
    return (
        f"{run_config['dataset']}_{run_config['partitioning']}_n{num_clients}"
        f"_rho{float(run_config['corrupted-fraction']):g}"
        f"_sev{run_config['corruption-severity']}"
        f"_{run_config['strategy']}_seed{run_config['seed']}"
    )


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run FedAgain (or a comparison strategy) and store the results."""
    run_config = dict(context.run_config)
    num_rounds = int(run_config["num-server-rounds"])
    set_seed(int(run_config["seed"]))

    spec = get_dataset_spec(str(run_config["dataset"]))
    model = build_model(str(run_config["model"]), spec.num_classes, spec.in_channels)
    initial_arrays = ArrayRecord(model.state_dict())

    num_nodes = len(list(grid.get_node_ids()))
    strategy = _build_strategy(run_config, max(num_nodes, 2))
    # Same seeded selection as in the ClientApps (dataset.corrupted_clients)
    bad_clients = corrupted_clients(
        num_nodes, float(run_config["corrupted-fraction"]), int(run_config["seed"])
    )
    log(INFO, "Corrupted clients (partition ids): %s", bad_clients)

    # Centralized evaluation of the global model on the clean test split
    testloader = DataLoader(load_centralized_testset(run_config), batch_size=512)
    device = get_device()

    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        net = build_model(str(run_config["model"]), spec.num_classes, spec.in_channels)
        net.load_state_dict(arrays.to_torch_state_dict())
        loss, acc = test_fn(net, testloader, device)
        log(INFO, "Round %d: test accuracy %.4f, loss %.4f", server_round, acc, loss)
        return MetricRecord({"accuracy": acc, "loss": loss})

    start = time.time()
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
        train_config=ConfigRecord(),
        evaluate_fn=evaluate_fn,
    )
    elapsed = time.time() - start

    central = result.evaluate_metrics_serverapp
    trust_history: list[dict[str, Any]] = getattr(strategy, "trust_history", [])
    results: dict[str, Any] = {
        "config": run_config,
        "corrupted_clients": bad_clients,
        "test_accuracy": {
            str(r): cast(float, m["accuracy"]) for r, m in central.items()
        },
        "test_loss": {str(r): cast(float, m["loss"]) for r, m in central.items()},
        "train_metrics": {
            str(r): dict(m) for r, m in result.train_metrics_clientapp.items()
        },
        "trust_history": trust_history,
        "elapsed_seconds": elapsed,
    }
    # Weight of the corrupted clients under data-size and FedAgain weighting
    history = trust_history
    if history:
        sizes = dict(
            zip(history[0]["client_ids"], history[0]["num_examples"], strict=False)
        )
        total = sum(sizes.values())
        results["fedavg_corrupted_share"] = (
            sum(n for cid, n in sizes.items() if cid in bad_clients) / total
            if total > 0
            else None
        )
        results["fedagain_corrupted_share"] = corrupted_weight_share(results)

    out_path = Path(str(run_config["output-dir"])).expanduser().resolve()
    out_file = out_path / f"{run_name(run_config, num_nodes)}.json"
    save_results(out_file, results)
    log(INFO, "Results saved to %s", out_file)
