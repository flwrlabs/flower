"""FedAvg with all members of the configured cohort participating in every round."""

import json
import os
import socket
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pbs_pytorch.task import make_model, model_hash

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run federated training and retain round metrics and the final checkpoint."""
    torch.set_num_threads(2)
    torch.manual_seed(42)
    count = int(context.run_config["num-partitions"])
    rounds = int(context.run_config["num-server-rounds"])
    initial = make_model().state_dict()
    initial_hash = model_hash(initial)
    strategy = FedAvg(
        min_train_nodes=count, min_evaluate_nodes=count, min_available_nodes=count
    )
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(initial),
        train_config=ConfigRecord({"lr": float(context.run_config["learning-rate"])}),
        num_rounds=rounds,
    )
    expected = set(range(1, rounds + 1))
    if (
        set(result.train_metrics_clientapp) != expected
        or set(result.evaluate_metrics_clientapp) != expected
    ):
        raise RuntimeError("Training or evaluation rounds are missing")
    final = result.arrays.to_torch_state_dict()
    final_hash = model_hash(final)
    if final_hash == initial_hash:
        raise RuntimeError("Global weights did not change")
    output = Path(os.environ["OUTPUT_ROOT"])
    torch.save(final, output / "final_model.pt")
    summary = {
        "run_id": context.run_id,
        "host": socket.gethostname(),
        "num_partitions": count,
        "rounds": rounds,
        "initial_sha256": initial_hash,
        "final_sha256": final_hash,
        "config": dict(context.run_config),
        "train": {str(k): dict(v) for k, v in result.train_metrics_clientapp.items()},
        "evaluate": {
            str(k): dict(v) for k, v in result.evaluate_metrics_clientapp.items()
        },
    }
    (output / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
