"""ServerApp for verifiable aggregation baseline."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp

from verifiableagg.model import Net
from verifiableagg.reporting import write_json_report
from verifiableagg.strategy import VerifiableFedAvg
from verifiableagg.utils import as_bool

app = ServerApp()


def _set_global_seeds(seed: int) -> None:
    """Set global seeds with deterministic CUDA behavior when available."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run federated training and write reproducibility report."""
    run_config = context.run_config

    seed = int(run_config["random-seed"])
    _set_global_seeds(seed)

    num_rounds = int(run_config["num-server-rounds"])
    num_features = int(run_config["num-features"])
    fraction_train = float(run_config["fraction-train"])
    fraction_evaluate = float(run_config["fraction-evaluate"])
    verify_aggregation = as_bool(run_config["verify-aggregation"])
    verification_tolerance = float(run_config["verification-tolerance"])
    artifacts_dir = Path(str(run_config["artifacts-dir"]))

    model = Net(num_features=num_features)
    initial_arrays = ArrayRecord(model.state_dict())

    strategy = VerifiableFedAvg(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_available_nodes=2,
        verify_aggregation=verify_aggregation,
        verification_tolerance=verification_tolerance,
        weighted_by_key="num-examples",
        arrayrecord_key="arrays",
        configrecord_key="config",
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
    )

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "final_model.pt"
    torch.save(result.arrays.to_torch_state_dict(), model_path)

    verification_rounds = [
        {
            "round": item.server_round,
            "num_replies": item.num_replies,
            "max_abs_diff": item.max_abs_diff,
            "passed": item.passed,
            "aggregate_hash": item.aggregate_hash,
        }
        for item in strategy.verification_rounds
    ]

    train_metrics = {
        str(round_id): dict(metrics)
        for round_id, metrics in result.train_metrics_clientapp.items()
    }
    eval_metrics = {
        str(round_id): dict(metrics)
        for round_id, metrics in result.evaluate_metrics_clientapp.items()
    }

    report = {
        "baseline": "verifiableagg",
        "run_config": {key: run_config[key] for key in sorted(run_config)},
        "train_metrics": train_metrics,
        "evaluate_metrics": eval_metrics,
        "verification_rounds": verification_rounds,
        "artifacts": {"model_path": str(model_path)},
    }

    report_path = artifacts_dir / "report.json"
    write_json_report(report=report, output_path=report_path)

    passed_rounds = sum(1 for item in verification_rounds if item["passed"])
    total_rounds = len(verification_rounds)
    print(
        f"Verification summary: {passed_rounds}/{total_rounds} rounds passed. "
        f"Report: {report_path}"
    )
