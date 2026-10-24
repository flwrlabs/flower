"""bcfl_edge: Saving results."""

import json
from datetime import datetime
from pathlib import Path

from flwr.app import MetricRecord


def save_results(
    run_config: dict, round_cost: dict, evaluation: dict[int, MetricRecord]
) -> Path:
    """Write the run config, blockchain round cost and accuracy per round to JSON."""
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    name = f"{run_config['selection']}_sd{run_config['selected-devices']}"
    path = out_dir / f"{name}_{datetime.now():%Y%m%d-%H%M%S}.json"
    results = {
        "run-config": run_config,
        "round-cost": round_cost,
        "evaluation": {rnd: dict(metrics) for rnd, metrics in evaluation.items()},
    }
    path.write_text(json.dumps(results, indent=2))
    return path
