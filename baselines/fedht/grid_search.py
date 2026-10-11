"""Grid search script for FedHT hyperparameter tuning.

Runs Fed-HT, FedIter-HT, and Distributed-IHT across a range of
learning rates and local step counts. Results are saved as CSV files
and can be plotted with plot_results.py.

Usage:
    python3 grid_search.py --experiment simulation_I
    python3 grid_search.py --experiment simulation_II
    python3 grid_search.py --experiment mnist
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

FLWR_BIN = str(Path(sys.executable).parent / "flwr")
PROJECT_DIR = str(Path(__file__).parent)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Learning rates to search, ordered from smallest to largest.
# We test the full paper range. Runs that diverge in the first 10
# rounds are skipped automatically.
LEARNING_RATES = [0.001, 0.003, 0.01, 0.03, 0.06, 0.1, 0.3]

# Local step counts K to search.
LOCAL_STEPS = [3, 5, 8, 10]

# Number of rounds for the full search.
NUM_ROUNDS = 100

# Clients and batch size per experiment.
# String values are marked with a leading "s:" prefix so the builder
# can wrap them in the double quotes that flwr --run-config requires.
EXPERIMENT_CONFIG = {
    "simulation_I": {
        "dataset.name": "s:simulation_I",
        "model.input-dim": "1000",
        "dataset.batch-size": "20",
        "algorithm.tau": "200",
        "algorithm.num-clients": "100",
    },
    "simulation_II": {
        "dataset.name": "s:simulation_II",
        "model.input-dim": "1000",
        "dataset.batch-size": "20",
        "algorithm.tau": "200",
        "algorithm.num-clients": "100",
    },
    "mnist": {
        "dataset.name": "s:mnist",
        "model.input-dim": "784",
        "model.num-classes": "10",
        "dataset.batch-size": "64",
        "algorithm.tau": "500",
        "algorithm.num-clients": "100",
    },
}


def fmt_val(val: str) -> str:
    """Format a config value for flwr --run-config.

    String values prefixed with "s:" are wrapped in double quotes.
    All other values are passed through unchanged.
    """
    if val.startswith("s:"):
        return f'"{val[2:]}"'
    return val


FIT_PROGRESS_RE = re.compile(r"fit progress: \((\d+), ([0-9.eE+\-inf]+), ")


def parse_objective(line: str):
    """Extract (round_num, objective) from a fit progress log line."""
    match = FIT_PROGRESS_RE.search(line)
    if not match:
        return None
    round_num = int(match.group(1))
    obj_str = match.group(2)
    try:
        obj = float(obj_str)
    except ValueError:
        obj = float("nan")
    return round_num, obj


def run_config(
    algorithm: str,
    lr: float,
    k: int,
    experiment: str,
    num_rounds: int = NUM_ROUNDS,
) -> list[tuple[int, float]]:
    """Run one flwr configuration and return per-round objective values.

    Parameters
    ----------
    algorithm : str
        One of "FedHT", "FedIterHT". Use k=1 for Distributed-IHT.
    lr : float
        Learning rate for local SGD.
    k : int
        Number of local SGD steps per round.
    experiment : str
        One of "simulation_I", "simulation_II", "mnist".
    num_rounds : int
        Number of communication rounds.

    Returns
    -------
    list[tuple[int, float]]
        List of (round, objective) pairs from the run log.
    """
    base_cfg = EXPERIMENT_CONFIG[experiment]
    cfg_parts = [f"{key}={fmt_val(val)}" for key, val in base_cfg.items()]
    cfg_parts += [
        f'algorithm.name="{algorithm}"',
        f"algorithm.learning-rate={lr}",
        f"algorithm.local-steps={k}",
        f"algorithm.num-server-rounds={num_rounds}",
        "algorithm.fraction-fit=1.0",
        "algorithm.min-available-clients=2",
    ]
    run_config_str = " ".join(cfg_parts)

    cmd = [
        FLWR_BIN,
        "run",
        PROJECT_DIR,
        "local-simulation",
        "--run-config",
        run_config_str,
        "--stream",
    ]

    env = os.environ.copy()
    results = []

    print(f"  Running: alg={algorithm} lr={lr} K={k} ...", flush=True)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        diverged = False
        for line in proc.stdout:
            parsed = parse_objective(line)
            if parsed is not None:
                round_num, obj = parsed
                results.append((round_num, obj))
                # Stop early if the run has clearly diverged.
                if round_num >= 5 and (obj != obj or obj > 1e15):
                    diverged = True
                    proc.terminate()
                    break
        proc.wait()
        if diverged:
            print(f"    Diverged at round {round_num} -- skipping", flush=True)
            return []
    except Exception as exc:
        print(f"    Run failed: {exc}", flush=True)
        return []

    print(
        (
            f"    Done. Final objective: {results[-1][1]:.4f}"
            if results
            else "    No results captured"
        ),
        flush=True,
    )
    return results


def save_results(
    results: list[tuple[int, float]],
    algorithm: str,
    lr: float,
    k: int,
    experiment: str,
) -> Path:
    """Save per-round results to a CSV file."""
    filename = f"{experiment}_{algorithm}_lr{lr}_K{k}.csv"
    path = RESULTS_DIR / filename
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "objective"])
        writer.writerows(results)
    return path


def main():
    """Parse CLI arguments and run the grid search."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=list(EXPERIMENT_CONFIG.keys()),
        default="simulation_I",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["FedHT", "FedIterHT"],
        help="Algorithms to run. Use FedHT with K=1 for Distributed-IHT.",
    )
    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=LEARNING_RATES,
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=LOCAL_STEPS,
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_ROUNDS,
    )
    parser.add_argument(
        "--distributed-iht",
        action="store_true",
        help="Also run Distributed-IHT (FedHT with K=1) as baseline.",
    )
    args = parser.parse_args()

    print(
        f"Grid search: experiment={args.experiment}, "
        f"algorithms={args.algorithms}, "
        f"lrs={args.lrs}, "
        f"K={args.k_values}, "
        f"rounds={args.rounds}"
    )
    print(f"Results will be saved to: {RESULTS_DIR}")
    print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
    print()

    # Distributed-IHT baseline: FedHT with K=1.
    if args.distributed_iht:
        print("=== Distributed-IHT baseline (K=1) ===")
        for lr in args.lrs:
            results = run_config(
                "FedHT", lr, k=1, experiment=args.experiment, num_rounds=args.rounds
            )
            if results:
                save_results(
                    results, "DistributedIHT", lr, k=1, experiment=args.experiment
                )

    for algorithm in args.algorithms:
        print(f"\n=== {algorithm} ===")
        for k in args.k_values:
            print(f"\n  K={k}")
            for lr in args.lrs:
                results = run_config(
                    algorithm, lr, k, experiment=args.experiment, num_rounds=args.rounds
                )
                if results:
                    save_results(results, algorithm, lr, k, experiment=args.experiment)

    print(f"\nGrid search complete at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
