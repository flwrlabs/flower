"""Plot grid search results for the FedHT baseline.

Reads CSV files from the results/ directory and generates comparison
plots of objective value vs communication rounds, matching Figure 3
and Figure 5 from the paper.

Usage:
    python3 plot_results.py --experiment simulation_I
    python3 plot_results.py --experiment simulation_I --best-only
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
STATIC_DIR = Path(__file__).parent / "_static"
STATIC_DIR.mkdir(exist_ok=True)

ALGORITHM_COLORS = {
    "FedHT": "#1f77b4",
    "FedIterHT": "#ff7f0e",
    "DistributedIHT": "#2ca02c",
}

ALGORITHM_LABELS = {
    "FedHT": "Fed-HT",
    "FedIterHT": "FedIter-HT",
    "DistributedIHT": "Distributed-IHT",
}


def load_csv(path: Path) -> tuple[list[int], list[float]]:
    """Load rounds and objective values from a results CSV file."""
    rounds, objectives = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rounds.append(int(row["round"]))
            objectives.append(float(row["objective"]))
    return rounds, objectives


def find_best_run(experiment: str, algorithm: str) -> tuple[Path | None, float, int]:
    """Find the CSV with the lowest final objective for a given algorithm.

    Returns
    -------
    tuple[Path | None, float, int]
        Best file path, best learning rate, best K value.
    """
    pattern = f"{experiment}_{algorithm}_lr*_K*.csv"
    candidates = list(RESULTS_DIR.glob(pattern))
    if not candidates:
        return None, 0.0, 0

    best_path = None
    best_final = float("inf")
    best_lr = 0.0
    best_k = 0

    for path in candidates:
        _, objectives = load_csv(path)
        if not objectives:
            continue
        final = objectives[-1]
        if final < best_final:
            best_final = final
            best_path = path
            stem = path.stem
            parts = stem.split("_")
            for part in parts:
                if part.startswith("lr"):
                    best_lr = float(part[2:])
                if part.startswith("K"):
                    best_k = int(part[1:])

    return best_path, best_lr, best_k


def plot_best_comparison(experiment: str):
    """Plot the best run per algorithm on one figure.

    Mirrors the layout of Figure 3 and Figure 5 in the paper:
    objective value on the y-axis, communication rounds on the x-axis.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    plotted = False
    for algorithm, color in ALGORITHM_COLORS.items():
        path, lr, k = find_best_run(experiment, algorithm)
        if path is None:
            continue
        rounds, objectives = load_csv(path)
        label = f"{ALGORITHM_LABELS[algorithm]} (lr={lr}, K={k})"
        ax.plot(rounds, objectives, label=label, color=color, linewidth=2)
        plotted = True

    if not plotted:
        print(f"No results found for experiment: {experiment}")
        return

    ax.set_xlabel("Communication rounds", fontsize=13)
    ax.set_ylabel("Objective value", fontsize=13)
    ax.set_title(f"Objective vs rounds ({experiment.replace('_', ' ')})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = STATIC_DIR / f"{experiment}_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_lr_sweep(experiment: str, algorithm: str):
    """Plot all lr values for one algorithm to visualize the sweep."""
    pattern = f"{experiment}_{algorithm}_lr*_K*.csv"
    candidates = sorted(RESULTS_DIR.glob(pattern))
    if not candidates:
        print(f"No results for {algorithm} on {experiment}")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    colors = [cmap(i / max(len(candidates) - 1, 1)) for i in range(len(candidates))]

    for path, color in zip(candidates, colors):
        rounds, objectives = load_csv(path)
        lr_str = [p for p in path.stem.split("_") if p.startswith("lr")][0][2:]
        ax.plot(rounds, objectives, label=f"lr={lr_str}", color=color, linewidth=1.5)

    ax.set_xlabel("Communication rounds", fontsize=12)
    ax.set_ylabel("Objective value", fontsize=12)
    ax.set_title(
        f"{ALGORITHM_LABELS.get(algorithm, algorithm)} lr sweep ({experiment})",
        fontsize=13,
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    out_path = STATIC_DIR / f"{experiment}_{algorithm}_lr_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    """Parse CLI arguments and generate plots from grid search results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="simulation_I")
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Only plot the best run per algorithm (comparison figure).",
    )
    args = parser.parse_args()

    plot_best_comparison(args.experiment)

    if not args.best_only:
        for algorithm in ALGORITHM_COLORS:
            plot_lr_sweep(args.experiment, algorithm)


if __name__ == "__main__":
    main()
