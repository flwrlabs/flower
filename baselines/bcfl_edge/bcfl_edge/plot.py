"""bcfl_edge: Plot test accuracy and F1 of finished runs.

    python -m bcfl_edge.plot outputs _static/accuracy.png [--selected-devices 40]

For each selection, uses the most recent run with the given S_D
(`outputs/<selection>_sd<S_D>_<time>.json`), so all curves share one budget.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

LABELS = {"bcfl": "Algorithm 1", "random": "Random selection"}


def load_runs(results_dir: Path, selected_devices: int) -> dict[str, dict]:
    """Return the most recent run of each selection with the given S_D."""
    runs = {}
    # File names end in a sortable timestamp, so later files are newer runs.
    for path in sorted(results_dir.glob(f"*_sd{selected_devices}_*.json")):
        run = json.loads(path.read_text())
        runs[run["run-config"]["selection"]] = run
    if not runs:
        raise SystemExit(f"No runs with selected-devices={selected_devices}.")
    return runs


def main() -> None:
    """Plot the runs and print a summary table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("figure", type=Path)
    parser.add_argument("--selected-devices", type=int, default=40)
    args = parser.parse_args()
    runs = load_runs(args.results_dir, args.selected_devices)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for selection, run in runs.items():
        rounds = sorted(int(r) for r in run["evaluation"])
        for ax, key in zip(axes, ["accuracy", "f1"], strict=False):
            values = [run["evaluation"][str(r)][key] for r in rounds]
            ax.plot(rounds, values, label=LABELS.get(selection, selection))
    for ax, label in zip(axes, ["Test accuracy", "Test F1 (macro)"], strict=False):
        ax.set(xlabel="Round", ylabel=label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.figure, dpi=150)

    print(
        "| Selection | Miners | A_fork | Round latency (s) | Round energy (J) "
        "| Final accuracy | Final F1 |"
    )
    print("|---|---|---|---|---|---|---|")
    for selection, run in runs.items():
        cost = run["round-cost"]
        last = run["evaluation"][str(max(int(r) for r in run["evaluation"]))]
        print(
            f"| {LABELS.get(selection, selection)} | {len(cost['selected-miners'])} "
            f"| {cost['a-fork']} | {cost['latency']:.0f} | {cost['energy']:.0f} "
            f"| {last['accuracy']:.3f} | {last['f1']:.3f} |"
        )


if __name__ == "__main__":
    main()
