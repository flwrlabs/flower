"""fedagain: Build the figures and the results table shown in the README.

Usage (from the baseline directory, after ``run_experiments.sh``)::

    python -m fedagain.plot --dataset cifar10 --seed 42
"""

import argparse
from pathlib import Path

from fedagain.utils import STRATEGY_COLORS, STRATEGY_LABELS, load_results, plot_results

# (label, num_clients, corrupted_fraction, severity) of the paper's Fig. 11
CORRUPTION_SCENARIOS = [
    ("Light", 10, "0.1", 3),
    ("Medium", 10, "0.3", 4),
    ("Heavy", 10, "0.4", 5),
]
SCALABILITY_SCENARIOS = [
    ("5 clients", 5, "0.3", 3),
    ("10 clients", 10, "0.3", 3),
    ("20 clients", 20, "0.3", 3),
]


def result_file(outputs: Path, args, clients: int, rho: str, sev: int, strategy: str):
    """Return the path of the JSON written by the ServerApp for one run."""
    return outputs / (
        f"{args.dataset}_{args.partitioning}_n{clients}_rho{rho}_sev{sev}"
        f"_{strategy}_seed{args.seed}.json"
    )


def final_accuracy(path: Path) -> float:
    """Return the final centralized test accuracy (%) of the global model."""
    accs = load_results(path)["test_accuracy"]
    return 100 * accs[str(max(int(r) for r in accs))]


def _bar_panel(ax, scenarios, outputs, args, strategies, title):
    """Draw grouped bars of final accuracy, one group per scenario."""
    width = 0.8 / len(strategies)
    for i, strategy in enumerate(strategies):
        xs, ys = [], []
        for j, (_, clients, rho, sev) in enumerate(scenarios):
            path = result_file(outputs, args, clients, rho, sev, strategy)
            if path.exists():
                xs.append(j + (i - (len(strategies) - 1) / 2) * width)
                ys.append(final_accuracy(path))
        bars = ax.bar(
            xs,
            ys,
            width=width * 0.92,  # small surface gap between adjacent bars
            color=STRATEGY_COLORS[strategy],
            label=STRATEGY_LABELS[strategy],
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2, color="#333333")
    ax.set_xticks(range(len(scenarios)), [s[0] for s in scenarios])
    ax.set_ylabel("Final test accuracy (%)")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_figure11(outputs: Path, args, strategies: list[str], output: Path) -> None:
    """Plot final accuracies with the layout of Fig. 11 of the paper."""
    # pylint: disable=import-outside-toplevel
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _bar_panel(
        axes[0],
        CORRUPTION_SCENARIOS,
        outputs,
        args,
        strategies,
        "Corruption: light (10%, s3), medium (30%, s4), heavy (40%, s5)",
    )
    _bar_panel(
        axes[1],
        SCALABILITY_SCENARIOS,
        outputs,
        args,
        strategies,
        "Scalability: 30% corrupted clients, severity 3",
    )
    fig.suptitle(f"{args.dataset.upper()}, label skew, seed {args.seed}")
    # One shared legend below the panels, so it never covers the bars
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def markdown_table(outputs: Path, args, strategies: list[str]) -> str:
    """Build a Markdown table of final accuracies and corrupted-client weights."""
    header = (
        "| Scenario | Clients | Corrupted | Severity | "
        + " | ".join(STRATEGY_LABELS[s] for s in strategies)
        + " | Weight of corrupted clients (FedAgain / data-size) |"
    )
    lines = [
        header,
        "| --- | ---: | ---: | ---: | " + "---: | " * len(strategies) + "--- |",
    ]
    for label, clients, rho, sev in CORRUPTION_SCENARIOS + SCALABILITY_SCENARIOS:
        accs = []
        for strategy in strategies:
            path = result_file(outputs, args, clients, rho, sev, strategy)
            accs.append(f"{final_accuracy(path):.2f}" if path.exists() else "-")
        share = "-"
        path = result_file(outputs, args, clients, rho, sev, "fedagain")
        if path.exists():
            res = load_results(path)
            if res.get("fedagain_corrupted_share"):
                share = (
                    f"{100 * res['fedagain_corrupted_share'][-1]:.1f}% / "
                    f"{100 * res['fedavg_corrupted_share']:.1f}%"
                )
        lines.append(
            f"| {label} | {clients} | {float(rho):.0%} | {sev} | "
            + " | ".join(accs)
            + f" | {share} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Generate the figures and print the results table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--partitioning", default="label-skew")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--static", default="_static")
    args = parser.parse_args()

    outputs, static = Path(args.outputs), Path(args.static)
    strategies = [
        s
        for s in ("fedavg", "fedmedian", "fedagain")
        if any(outputs.glob(f"{args.dataset}_*_{s}_seed{args.seed}.json"))
    ]
    plot_figure11(outputs, args, strategies, static / f"{args.dataset}_fig11.png")

    # Per-round curves of the medium scenario, with FedAgain's trust weights
    _, clients, rho, sev = CORRUPTION_SCENARIOS[1]
    files = [result_file(outputs, args, clients, rho, sev, s) for s in strategies]
    files = [f for f in files if f.exists()]
    if files:
        plot_results(
            [str(f) for f in files],
            output=str(static / f"{args.dataset}_medium_rounds.png"),
            title=f"{args.dataset.upper()}, medium corruption (30%, severity 4)",
        )
    print(markdown_table(outputs, args, strategies))


if __name__ == "__main__":
    main()
