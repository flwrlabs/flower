#!/usr/bin/env python3
"""Turn the sweep output into the figures a Flower baseline README needs.

Usage
-----
    python plot_results.py --results results --out docs

Produces four figures:

1. ``accuracy_vs_round.png``    — the headline curve, mean +/- std over seeds.
   Answers "does the MAC gate hurt convergence, and does energy-aware selection
   recover it?"
2. ``accuracy_vs_energy.png``   — accuracy against *cumulative millijoules*.
   This is the plot that actually makes the argument: rounds are free, joules
   are not. An arm that converges slower per round but cheaper per joule wins.
3. ``participation.png``        — active vs skipped clients per round. Makes the
   all-depleted rounds legible instead of looking like a crashed run.
4. ``energy_per_round.png``     — CR vs NCR energy draw.

The JSON reader is deliberately tolerant: it walks the whole document looking
for per-round metric series, so it keeps working if the Flower Result object
changes shape.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Substrings used to locate a series inside an arbitrarily nested JSON blob.
WANTED = {
    "accuracy": ("eval_acc", "accuracy"),
    "train_loss": ("train_loss", "trainloss"),
    "energy_used_mj": ("energy_used_mj", "energy"),
    "skipped_clients": ("skipped_clients", "skipped"),
    "active_clients": ("active_clients",),
    "bandwidth_frac": ("bandwidth_frac", "bandwidth"),
}

ARM_STYLE = {
    "fedavg_no_dsme": ("#444444", "--", "FedAvg (no DSME gate)"),
    "dsme_random": ("#c1440e", "-", "DSME + random selection"),
    "dsme_eligible": ("#1f6feb", "-", "DSME + energy-aware"),
    "dsme_greedy": ("#2a9d5c", "-", "DSME + greedy residual"),
    "dsme_proportional": ("#7d4bd4", "-", "DSME + proportional"),
    "dsme_eligible_harvest": ("#d9a400", "-.", "DSME + energy-aware + harvest"),
}


def walk(node, path=()):
    """Yield (path, value) for every leaf in a nested dict/list."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (str(key),))
    elif isinstance(node, list):
        yield path, node
    else:
        yield path, node


def extract_series(blob: dict) -> dict[str, dict[int, float]]:
    """Pull per-round numeric series out of a run's JSON.

    Tolerant of where the round number sits in the structure. All of these work:

        {"rounds": {"train": {"1": {"train_loss": 2.2}}}}   <- round above metric
        {"accuracy": {"1": 0.11, "2": 0.14}}                <- round below metric
        {"accuracy": [[1, 0.11], [2, 0.14]]}                <- pairs
    """
    series: dict[str, dict[int, float]] = defaultdict(dict)

    def record(name: str, rnd, val) -> None:
        try:
            series[name][int(rnd)] = float(val)
        except (TypeError, ValueError):
            pass

    def round_from_path(path) -> int | None:
        """Nearest integer-looking element, searching from the leaf upwards."""
        for part in reversed(path):
            if part.lstrip("-").isdigit():
                return int(part)
        return None

    for path, value in walk(blob):
        joined = "/".join(path).lower()
        for name, keys in WANTED.items():
            if not any(k in joined for k in keys):
                continue
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        record(name, item[0], item[1])
                    elif isinstance(item, (int, float)):
                        record(name, len(series[name]) + 1, item)
            else:
                rnd = round_from_path(path)
                if rnd is not None:
                    record(name, rnd, value)
            break
    return {k: v for k, v in series.items() if v}


def load(results_dir: Path) -> dict[str, list[dict[str, dict[int, float]]]]:
    index = json.loads((results_dir / "index.json").read_text())
    by_arm: dict[str, list] = defaultdict(list)
    for run in index["runs"]:
        if run.get("status") != "ok":
            continue
        path = Path(run["json"])
        if not path.exists():
            continue
        by_arm[run["arm"]].append(extract_series(json.loads(path.read_text())))
    return by_arm


def stack(runs: list[dict], name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rounds, mean, std) across seeds for one metric."""
    present = [r[name] for r in runs if name in r]
    if not present:
        return np.array([]), np.array([]), np.array([])
    rounds = sorted(set.intersection(*[set(p) for p in present]))
    matrix = np.array([[p[r] for r in rounds] for p in present], dtype=float)
    return np.array(rounds), matrix.mean(axis=0), matrix.std(axis=0)


def band(ax, x, mean, std, arm, marker=None):
    color, ls, label = ARM_STYLE.get(arm, ("#888888", "-", arm))
    ax.plot(x, mean, ls, color=color, label=label, marker=marker, markersize=3)
    if std.size and std.max() > 0:
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, lw=0)


def fig_accuracy(by_arm, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for arm, runs in by_arm.items():
        x, m, s = stack(runs, "accuracy")
        if x.size:
            band(ax, x, m * (100 if m.max() <= 1.0 else 1), s * (100 if m.max() <= 1.0 else 1), arm)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Global test accuracy (%)")
    ax.set_title("CIFAR-10 non-IID, 10 clients (mean ± std over seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_round.png", dpi=160)
    plt.close(fig)


def fig_accuracy_vs_energy(by_arm, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for arm, runs in by_arm.items():
        xr, acc, _ = stack(runs, "accuracy")
        _, energy, _ = stack(runs, "energy_used_mj")
        if not xr.size or not energy.size:
            continue
        n = min(len(acc), len(energy))
        cumulative = np.cumsum(energy[:n])
        color, ls, label = ARM_STYLE.get(arm, ("#888888", "-", arm))
        ax.plot(
            cumulative,
            acc[:n] * (100 if acc.max() <= 1.0 else 1),
            ls,
            color=color,
            label=label,
            marker="o",
            markersize=3,
        )
    ax.set_xlabel("Cumulative radio+compute energy (mJ)")
    ax.set_ylabel("Global test accuracy (%)")
    ax.set_title("Accuracy per joule — the metric that matters on 802.15.4")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "accuracy_vs_energy.png", dpi=160)
    plt.close(fig)


def fig_participation(by_arm, out: Path):
    arms = [a for a in by_arm if any("skipped_clients" in r for r in by_arm[a])]
    if not arms:
        return
    fig, axes = plt.subplots(
        len(arms), 1, figsize=(6.5, 2.1 * len(arms)), sharex=True, squeeze=False
    )
    for ax, arm in zip(axes[:, 0], arms):
        x, active, _ = stack(by_arm[arm], "active_clients")
        xs, skipped, _ = stack(by_arm[arm], "skipped_clients")
        if not xs.size:
            continue
        if not x.size:
            x, active = xs, np.zeros_like(skipped)
        n = min(len(x), len(skipped))
        ax.bar(x[:n], active[:n], color="#1f6feb", label="active")
        ax.bar(x[:n], skipped[:n], bottom=active[:n], color="#c1440e", label="energy-depleted")
        ax.set_ylabel("clients")
        ax.set_title(ARM_STYLE.get(arm, ("", "", arm))[2], fontsize=9, loc="left")
        ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("Federated round")
    fig.tight_layout()
    fig.savefig(out / "participation.png", dpi=160)
    plt.close(fig)


def fig_energy(by_arm, out: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for arm, runs in by_arm.items():
        x, m, s = stack(runs, "energy_used_mj")
        if x.size:
            band(ax, x, m, s, arm)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Energy per participating client (mJ)")
    ax.set_title("MAC-layer energy draw (CR vs SeCAP NCR mode)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "energy_per_round.png", dpi=160)
    plt.close(fig)


def summary_table(by_arm) -> str:
    rows = ["| Arm | Final acc (%) | Peak acc (%) | Total energy (mJ) | Skipped rounds |",
            "|---|---|---|---|---|"]
    for arm, runs in by_arm.items():
        _, acc, acc_s = stack(runs, "accuracy")
        _, energy, _ = stack(runs, "energy_used_mj")
        _, skipped, _ = stack(runs, "skipped_clients")
        if not acc.size:
            continue
        scale = 100 if acc.max() <= 1.0 else 1
        dead = int((skipped > 0).sum()) if skipped.size else 0
        energy_cell = f"{energy.sum():.1f}" if energy.size else "n/a"
        label = ARM_STYLE.get(arm, ("", "", arm))[2]
        rows.append(
            f"| {label} "
            f"| {acc[-1] * scale:.1f} ± {acc_s[-1] * scale:.1f} "
            f"| {acc.max() * scale:.1f} "
            f"| {energy_cell} "
            f"| {dead} |"
        )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("docs"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    by_arm = load(args.results)
    if not by_arm:
        print("No successful runs found. Check results/index.json and the .log files.")
        return 1

    for arm, runs in by_arm.items():
        found = sorted({k for r in runs for k in r})
        print(f"{arm}: {len(runs)} seed(s), metrics found: {', '.join(found) or 'none'}")

    fig_accuracy(by_arm, args.out)
    fig_accuracy_vs_energy(by_arm, args.out)
    fig_participation(by_arm, args.out)
    fig_energy(by_arm, args.out)

    table = summary_table(by_arm)
    (args.out / "summary_table.md").write_text(table + "\n")
    print("\n" + table)
    print(f"\nFigures written to {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
