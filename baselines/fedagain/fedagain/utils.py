"""fedagain: Helpers for seeding, devices, metrics and result files."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from flwr.app import MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords

from fedagain.strategy import CLIENT_ID_KEY

# Per-client identifiers that must not be averaged across clients
_NON_AVERAGED_KEYS = (CLIENT_ID_KEY,)

# Fixed color and label per strategy, so a strategy looks the same in every figure
STRATEGY_COLORS = {"fedavg": "#257de0", "fedagain": "#f47230", "fedmedian": "#00b894"}
STRATEGY_LABELS = {"fedavg": "FedAvg", "fedagain": "FedAgain", "fedmedian": "FedMedian"}


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def get_device() -> torch.device:
    """Return the best available device (CUDA, then Apple MPS, then CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def aggregate_train_metrics(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Weighted average of training metrics, dropping per-client identifiers."""
    aggregated = aggregate_metricrecords(records, weighting_metric_name)
    for key in _NON_AVERAGED_KEYS:
        if key in aggregated:
            del aggregated[key]
    return aggregated


def save_results(path: Path, results: dict) -> None:
    """Write ``results`` as JSON, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


def load_results(path: str | Path) -> dict:
    """Load a results JSON written by the ServerApp."""
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def corrupted_weight_share(results: dict, by: str = "weight") -> list[float]:
    """Share of the aggregation weight given to corrupted clients in each round.

    ``by="weight"`` uses the FedAgain trust weights and ``by="num_examples"`` the
    data-size weights FedAvg would give to the same participants of that round.
    """
    corrupted = set(results["corrupted_clients"])
    shares = []
    for entry in results.get("trust_history", []):
        values = dict(zip(entry["client_ids"], entry[by], strict=True))
        total = sum(values.values())
        bad = sum(v for cid, v in values.items() if cid in corrupted)
        shares.append(float(bad / total) if total > 0 else float("nan"))
    return shares


def plot_results(result_files: list[str], output: str, title: str = "") -> None:
    """Plot centralized test accuracy per round for several result files.

    Each file is labelled with its strategy name. If a FedAgain run is present, a
    second panel shows the share of the aggregation weight given to corrupted
    clients, next to their share under data-size (FedAvg) weighting.
    """
    # pylint: disable=import-outside-toplevel
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [load_results(f) for f in result_files]
    fedagain_runs = [r for r in runs if r.get("trust_history")]
    ncols = 2 if fedagain_runs else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4), squeeze=False)

    ax = axes[0][0]
    for run in runs:
        rounds = [int(r) for r in run["test_accuracy"]]
        accs = [100 * run["test_accuracy"][str(r)] for r in rounds]
        strategy = run["config"]["strategy"]
        ax.plot(
            rounds,
            accs,
            marker="o",
            linewidth=2,
            color=STRATEGY_COLORS.get(strategy),
            label=STRATEGY_LABELS.get(strategy, strategy),
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Centralized test accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend()

    if fedagain_runs:
        ax = axes[0][1]
        run = fedagain_runs[0]
        shares = corrupted_weight_share(run)
        rounds = [e["round"] for e in run["trust_history"]]
        ax.plot(
            rounds,
            [100 * s for s in shares],
            marker="o",
            linewidth=2,
            color=STRATEGY_COLORS["fedagain"],
            label="FedAgain trust weights",
        )
        # Data-size reference of the same participants, round by round
        ax.plot(
            rounds,
            [100 * s for s in corrupted_weight_share(run, by="num_examples")],
            linestyle="--",
            color="gray",
            label="data-size weighting",
        )
        ax.set_xlabel("Round")
        ax.set_ylabel("Weight of corrupted clients (%)")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
