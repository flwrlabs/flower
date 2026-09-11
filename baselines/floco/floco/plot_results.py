"""Generate plots from json files."""

import json
import os

import matplotlib.pyplot as plt

# Consistent ordering and colors for the legend
ALGORITHMS = ["FedAvg", "Floco", r"Floco$^{+}$"]
COLORS = {"FedAvg": "red", "Floco": "blue", r"Floco$^{+}$": "green"}


def make_plot(dir_path: str, dataset: str, split_name: str, title: str) -> None:
    """Generate a two-panel accuracy plot from all result JSONs in a directory."""
    results: dict[str, tuple[list[float], list[float]]] = {}

    for entry in os.scandir(dir_path):
        if not entry.name.endswith(".json"):
            continue
        with open(entry.path, encoding="UTF-8") as f:
            data = json.load(f)

        cfg = data["run_config"]
        if cfg["dataset"] != dataset or cfg["dataset-split"] != split_name:
            continue

        algorithm = cfg["algorithm"]
        if algorithm == "Floco" and cfg.get("pers_lamda", 0) > 0:
            algorithm = r"Floco$^{+}$"

        fed_acc = [r["accuracy"] * 100 for r in data["round_res"] if "accuracy" in r]
        cen_acc = [
            r["centralized_accuracy"] * 100
            for r in data["round_res"]
            if "centralized_accuracy" in r
        ]
        results[algorithm] = (fed_acc, cen_acc)

    _, ax = plt.subplots(1, 2, figsize=(8, 3))

    for alg in ALGORITHMS:
        if alg not in results:
            continue
        fed_acc, cen_acc = results[alg]
        print(f"Max federated accuracy ({alg}): {max(fed_acc):.2f}")
        ax[0].plot(cen_acc, color=COLORS[alg], label=alg)
        ax[1].plot(fed_acc, color=COLORS[alg], label=alg)

    ax[0].set(xlabel="Rounds", ylabel="Accuracy", ylim=(20,82), title="Centralized Test Accuracy")
    ax[1].set(xlabel="Rounds", ylabel="Accuracy", ylim=(20,82), title="Federated Test Accuracy")
    ax[0].legend()
    ax[1].legend()
    plt.tight_layout()

    save_path = os.path.join("_static/", f"{'_'.join(title.split())}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    DATASET = "CIFAR10"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(script_dir, "../results/")
    for split in ["Dirichlet", "Fold"]:
        make_plot(res_dir, DATASET, split, title=f"{DATASET} {split}")
