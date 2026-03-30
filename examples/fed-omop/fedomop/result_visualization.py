import json
import matplotlib.pyplot as plt
import os


def plot_metrics(path):
    with open(path) as f:
        data = json.load(f)

    # config
    n_clients = data["number_of_nodes"]
    partition = data["run_config"]["partitioner"]
    strategy  = data["run_config"]["strategy"]
    lr        = data["run_config"]["lr"]
    epochs    = data["run_config"]["local-epochs"]

    # data
    rounds = [r["round"] for r in data["round_res"]]
    auroc  = [r["evaluate_metrics_clientapp"]["auroc"] for r in data["round_res"]]
    auprc  = [r["evaluate_metrics_clientapp"]["auprc"] for r in data["round_res"]]
    acc    = [r["evaluate_metrics_clientapp"]["accuracy"] for r in data["round_res"]]

    # plot
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, auroc, marker='o', linewidth=2, label="AUROC")
    plt.plot(rounds, auprc, marker='s', linewidth=2, label="AUPRC")
    plt.plot(rounds, acc,   marker='^', linewidth=2, label="Accuracy")

    # one-line styled title
    plt.title(
        rf"$\bf{{Nodes}}$: {n_clients} | "
        rf"$\bf{{Strategy}}$: {strategy} | "
        rf"$\bf{{Partition}}$: {partition.upper()} | "
        rf"$\bf{{LR}}$: {lr} | "
        rf"$\bf{{Local\ Epochs}}$: {epochs}",
        fontsize=11,
        pad=10
    )

    plt.xlabel("Round")
    plt.ylabel("Score")
    plt.grid(alpha=0.3)
    plt.legend(frameon=False)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # filename
    base = os.path.splitext(os.path.basename(path))[0]
    filename = f"plot_{base}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")