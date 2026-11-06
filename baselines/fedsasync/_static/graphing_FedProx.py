import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# 1.1 EVALUATION LOSS: Line Plot vs Real Time
# ========================================== 
def plot_eval_loss_lines_split_fs(
    base_path: str,
    datasets: list,
    data_dists: list,
    strategies: list,
    fraction_slows: list,
    m_values: list,
    prox_values: list,
    n_cols: int = 3,
) -> None:
    """
    Generates Evaluation Loss vs Time plots split into subplots by proximal_mu value.
    Each subplot shows and compares the lines corresponding to different values of M.
    """
    for dataset in datasets:
        for data_dist in data_dists:
            dataset_dist_name = f"{dataset}_{data_dist}"
            target_dir = os.path.join(base_path, dataset_dist_name)

            if not os.path.exists(target_dir):
                print(f"Directory not found: {target_dir}. Skipping...")
                continue

            # Get available subdirectories in the path (e.g., FedProxSaSync_0.0_fs0.2_m14)
            available_subdirs = [
                d for d in os.listdir(target_dir)
                if os.path.isdir(os.path.join(target_dir, d))
            ]

            for fs in fraction_slows:
                for strat in strategies:
                    n_prox = len(prox_values)
                    n_rows = math.ceil(n_prox / n_cols)

                    fig, axes = plt.subplots(
                        n_rows,
                        n_cols,
                        figsize=(4.5 * n_cols, 3.8 * n_rows),
                        sharey=True,
                    )
                    axes = np.atleast_1d(axes).flatten()

                    # Hide unused subplots if n_prox < n_rows * n_cols
                    for idx in range(n_prox, len(axes)):
                        fig.delaxes(axes[idx])

                    # Iterate over each proximal_mu value (one subplot per value)
                    for p_idx, prox in enumerate(prox_values):
                        ax = axes[p_idx]

                        # In the current subplot, iterate over each M to compare lines
                        for m in m_values:
                            # Pattern matching search: {strategy}_{prox}_fs{fs}_m{m}
                            expected_prefix = f"{strat}_{prox}"
                            expected_suffix = f"_fs{fs}_m{m}"

                            matched_folder = None
                            for subdir in available_subdirs:
                                if subdir.startswith(expected_prefix) and subdir.endswith(expected_suffix):
                                    matched_folder = subdir
                                    break

                            if matched_folder:
                                csv_path = os.path.join(target_dir, matched_folder, "ex1.csv")

                                if os.path.exists(csv_path):
                                    df = pd.read_csv(csv_path)

                                    if "eval_loss" in df.columns and "time" in df.columns:
                                        df = df.sort_values(by="time")
                                        raw_time = df["time"].values
                                        loss_vals = df["eval_loss"].values

                                        if len(loss_vals) > 0:
                                            # Normalize time to start at t = 0
                                            time_vals = raw_time - raw_time[0]

                                            ax.plot(
                                                time_vals,
                                                loss_vals,
                                                label=f"FedProxSaSync (M = {m})",
                                                linewidth=1.8,
                                                zorder=3,
                                            )

                        ax.set_title(f"$\\mu$ (proximal_mu) = {prox}", fontsize=11, pad=8)
                        ax.set_xlabel("Elapsed time (s)", fontsize=10)

                        if p_idx % n_cols == 0:
                            ax.set_ylabel("Evaluation Loss", fontsize=10)

                        ax.set_xlim(left=-2)
                        ax.grid(True, linestyle="--", alpha=0.5)

                    # Build common bottom legend with M values
                    handles, labels = axes[0].get_legend_handles_labels()
                    if handles:
                        fig.legend(
                            handles,
                            labels,
                            loc="lower center",
                            ncol=min(len(m_values), 5),
                            fontsize=9,
                            bbox_to_anchor=(0.5, 0.01),
                            frameon=True,
                        )

                    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

                    output_img = os.path.join(
                        target_dir,
                        f"eval_loss_{strat}_fs{fs}_{dataset_dist_name}.pdf"
                    )
                    plt.savefig(output_img, format="pdf", bbox_inches="tight")
                    plt.close()
                    print(f"Plot saved: {output_img}")


# ==========================================
# 1.2 EVALUATION ACCURACY: Line Plot vs Real Time
# ==========================================
def plot_eval_acc_lines_split_fs(
    base_path: str,
    datasets: list,
    data_dists: list,
    strategies: list,
    fraction_slows: list,
    m_values: list,
    prox_values: list,
    n_cols: int = 3,
) -> None:
    """
    Generates Evaluation Accuracy vs Time plots split into subplots by proximal_mu value.
    Each subplot shows and compares the lines corresponding to different values of M.
    """
    for dataset in datasets:
        for data_dist in data_dists:
            dataset_dist_name = f"{dataset}_{data_dist}"
            target_dir = os.path.join(base_path, dataset_dist_name)

            if not os.path.exists(target_dir):
                print(f"Directory not found: {target_dir}. Skipping...")
                continue

            # Get available subdirectories in the path (e.g., FedProxSaSync_0.0_fs0.2_m14)
            available_subdirs = [
                d for d in os.listdir(target_dir)
                if os.path.isdir(os.path.join(target_dir, d))
            ]

            for fs in fraction_slows:
                for strat in strategies:
                    n_prox = len(prox_values)
                    n_rows = math.ceil(n_prox / n_cols)

                    fig, axes = plt.subplots(
                        n_rows,
                        n_cols,
                        figsize=(4.5 * n_cols, 3.8 * n_rows),
                        sharey=True,
                    )
                    axes = np.atleast_1d(axes).flatten()

                    # Hide unused subplots if n_prox < n_rows * n_cols
                    for idx in range(n_prox, len(axes)):
                        fig.delaxes(axes[idx])

                    # Iterate over each proximal_mu value (one subplot per value)
                    for p_idx, prox in enumerate(prox_values):
                        ax = axes[p_idx]

                        # In the current subplot, iterate over each M to compare lines
                        for m in m_values:
                            expected_prefix = f"{strat}_{prox}"
                            expected_suffix = f"_fs{fs}_m{m}"

                            matched_folder = None
                            for subdir in available_subdirs:
                                if subdir.startswith(expected_prefix) and subdir.endswith(expected_suffix):
                                    matched_folder = subdir
                                    break

                            if matched_folder:
                                csv_path = os.path.join(target_dir, matched_folder, "ex1.csv")

                                if os.path.exists(csv_path):
                                    df = pd.read_csv(csv_path)

                                    if "eval_acc" in df.columns and "time" in df.columns:
                                        df = df.sort_values(by="time")
                                        raw_time = df["time"].values
                                        acc_vals = df["eval_acc"].values

                                        if len(acc_vals) > 0:
                                            # Convert ratio to percentage if in [0, 1] format
                                            if np.max(acc_vals) <= 1.0:
                                                acc_vals = acc_vals * 100.0

                                            # Normalize time to start at t = 0
                                            time_vals = raw_time - raw_time[0]

                                            ax.plot(
                                                time_vals,
                                                acc_vals,
                                                label=f"FedProxSaSync (M = {m})",
                                                linewidth=1.8,
                                                zorder=3,
                                            )

                        ax.set_title(f"$\\mu$ (proximal_mu) = {prox}", fontsize=11, pad=8)
                        ax.set_xlabel("Elapsed time (s)", fontsize=10)

                        if p_idx % n_cols == 0:
                            ax.set_ylabel("Evaluation Accuracy (%)", fontsize=10)

                        ax.set_xlim(left=-2)
                        ax.grid(True, linestyle="--", alpha=0.5)

                    # Build common bottom legend with M values
                    handles, labels = axes[0].get_legend_handles_labels()
                    if handles:
                        fig.legend(
                            handles,
                            labels,
                            loc="lower center",
                            ncol=min(len(m_values), 5),
                            fontsize=9,
                            bbox_to_anchor=(0.5, 0.01),
                            frameon=True,
                        )

                    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

                    output_img = os.path.join(
                        target_dir,
                        f"eval_acc_{strat}_fs{fs}_{dataset_dist_name}.pdf"
                    )
                    plt.savefig(output_img, format="pdf", bbox_inches="tight")
                    plt.close()
                    print(f"Plot saved: {output_img}")


if __name__ == "__main__":
    BASE_PATH = "_static"
    DATASETS = ["cifar10"]
    DATA_DISTS = ["iid"]
    STRATEGIES = ["FedProxSaSync"]
    FRACTION_SLOWS = [0.2]
    M_VALUES = [14, 16, 18, 20]
    PROX_VALUES = [0.0, 0.1, 0.2]

    # Execute all graphing scripts
    plot_eval_loss_lines_split_fs(BASE_PATH, DATASETS, DATA_DISTS, STRATEGIES, FRACTION_SLOWS, M_VALUES, PROX_VALUES, 3)
    plot_eval_acc_lines_split_fs(BASE_PATH, DATASETS, DATA_DISTS, STRATEGIES, FRACTION_SLOWS, M_VALUES, PROX_VALUES, 3)