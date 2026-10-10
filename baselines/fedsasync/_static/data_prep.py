import os
import glob
import json
import numpy as np
import pandas as pd


def aggregate_and_save_config(
    base_path: str,
    datasets: list,
    data_dists: list,
    strategies: list,
    fraction_slows: list,
    m_values: list,
):
    """Iterates over the experimental matrix, aggregates metrics across all execution runs,

    and exports summarized results to 'agg.csv' and 'agg.json'.

    Parameters
    ----------
    base_path : str
        Root directory where experimental results are stored.
    datasets : list
        List of dataset identifiers (e.g., ['cifar10']).
    data_dists : list
        List of data distribution strategies (e.g., ['dirichlet', 'iid']).
    strategies : list
        List of federated learning strategies (e.g., ['FedAvg', 'FedSaSync']).
    fraction_slows : list
        List of straggler fractions tested in the experiments.
    m_values : list
        List of concurrency/staleness threshold parameters (M) for semi-asynchronous strategies.
    """

    for dataset in datasets:
        for data_dist in data_dists:
            for strategy in strategies:
                for fs in fraction_slows:

                    # Determine parameter grid for 'M' based on the strategy
                    if strategy == "FedAvg":
                        config_folder_name = f"{strategy}_fs{fs}"
                        current_m_list = [None]
                        
                    else:
                        current_m_list = m_values

                    for m in current_m_list:
                        # Construct directory path according to naming conventions
                        dataset_dist_name = f"{dataset}_{data_dist}"

                        if m is not None:
                            config_dir = os.path.join(
                                base_path,
                                dataset_dist_name,
                                f"{strategy}_fs{fs}_m{m}",
                            )
                            config_label = f"{dataset_dist_name} | {strategy} (fs={fs}, M={m})"
                        else:
                            config_dir = os.path.join(
                                base_path, dataset_dist_name, config_folder_name
                            )
                            config_label = (
                                f"{dataset_dist_name} | {strategy} (fs={fs})"
                            )

                        if not os.path.exists(config_dir):
                            continue

                        print(
                            f"Processing and generating aggregated metrics for: {config_label}..."
                        )

                        csv_files = sorted(
                            glob.glob(os.path.join(config_dir, "ex*.csv"))
                        )
                        json_files = sorted(
                            glob.glob(
                                os.path.join(config_dir, "summary_ex*.json")
                            )
                        )

                        if not csv_files:
                            continue

                        # ======================================================
                        # 1. PROCESS AND SAVE CSV (Per-round time series data)
                        # ======================================================
                        all_dfs = []
                        for file in csv_files:
                            df = pd.read_csv(file)
                            df["round"] = range(1, len(df) + 1)
                            all_dfs.append(df)

                        combined_df = pd.concat(all_dfs)

                        mean_df = combined_df.groupby("round").mean()
                        std_df = combined_df.groupby("round").std().fillna(0)

                        agg_df = pd.DataFrame(index=mean_df.index)

                        for col in mean_df.columns:
                            agg_df[f"{col}_mean"] = mean_df[col].values
                            agg_df[f"{col}_std"] = std_df[col].values

                        agg_csv_path = os.path.join(config_dir, "agg.csv")
                        agg_df.to_csv(
                            agg_csv_path, index=True, index_label="round"
                        )

                        # ======================================================
                        # 2. EVALUATE TTA, ACC@TIME, AND LOSS/SEC FROM RUN CSVs
                        # ======================================================
                        target_accs = [50.0, 58.0]
                        tta_per_target = {target: [] for target in target_accs}

                        target_times = [15.0, 25.0]
                        acc_at_time_per_target = {t: [] for t in target_times}

                        loss_per_sec_runs = []
                        total_valid_runs = 0

                        for file in csv_files:
                            df_ex = pd.read_csv(file)

                            # Validate schema against required metric columns
                            if (
                                "eval_acc" in df_ex.columns
                                and "eval_loss" in df_ex.columns
                                and "time" in df_ex.columns
                            ):
                                total_valid_runs += 1

                                # Normalize accuracy scale to percentage [0, 100]
                                acc_values = df_ex["eval_acc"].values.copy()
                                if acc_values.max() <= 1.0:
                                    acc_values = acc_values * 100.0

                                loss_values = df_ex["eval_loss"].values.copy()
                                cumulative_time = df_ex["time"].values

                                if len(cumulative_time) == 0:
                                    total_valid_runs -= 1
                                    continue

                                base_time = cumulative_time[0]

                                # Net Time-To-Accuracy (TTA) computation
                                for target in target_accs:
                                    indices = np.where(acc_values >= target)[0]
                                    if len(indices) > 0:
                                        net_tta_val = (
                                            cumulative_time[indices[0]]
                                            - base_time
                                        )
                                        net_tta_val = max(0.0, net_tta_val)
                                        tta_per_target[target].append(
                                            float(net_tta_val)
                                        )

                                # Accuracy at specific wall-clock time intervals (Acc@time)
                                for t_target in target_times:
                                    effective_target = base_time + t_target
                                    valid_indices = np.where(
                                        cumulative_time <= effective_target
                                    )[0]
                                    if len(valid_indices) > 0:
                                        idx = valid_indices[-1]
                                        acc_at_time_per_target[t_target].append(
                                            float(acc_values[idx])
                                        )

                                # Convergence rate metric (Final Loss / Total Wall-Clock Time)
                                if (
                                    len(loss_values) > 0
                                    and cumulative_time[-1] > 0
                                ):
                                    final_loss = float(loss_values[-1])
                                    total_time = float(cumulative_time[-1])
                                    loss_per_sec_runs.append(
                                        final_loss / total_time
                                    )

                        # ======================================================
                        # 3. PROCESS JSON RUN SUMMARIES & SCHEDULER METRICS
                        # ======================================================
                        all_round_staleness = []
                        client_part_dict = {}
                        client_stal_dict = {}

                        scheduler_target_keys = [
                            "polling_calls_count",
                            "time_in_pull_sec",
                            "server_cpu_time_sec",
                            "messages_received_count",
                            "total_time",
                        ]
                        scheduler_runs_data = {
                            col: [] for col in scheduler_target_keys
                        }

                        for file in json_files:
                            with open(file, "r", encoding="utf-8") as f:
                                data = json.load(f)

                            all_round_staleness.append(
                                data.get("round_staleness", [])
                            )

                            for client_id, val in data.get(
                                "client_participation", {}
                            ).items():
                                client_part_dict.setdefault(
                                    client_id, []
                                ).append(val)

                            for client_id, val in data.get(
                                "client_staleness", {}
                            ).items():
                                client_stal_dict.setdefault(
                                    client_id, []
                                ).append(val)

                            # Extract scheduler performance metrics from individual JSON execution files
                            metrics_block = data.get("metrics", {})
                            for col in scheduler_target_keys:
                                if col in metrics_block:
                                    val = metrics_block[col]
                                    if val is not None and not pd.isna(val):
                                        scheduler_runs_data[col].append(
                                            float(val)
                                        )

                        agg_json_data = {}

                        # Aggregate round-level staleness metrics
                        if all_round_staleness:
                            arr_staleness = np.array(all_round_staleness)
                            agg_json_data["round_staleness_mean"] = (
                                np.mean(arr_staleness, axis=0).tolist()
                            )
                            agg_json_data["round_staleness_std"] = (
                                np.std(arr_staleness, axis=0).tolist()
                            )

                        # Aggregate client-level participation metrics
                        agg_json_data["client_participation"] = {
                            cid: {
                                "mean": float(np.mean(vals)),
                                "std": float(np.std(vals)),
                            }
                            for cid, vals in client_part_dict.items()
                        }

                        # Aggregate client-level staleness metrics
                        agg_json_data["client_staleness"] = {
                            cid: {
                                "mean": float(np.mean(vals)),
                                "std": float(np.std(vals)),
                            }
                            for cid, vals in client_stal_dict.items()
                        }

                        # Aggregate Net TTA metrics
                        tta_metrics = {}
                        for target in target_accs:
                            values = tta_per_target[target]
                            success_count = len(values)
                            tta_metrics[f"tta{int(target)}_success_count"] = (
                                success_count
                            )
                            tta_metrics[f"tta{int(target)}_total_runs"] = (
                                total_valid_runs
                            )

                            if values:
                                tta_metrics[f"tta{int(target)}_mean"] = float(
                                    np.mean(values)
                                )
                                tta_metrics[f"tta{int(target)}_std"] = float(
                                    np.std(values)
                                )
                            else:
                                tta_metrics[f"tta{int(target)}_mean"] = None
                                tta_metrics[f"tta{int(target)}_std"] = 0.0
                        agg_json_data["tta_metrics"] = tta_metrics

                        # Aggregate Acc@time metrics
                        acc_at_time_metrics = {}
                        for t_target in target_times:
                            values = acc_at_time_per_target[t_target]
                            success_count = len(values)
                            key_name = f"acc_at_{int(t_target)}s"

                            acc_at_time_metrics[
                                f"{key_name}_success_count"
                            ] = success_count
                            acc_at_time_metrics[f"{key_name}_total_runs"] = (
                                total_valid_runs
                            )

                            if values:
                                acc_at_time_metrics[f"{key_name}_mean"] = float(
                                    np.mean(values)
                                )
                                acc_at_time_metrics[f"{key_name}_std"] = float(
                                    np.std(values)
                                )
                            else:
                                acc_at_time_metrics[f"{key_name}_mean"] = None
                                acc_at_time_metrics[f"{key_name}_std"] = 0.0
                        agg_json_data["acc_at_time_metrics"] = (
                            acc_at_time_metrics
                        )

                        # Aggregate Loss/sec metrics
                        loss_per_sec_metrics = {}
                        if loss_per_sec_runs:
                            loss_per_sec_metrics["loss_per_sec_mean"] = float(
                                np.mean(loss_per_sec_runs)
                            )
                            loss_per_sec_metrics["loss_per_sec_std"] = float(
                                np.std(loss_per_sec_runs)
                            )
                        else:
                            loss_per_sec_metrics["loss_per_sec_mean"] = None
                            loss_per_sec_metrics["loss_per_sec_std"] = 0.0
                        agg_json_data["loss_per_sec_metrics"] = (
                            loss_per_sec_metrics
                        )

                        # Aggregate Scheduler and Communication Overhead metrics
                        scheduler_metrics = {}
                        for col in scheduler_target_keys:
                            vals = scheduler_runs_data[col]
                            if vals:
                                scheduler_metrics[f"{col}_mean"] = float(
                                    np.mean(vals)
                                )
                                scheduler_metrics[f"{col}_std"] = float(
                                    np.std(vals, ddof=1)
                                    if len(vals) > 1
                                    else 0.0
                                )
                            else:
                                scheduler_metrics[f"{col}_mean"] = None
                                scheduler_metrics[f"{col}_std"] = 0.0
                        agg_json_data["scheduler_metrics"] = scheduler_metrics

                        # Save aggregated JSON summary file
                        agg_json_path = os.path.join(config_dir, "agg.json")
                        with open(agg_json_path, "w", encoding="utf-8") as f:
                            json.dump(agg_json_data, f, indent=4)

                        print(
                            f"  -> Successfully generated: {agg_csv_path} and {agg_json_path}"
                        )


if __name__ == "__main__":
    BASE_PATH = "_static"
    DATASETS = ["cifar10", "mnist"]
    DATA_DISTS = ["dirichlet", "iid"]
    STRATEGIES = ["FedAvg", "FedSaSync"]
    FRACTION_SLOWS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    M_VALUES = [8, 10, 12, 14, 16, 18, 20]

    aggregate_and_save_config(
        BASE_PATH,
        DATASETS,
        DATA_DISTS,
        STRATEGIES,
        FRACTION_SLOWS,
        M_VALUES,
    )