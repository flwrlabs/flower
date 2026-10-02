import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


# ==========================================
# 1. PARTICIPATION: Heatmap Version
# ==========================================
def plot_participation_heatmap_split_fs(
    base_path: str,
    datasets: list,
    data_dists: list,
    strategies: list,
    fraction_slows: list,
    m_values: list,
    n_cols: int,
) -> None:
    """Generate and save client participation heatmaps per dataset and data distribution."""
    ext = "svg" if len(fraction_slows) == 6 else "pdf"
    suffix = f"_complete.{ext}" if len(fraction_slows) == 6 else ".pdf"

    for dataset in datasets:
        for data_dist in data_dists:
            dataset_dist_name = f"{dataset}_{data_dist}"

            configs_meta = []
            if "FedSaSync" in strategies:
                for m in m_values:
                    configs_meta.append({"name": f"FedSaSync (M={m})", "type": "FedSaSync", "m": m})
            if "FedAvg" in strategies:
                configs_meta.append({"name": "FedAvg", "type": "FedAvg"})
                    
            max_clients = 0
            for fs in fraction_slows:
                for cfg in configs_meta:
                    d = os.path.join(
                        base_path,
                        dataset_dist_name,
                        f"FedAvg_fs{fs}" if cfg["type"] == "FedAvg" else f"FedSaSync_fs{fs}_m{cfg['m']}",
                    )
                    agg_path = os.path.join(d, "agg.json")
                    if os.path.exists(agg_path):
                        with open(agg_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        max_clients = max(max_clients, len(data.get("client_participation", {})))

            max_clients = max(max_clients, 10)
            
            n_fs = len(fraction_slows)
            n_rows = math.ceil(n_fs / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 0.8, 4.2 * n_rows), sharey=True)
            axes = np.atleast_1d(axes).flatten()
                
            for idx in range(n_fs, len(axes)):
                fig.delaxes(axes[idx])

            for fs_idx, fs in enumerate(fraction_slows):
                ax = axes[fs_idx]
                rows_data = {}
                
                for cfg in configs_meta:
                    d = os.path.join(
                        base_path,
                        dataset_dist_name,
                        f"FedAvg_fs{fs}" if cfg["type"] == "FedAvg" else f"FedSaSync_fs{fs}_m{cfg['m']}",
                    )
                    agg_path = os.path.join(d, "agg.json")
                    items = []
                    if os.path.exists(agg_path):
                        with open(agg_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        part = data.get("client_participation", {})
                        for cid, stats in part.items():
                            items.append(stats.get("mean", 0.0) if isinstance(stats, dict) else float(stats))
                    
                    sorted_items = np.sort(np.array(items))[::-1] if items else np.array([])
                    rows_data[cfg["name"]] = sorted_items
                
                strategy_names, matrix_data = [], []
                for cfg in configs_meta:
                    name = cfg["name"]
                    strategy_names.append(name)
                    arr = rows_data.get(name, np.array([]))
                    padded = np.pad(arr, (0, max(0, max_clients - len(arr))), "constant", constant_values=0)
                    matrix_data.append(padded[:max_clients])
                    
                client_labels = [f"C{i}" for i in range(1, max_clients + 1)]
                df_fs = pd.DataFrame(matrix_data, index=strategy_names, columns=client_labels)
                
                sns.heatmap(df_fs, annot=False, cmap="Blues", cbar=False, ax=ax, linewidths=0.3, linecolor="#e0e0e0")
                
                slows_pct = int(fs * 100) if isinstance(fs, float) and fs <= 1.0 else fs
                ax.set_title(f"Slow clients = {slows_pct}%", fontsize=14, pad=8)
                ax.set_xlabel("Clients (Sorted)", fontsize=12)
                
                if fs_idx % n_cols == 0:
                    ax.set_ylabel("Strategy", fontsize=12)
                else:
                    ax.set_ylabel("")
                    
                odd_indices = np.arange(0, max_clients, 2)
                ax.set_xticks(odd_indices + 0.5)
                ax.set_xticklabels([client_labels[i] for i in odd_indices], rotation=45, ha="center", fontsize=10)
                ax.tick_params(left=False)

            fig.subplots_adjust(right=0.90, hspace=0.45)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            
            sm = plt.cm.ScalarMappable(cmap="Blues", norm=plt.Normalize(vmin=0, vmax=50))
            sm.set_array([])
            fig.colorbar(sm, cax=cbar_ax)
                
            output_img = os.path.join(base_path, dataset_dist_name, f"heatmap_participation_{dataset_dist_name}{suffix}")
            plt.savefig(output_img, format=ext, bbox_inches="tight")
            plt.close()
            print(f"Participation Heatmap saved: {output_img}")


# ==========================================
# 2. ROUND STALENESS: Heatmap Version
# ==========================================
def plot_round_staleness_heatmap_split_fs(
    base_path: str,
    datasets: list,
    data_dists: list,
    strategies: list,
    fraction_slows: list,
    m_values: list,
    n_cols: int,
) -> None:
    """Generate and save round staleness heatmaps per dataset and data distribution."""
    ext = "svg" if len(fraction_slows) == 6 else "pdf"
    suffix = f"_complete.{ext}" if len(fraction_slows) == 6 else ".pdf"

    for dataset in datasets:
        for data_dist in data_dists:
            dataset_dist_name = f"{dataset}_{data_dist}"

            configs_meta = []
            if "FedSaSync" in strategies:
                for m in m_values:
                    configs_meta.append({"name": f"FedSaSync (M={m})", "type": "FedSaSync", "m": m})
            if "FedAvg" in strategies:
                configs_meta.append({"name": "FedAvg", "type": "FedAvg"})
            
            all_data = {}
            all_values_list = []

            for fs in fraction_slows:
                all_data[fs] = {}
                for cfg in configs_meta:
                    d = os.path.join(
                        base_path,
                        dataset_dist_name,
                        f"FedAvg_fs{fs}" if cfg["type"] == "FedAvg" else f"FedSaSync_fs{fs}_m{cfg['m']}",
                    )
                    agg_path = os.path.join(d, "agg.json")
                    mean_vals = []
                    if os.path.exists(agg_path):
                        with open(agg_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        mean_vals = data.get("round_staleness_mean", [])
                    
                    all_data[fs][cfg["name"]] = mean_vals
                    all_values_list.extend(mean_vals)

            if not all_values_list:
                continue

            global_vmax = np.percentile(all_values_list, 95) 
            global_vmin = 0

            n_fs = len(fraction_slows)
            n_rows = math.ceil(n_fs / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 0.8, 3.5 * n_rows), sharey=True)
            axes = np.atleast_1d(axes).flatten()
                
            for idx in range(n_fs, len(axes)):
                fig.delaxes(axes[idx])

            for fs_idx, fs in enumerate(fraction_slows):
                ax = axes[fs_idx]
                strategy_names = []
                matrix_data = []
                
                temp_rows = all_data[fs]
                max_rounds = max((len(v) for v in temp_rows.values()), default=10)
                if max_rounds == 0:
                    max_rounds = 10

                for cfg in configs_meta:
                    name = cfg["name"]
                    strategy_names.append(name)
                    arr = np.array(temp_rows.get(name, []))
                    padded = np.pad(arr, (0, max(0, max_rounds - len(arr))), "constant", constant_values=np.nan)
                    matrix_data.append(padded[:max_rounds])
                    
                round_labels = [str(i) for i in range(1, max_rounds + 1)]
                df_fs = pd.DataFrame(matrix_data, index=strategy_names, columns=round_labels)
                
                sns.heatmap(
                    df_fs, 
                    annot=False, 
                    cmap="Oranges", 
                    cbar=False, 
                    ax=ax, 
                    linewidths=0.2, 
                    linecolor="#e0e0e0",
                    vmin=global_vmin,
                    vmax=global_vmax,
                )
                
                slows_pct = int(fs * 100) if isinstance(fs, float) and fs <= 1.0 else fs
                ax.set_title(f"Slow clients = {slows_pct}%", fontsize=12, pad=8)
                ax.set_xlabel("Training Rounds", fontsize=10)
                
                if fs_idx % n_cols == 0:
                    ax.set_ylabel("Strategy", fontsize=10)
                else:
                    ax.set_ylabel("")
                    
                ticks_loc = np.arange(max_rounds) + 0.5
                ax.set_xticks(ticks_loc)
                
                if max_rounds > 20:
                    custom_labels = [str(i) if (i == 1 or i % 10 == 0) else "" for i in range(1, max_rounds + 1)]
                    ax.set_xticklabels(custom_labels, rotation=0, fontsize=9)
                else:
                    ax.set_xticklabels(round_labels, rotation=0, fontsize=9)
                
                ax.tick_params(left=False)
                
            fig.subplots_adjust(right=0.90, hspace=0.45)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            
            mappable = axes[0].collections[0]
            fig.colorbar(mappable, cax=cbar_ax)

            output_img = os.path.join(base_path, dataset_dist_name, f"heatmap_staleness_{dataset_dist_name}{suffix}")
            plt.savefig(output_img, format=ext, bbox_inches="tight")
            plt.close()
            print(f"Round Staleness Heatmap saved: {output_img}")


# ==========================================
# 3.1 EVALUATION LOSS: Line Plot vs Real Time
# ========================================== 
def plot_eval_loss_lines_split_fs(
    base_path: str,
    datasets: list,
    data_dists: list,
    strategies: list,
    fraction_slows: list,
    m_values: list,
    n_cols: int,
) -> None:
    """Generate a dynamic grid of Evaluation Loss vs Wall-Clock Time line plots."""
    ext = "svg" if len(fraction_slows) == 6 else "pdf"
    suffix = f"_complete.{ext}" if len(fraction_slows) == 6 else ".pdf"

    for dataset in datasets:
        for data_dist in data_dists:
            dataset_dist_name = f"{dataset}_{data_dist}"

            configs_meta = []
            if "FedSaSync" in strategies:
                for m in m_values:
                    configs_meta.append({"name": f"FedSaSync (M={m})", "type": "FedSaSync", "m": m})
            if "FedAvg" in strategies:
                configs_meta.append({"name": "FedAvg", "type": "FedAvg"})
                    
            n_fs = len(fraction_slows)
            n_rows = math.ceil(n_fs / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 0.8, 3.5 * n_rows), sharey=True)
            axes = np.atleast_1d(axes).flatten()
                
            for idx in range(n_fs, len(axes)):
                fig.delaxes(axes[idx])
                
            for fs_idx, fs in enumerate(fraction_slows):
                ax = axes[fs_idx]
                
                max_time_limit = None
                if "FedAvg" in strategies:
                    ref_d = os.path.join(base_path, dataset_dist_name, f"FedAvg_fs{fs}")
                    ref_csv = os.path.join(ref_d, "agg.csv")
                    if os.path.exists(ref_csv):
                        df_ref = pd.read_csv(ref_csv)
                        time_col = (
                            "round_total_wall_clock_sec_mean"
                            if "round_total_wall_clock_sec_mean" in df_ref.columns
                            else "time_mean"
                        )
                        if "eval_loss_mean" in df_ref.columns and time_col in df_ref.columns and len(df_ref) > 0:
                            raw_ref_time = df_ref[time_col].values
                            loss_ref = df_ref["eval_loss_mean"].values
                            
                            time_ref = raw_ref_time - raw_ref_time[0]
                            
                            start_idx = min(3, len(loss_ref) - 1)
                            min_loss_idx = start_idx + np.argmin(loss_ref[start_idx:])
                            time_at_min_loss = time_ref[min_loss_idx]
                            
                            max_time_limit = time_at_min_loss * 1.10
                
                for cfg in configs_meta:
                    d = os.path.join(
                        base_path,
                        dataset_dist_name,
                        f"FedAvg_fs{fs}" if cfg["type"] == "FedAvg" else f"FedSaSync_fs{fs}_m{cfg['m']}",
                    )
                    csv_path = os.path.join(d, "agg.csv")
                    
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        time_col = (
                            "round_total_wall_clock_sec_mean"
                            if "round_total_wall_clock_sec_mean" in df.columns
                            else "time_mean"
                        )
                        
                        if "eval_loss_mean" in df.columns and time_col in df.columns:
                            raw_time = df[time_col].values
                            loss_mean = df["eval_loss_mean"].values
                            loss_std = (
                                df["eval_loss_std"].values
                                if "eval_loss_std" in df.columns
                                else np.zeros_like(loss_mean)
                            )
                            
                            if len(loss_mean) > 0:
                                time_vals = raw_time - raw_time[0]
                                loss_vals = loss_mean
                                std_vals = loss_std
                                
                                (line,) = ax.plot(time_vals, loss_vals, label=cfg["name"], linewidth=1.5, zorder=3)
                                
                                lower_bound = np.maximum(0.0, loss_vals - std_vals)
                                upper_bound = loss_vals + std_vals
                                ax.fill_between(
                                    time_vals,
                                    lower_bound,
                                    upper_bound,
                                    color=line.get_color(),
                                    alpha=0.10,
                                    zorder=2,
                                )
                
                slows_pct = int(fs * 100) if isinstance(fs, float) and fs <= 1.0 else fs
                ax.set_title(f"Slow clients = {slows_pct}%", fontsize=12, pad=8)
                ax.set_xlabel("Elapsed Time (s)", fontsize=10)
                
                if fs_idx % n_cols == 0:
                    ax.set_ylabel("Evaluation Loss (Mean ± Std)", fontsize=10)
                else:
                    ax.set_ylabel("")
                    
                if max_time_limit:
                    ax.set_xlim(left=-3, right=max_time_limit)
                else:
                    ax.set_xlim(left=-3)
                    
                ax.grid(True, linestyle="--", alpha=0.5)
                    
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    ncol=4,
                    fontsize=9,
                    bbox_to_anchor=(0.5, 0.01),
                    frameon=True,
                )

            plt.tight_layout(rect=[0, 0.14, 1, 0.95])
            
            output_img = os.path.join(base_path, dataset_dist_name, f"eval_loss_{dataset_dist_name}{suffix}")
            plt.savefig(output_img, format=ext, bbox_inches="tight")
            plt.close()
            print(f"Dynamic Eval Loss Plot saved: {output_img}")


# ==========================================
# 3.2 EVALUATION ACCURACY: Line Plot vs Real Time
# ==========================================
def plot_eval_acc_lines_split_fs(
    base_path: str,
    datasets: list,
    data_dists: list,
    strategies: list,
    fraction_slows: list,
    m_values: list,
    n_cols: int,
) -> None:
    """Generate a dynamic grid of Evaluation Accuracy (%) vs Wall-Clock Time line plots."""
    ext = "svg" if len(fraction_slows) == 6 else "pdf"
    suffix = f"_complete.{ext}" if len(fraction_slows) == 6 else ".pdf"

    for dataset in datasets:
        for data_dist in data_dists:
            dataset_dist_name = f"{dataset}_{data_dist}"

            configs_meta = []
            if "FedSaSync" in strategies:
                for m in m_values:
                    configs_meta.append({"name": f"FedSaSync (M={m})", "type": "FedSaSync", "m": m})
            if "FedAvg" in strategies:
                configs_meta.append({"name": "FedAvg", "type": "FedAvg"})
                    
            n_fs = len(fraction_slows)
            n_rows = math.ceil(n_fs / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols + 0.8, 3.5 * n_rows), sharey=True)
            axes = np.atleast_1d(axes).flatten()
                
            for idx in range(n_fs, len(axes)):
                fig.delaxes(axes[idx])
                    
            for fs_idx, fs in enumerate(fraction_slows):
                ax = axes[fs_idx]
                
                max_time_limit = None
                if "FedAvg" in strategies:
                    ref_d = os.path.join(base_path, dataset_dist_name, f"FedAvg_fs{fs}")
                    ref_csv = os.path.join(ref_d, "agg.csv")
                    if os.path.exists(ref_csv):
                        df_ref = pd.read_csv(ref_csv)
                        time_col = "round_total_wall_clock_sec_mean" if "round_total_wall_clock_sec_mean" in df_ref.columns else "time_mean"
                        if "eval_loss_mean" in df_ref.columns and time_col in df_ref.columns and len(df_ref) > 0:
                            raw_ref_time = df_ref[time_col].values
                            loss_ref = df_ref["eval_loss_mean"].values
                            
                            time_ref = raw_ref_time - raw_ref_time[0]
                            
                            min_loss_idx = np.argmin(loss_ref)
                            time_at_min_loss = time_ref[min_loss_idx]
                            
                            max_time_limit = time_at_min_loss * 1.10
                
                for cfg in configs_meta:
                    d = os.path.join(base_path, dataset_dist_name, f"FedAvg_fs{fs}" if cfg["type"] == "FedAvg" else f"FedSaSync_fs{fs}_m{cfg['m']}")
                    csv_path = os.path.join(d, "agg.csv")
                    
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        time_col = "round_total_wall_clock_sec_mean" if "round_total_wall_clock_sec_mean" in df.columns else "time_mean"
                        
                        if "eval_acc_mean" in df.columns and time_col in df.columns:
                            raw_time = df[time_col].values
                            acc_mean = df["eval_acc_mean"].values.copy()
                            acc_std = df["eval_acc_std"].values if "eval_acc_std" in df.columns else np.zeros_like(acc_mean)
                            
                            if acc_mean.max() <= 1.0:
                                acc_mean = acc_mean * 100.0
                                acc_std = acc_std * 100.0
                            
                            if len(acc_mean) > 0:
                                time_vals = raw_time - raw_time[0]
                                acc_vals = acc_mean
                                std_vals = acc_std
                                
                                line, = ax.plot(time_vals, acc_vals, label=cfg["name"], linewidth=1.5, zorder=3)
                                
                                lower_bound = np.maximum(0.0, acc_vals - std_vals)
                                upper_bound = np.minimum(100.0, acc_vals + std_vals)
                                ax.fill_between(
                                    time_vals,
                                    lower_bound,
                                    upper_bound,
                                    color=line.get_color(),
                                    alpha=0.10,
                                    zorder=2
                                )
                
                slows_pct = int(fs * 100) if isinstance(fs, float) and fs <= 1.0 else fs
                ax.set_title(f"Slow clients = {slows_pct}%", fontsize=12, pad=8)
                ax.set_xlabel("Elapsed Time (s)", fontsize=10)
                
                if fs_idx % n_cols == 0:
                    ax.set_ylabel("Evaluation Accuracy (%) (Mean ± Std)", fontsize=10)
                else:
                    ax.set_ylabel("")
                    
                if max_time_limit:
                    ax.set_xlim(left=-3, right=max_time_limit)
                else:
                    ax.set_xlim(left=-3)
                    
                ax.grid(True, linestyle='--', alpha=0.5)
                    
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=9, bbox_to_anchor=(0.5, 0.01), frameon=True)

            plt.tight_layout(rect=[0, 0.14, 1, 0.95])
            
            output_img = os.path.join(base_path, dataset_dist_name, f"eval_acc_{dataset_dist_name}{suffix}")
            plt.savefig(output_img, format=ext, bbox_inches='tight')
            plt.close()
            print(f"Eval Accuracy Líneas dinámicas guardada: {output_img}")


# ==========================================
# 4.1. TABLA COMPREHENSIVA
# ==========================================
def generate_efficiency_table(base_path: str, datasets: list, data_dists: list, strategies: list, fraction_slows: list, m_values: list):
    """Genera una tabla LaTeX vertical estándar por cada fraction_slow, con espaciado adecuado y superíndices numéricos de éxito."""
    m_filtered = [m for m in m_values if m != 4]
    row_strategies = [{"type": "FedSaSync", "m": m} for m in m_filtered]
    if "FedAvg" in strategies:
        row_strategies.append({"type": "FedAvg"})
        
    TARGET_ACCS = [50.0, 58.0]
    TARGET_TIMES = [15.0, 25.0]
    metrics_names = [r"$\Delta_{loss}/time$", "Max Acc.", "TTA 50\\%", "TTA 58\\%", "Acc@15s", "Acc@25s"]
    num_metrics = len(metrics_names)
    
    def parse_metric_tuple(val_str, metric_idx):
        if val_str is None:
            return None
        val_clean = str(val_str).strip()
        if not val_clean or val_clean.startswith("N/A") or val_clean in ["-", ""]:
            return None
        try:
            val_clean = re.sub(r'\\textsuperscript\{[^}]*\}', '', val_clean).strip()
            parts = re.split(r'\s*(?:\\pm|±)\s*', val_clean)
            cleaned_val = parts[0].replace("$", "").replace("\\%", "").replace("%", "").replace("s", "").strip()
            if not cleaned_val:
                return None
            main_val = float(cleaned_val)
            std_val = 0.0
            if len(parts) >= 2:
                std_clean = parts[1].replace("$", "").replace("\\%", "").replace("%", "").replace("s", "").strip()
                if std_clean:
                    try:
                        std_val = float(std_clean)
                    except ValueError:
                        std_val = 0.0
            return (main_val, std_val)
        except (ValueError, TypeError, IndexError):
            return None

    def is_lower_better(metric_idx):
        return metric_idx in [2, 3]

    for dataset in datasets:
        for data_dist in data_dists:
            dataset_dist_name = f"{dataset}_{data_dist}"

            table_rows_data = []
            row_labels = []
            
            for cfg in row_strategies:
                row_label = "FedAvg" if cfg["type"] == "FedAvg" else f"FedSaSync ($M = {cfg['m']}$)"
                row_labels.append(row_label)
                
                row_values = []
                for fs in fraction_slows:
                    if cfg["type"] == "FedAvg":
                        d = os.path.join(base_path, dataset_dist_name, f"FedAvg_fs{fs}")
                    else:
                        d = os.path.join(base_path, dataset_dist_name, f"FedSaSync_fs{fs}_m{cfg['m']}")
                        
                    csv_path = os.path.join(d, "agg.csv")
                    json_path = os.path.join(d, "agg.json")
                    
                    eff, max_acc = "-", "-"
                    tta_results = ["N/A", "N/A"]
                    acc_at_times = ["N/A", "N/A"]
                    
                    jdata = {}
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, "r", encoding="utf-8") as jf:
                                jdata = json.load(jf)
                        except Exception:
                            pass

                    loss_per_sec_metrics = jdata.get("loss_per_sec_metrics", {})
                    if loss_per_sec_metrics.get("loss_per_sec_mean") is not None:
                        eff_mean = loss_per_sec_metrics["loss_per_sec_mean"]
                        eff_std = loss_per_sec_metrics.get("loss_per_sec_std", 0.0)
                        eff = f"{eff_mean:.4f} $\\pm$ {eff_std:.4f}"

                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        if "eval_acc_mean" in df.columns and len(df) > 0:
                            acc_values = df["eval_acc_mean"].values.copy()
                            acc_std_values = df["eval_acc_std"].values if "eval_acc_std" in df.columns else np.zeros_like(acc_values)
                            if acc_values.max() <= 1.0:
                                acc_values = acc_values * 100.0
                                acc_std_values = acc_std_values * 100.0
                            max_idx = np.argmax(acc_values)
                            max_acc = f"{acc_values[max_idx]:.1f} $\\pm$ {acc_std_values[max_idx]:.1f}\\%"

                    tta_metrics = jdata.get("tta_metrics", {})
                    for i, target in enumerate(TARGET_ACCS):
                        mean_key = f"tta{int(target)}_mean"
                        std_key = f"tta{int(target)}_std"
                        count_key = f"tta{int(target)}_success_count"
                        total_key = f"tta{int(target)}_total_runs"
                        
                        if mean_key in tta_metrics and tta_metrics[mean_key] is not None:
                            t_mean = tta_metrics[mean_key]
                            t_std = tta_metrics.get(std_key, 0.0)
                            success_count = tta_metrics.get(count_key, 5)
                            total_runs = tta_metrics.get(total_key, 5)
                            
                            if success_count == 0 or success_count < (total_runs * 0.6):
                                tta_results[i] = "N/A" if success_count == 0 else f"N/A\\textsuperscript{{{success_count}}}"
                            else:
                                sup_text = "" if success_count == total_runs else f"\\textsuperscript{{{success_count}}}"
                                tta_results[i] = f"{t_mean:.1f} $\\pm$ {t_std:.1f}s{sup_text}"

                    acc_at_time_metrics = jdata.get("acc_at_time_metrics", {})
                    for i, t_target in enumerate(TARGET_TIMES):
                        mean_key = f"acc_at_{int(t_target)}s_mean"
                        std_key = f"acc_at_{int(t_target)}s_std"
                        if mean_key in acc_at_time_metrics and acc_at_time_metrics[mean_key] is not None:
                            acc_mean = acc_at_time_metrics[mean_key]
                            acc_std = acc_at_time_metrics.get(std_key, 0.0)
                            if acc_mean <= 1.0:
                                acc_mean *= 100.0
                                acc_std *= 100.0
                            acc_at_times[i] = f"{acc_mean:.1f} $\\pm$ {acc_std:.1f}\\%"

                    row_values.extend([eff, max_acc, tta_results[0], tta_results[1], acc_at_times[0], acc_at_times[1]])
                table_rows_data.append(row_values)
                
            best_tuples_map = {}
            for fs_idx in range(len(fraction_slows)):
                for m_idx in range(num_metrics):
                    col_idx = fs_idx * num_metrics + m_idx
                    parsed_tuples = [parse_metric_tuple(r_vals[col_idx], m_idx) for r_vals in table_rows_data]
                    parsed_tuples = [t for t in parsed_tuples if t is not None]
                    
                    if parsed_tuples:
                        lower_better = is_lower_better(m_idx)
                        main_vals = [t[0] for t in parsed_tuples]
                        best_main = min(main_vals) if lower_better else max(main_vals)
                        tied_by_main = [t for t in parsed_tuples if abs(t[0] - best_main) < 1e-4]
                        best_std = min([t[1] for t in tied_by_main]) if tied_by_main else 0.0
                        best_tuples_map[(fs_idx, m_idx)] = (best_main, best_std)

            dataset_dist_latex = dataset_dist_name.replace("_", "\\_")
            latex_lines = [
                r"\begin{table*}[t]",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{4pt}",
                r"\renewcommand{\arraystretch}{0.85}",
                f"\\caption{{Comprehensive performance and efficiency metrics for {dataset_dist_latex}.}}",
                f"\\label{{tab:comprehensive_{dataset_dist_name}}}",
            ]
            
            for fs_idx, fs in enumerate(fraction_slows):
                if fs_idx > 0:
                    latex_lines.append(r"\par\vspace{0.8em}")
                
                latex_lines.append(f"\\noindent\\textbf{{Slow = {int(round(fs * 100))}\\%}} \\\\[0.2em]")
                latex_lines.append(r"\resizebox{\textwidth}{!}{%")
                latex_lines.append(r"\begin{tabular}{lcccccc}")
                latex_lines.append(r"\hline")
                latex_lines.append(" & ".join(["Strategy"] + metrics_names) + r" \\")
                latex_lines.append(r"\hline")
                
                for idx, r_label in enumerate(row_labels):
                    r_vals = table_rows_data[idx]
                    sub_vals = []
                    
                    for m_idx in range(num_metrics):
                        val_idx = fs_idx * num_metrics + m_idx
                        raw_val = r_vals[val_idx]
                        parsed_tup = parse_metric_tuple(raw_val, m_idx)
                        best_tup = best_tuples_map.get((fs_idx, m_idx))
                        
                        is_best = False
                        if parsed_tup is not None and best_tup is not None:
                            if abs(parsed_tup[0] - best_tup[0]) < 1e-4 and abs(parsed_tup[1] - best_tup[1]) < 1e-4:
                                is_best = True
                        
                        if is_best and raw_val not in ["-", "N/A", ""] and not str(raw_val).startswith("N/A"):
                            sub_vals.append(r"\textbf{" + str(raw_val) + "}")
                        else:
                            sub_vals.append(str(raw_val))
                            
                    latex_lines.append(" & ".join([r_label] + sub_vals) + r" \\")
                    
                latex_lines.extend([
                    r"\hline",
                    r"\end{tabular}%",
                    r"}"
                ])

            latex_lines.extend([
                r"\par\vspace{0.8em}",
                r"\raggedright \footnotesize \textit{Note:} Superscripts indicate the number of successful runs out of 5 repetitions when not all runs finished within the target.",
                r"\end{table*}"
            ])
            
            suffix = "_complete" if len(fraction_slows) == 6 else ""
            output_tex_path = os.path.join(base_path, dataset_dist_name, f"efficiency_table_{dataset_dist_name}{suffix}.tex")
            with open(output_tex_path, "w", encoding="utf-8") as f:
                f.write("\n".join(latex_lines))


# ==========================================
# 4.2. TABLA DE SCHEDULER
# ==========================================
def generate_scheduler_metrics_table(base_path: str, datasets: list, data_dists: list, strategies: list, fraction_slows: list, m_values: list):
    """Genera una tabla LaTeX vertical estándar por cada fraction_slow para las métricas del scheduler."""
    m_filtered = [m for m in m_values if m != 4]
    row_strategies = [{"type": "FedSaSync", "m": m} for m in m_filtered]
    if "FedAvg" in strategies:
        row_strategies.append({"type": "FedAvg"})
        
    metrics_names = [
        "Polling Calls", 
        "Polling Overhead (\\%)", 
        "Server CPU Util. (\\%)", 
        "Agg. Rate (msgs/s)"
    ]
    num_metrics = len(metrics_names)
    
    def parse_metric_value(val_str, metric_idx):
        if val_str is None or str(val_str).startswith("N/A") or str(val_str) in ["-", ""]:
            return None
        try:
            cleaned = str(val_str).split(r" $\pm$ ")[0]
            if "%" in cleaned or "\\%" in cleaned:
                cleaned = cleaned.replace("\\%", "").replace("%", "").strip()
            elif cleaned.endswith("s"):
                cleaned = cleaned[:-1].strip()
            return float(cleaned)
        except ValueError:
            return None

    def is_lower_better(metric_idx):
        return metric_idx in [0, 1, 2]

    for dataset in datasets:
        for data_dist in data_dists:
            dataset_dist_name = f"{dataset}_{data_dist}"

            table_rows_data = []
            row_labels = []
            
            for cfg in row_strategies:
                row_label = "FedAvg" if cfg["type"] == "FedAvg" else f"FedSaSync ($M = {cfg['m']}$)"
                row_labels.append(row_label)
                
                row_values = []
                for fs in fraction_slows:
                    if cfg["type"] == "FedAvg":
                        d = os.path.join(base_path, dataset_dist_name, f"FedAvg_fs{fs}")
                    else:
                        d = os.path.join(base_path, dataset_dist_name, f"FedSaSync_fs{fs}_m{cfg['m']}")
                        
                    json_path = os.path.join(d, "agg.json")
                    metrics_results = ["-", "-", "-", "-"]
                    
                    jdata = {}
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, "r", encoding="utf-8") as jf:
                                jdata = json.load(jf)
                        except Exception:
                            pass

                    scheduler_metrics = jdata.get("scheduler_metrics", {})
                    total_mean = scheduler_metrics.get("total_time_mean")
                    total_std = scheduler_metrics.get("total_time_std", 0.0)

                    # 1. Polling Calls
                    p_calls_mean = scheduler_metrics.get("polling_calls_count_mean")
                    p_calls_std = scheduler_metrics.get("polling_calls_count_std", 0.0)
                    if p_calls_mean is not None:
                        metrics_results[0] = f"{p_calls_mean:.1f} $\\pm$ {p_calls_std:.1f}"

                    # 2. Polling Overhead (%)
                    pull_mean = scheduler_metrics.get("time_in_pull_sec_mean")
                    pull_std = scheduler_metrics.get("time_in_pull_sec_std", 0.0)
                    if pull_mean is not None and total_mean is not None and total_mean > 0:
                        overhead_mean = (pull_mean / total_mean) * 100.0
                        overhead_std = overhead_mean * np.sqrt((pull_std / pull_mean)**2 + (total_std / total_mean)**2) if pull_mean > 0 and total_std > 0 else 0.0
                        metrics_results[1] = f"{overhead_mean:.1f} $\\pm$ {overhead_std:.1f}\\%"

                    # 3. Server CPU Util. (%)
                    cpu_time_mean = scheduler_metrics.get("server_cpu_time_sec_mean")
                    cpu_time_std = scheduler_metrics.get("server_cpu_time_sec_std", 0.0)
                    if cpu_time_mean is not None and total_mean is not None and total_mean > 0:
                        cpu_util_mean = (cpu_time_mean / total_mean) * 100.0
                        cpu_util_std = cpu_util_mean * np.sqrt((cpu_time_std / cpu_time_mean)**2 + (total_std / total_mean)**2) if cpu_time_mean > 0 and total_std > 0 else 0.0
                        metrics_results[2] = f"{cpu_util_mean:.1f} $\\pm$ {cpu_util_std:.1f}\\%"

                    # 4. Agg. Rate (msgs/s)
                    msgs_mean = scheduler_metrics.get("messages_received_count_mean")
                    msgs_std = scheduler_metrics.get("messages_received_count_std", 0.0)
                    if msgs_mean is not None and total_mean is not None and total_mean > 0:
                        agg_rate_mean = msgs_mean / total_mean
                        agg_rate_std = agg_rate_mean * np.sqrt((msgs_std / msgs_mean)**2 + (total_std / total_mean)**2) if msgs_mean > 0 and total_std > 0 else 0.0
                        metrics_results[3] = f"{agg_rate_mean:.2f} $\\pm$ {agg_rate_std:.2f}"

                    row_values.extend(metrics_results)
                table_rows_data.append(row_values)
                
            best_values_map = {}
            for fs_idx in range(len(fraction_slows)):
                for m_idx in range(num_metrics):
                    col_idx = fs_idx * num_metrics + m_idx
                    parsed_vals = []
                    for r_vals in table_rows_data:
                        val = parse_metric_value(r_vals[col_idx], m_idx)
                        if val is not None:
                            parsed_vals.append(val)
                    if parsed_vals:
                        if is_lower_better(m_idx):
                            best_values_map[(fs_idx, m_idx)] = min(parsed_vals)
                        else:
                            best_values_map[(fs_idx, m_idx)] = max(parsed_vals)

            dataset_dist_latex = dataset_dist_name.replace("_", "\\_")
            latex_lines = [
                r"\begin{table*}[t]",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{4pt}",
                r"\renewcommand{\arraystretch}{0.85}",
                f"\\caption{{Scheduler and communication overhead metrics for {dataset_dist_latex}.}}",
                f"\\label{{tab:scheduler_{dataset_dist_name}}}",
            ]
            
            for fs_idx, fs in enumerate(fraction_slows):
                if fs_idx > 0:
                    latex_lines.append(r"\par\vspace{0.8em}")
                
                latex_lines.append(f"\\noindent\\textbf{{Slow = {int(round(fs * 100))}\\%}} \\\\[0.2em]")
                latex_lines.append(r"\resizebox{\textwidth}{!}{%")
                col_spec = "l" + "".join(["c"] * num_metrics)
                latex_lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
                latex_lines.append(r"\hline")
                latex_lines.append(" & ".join(["Strategy"] + metrics_names) + r" \\")
                latex_lines.append(r"\hline")
                
                for idx, r_label in enumerate(row_labels):
                    r_vals = table_rows_data[idx]
                    sub_vals = []
                    
                    for m_idx in range(num_metrics):
                        val_idx = fs_idx * num_metrics + m_idx
                        raw_val = r_vals[val_idx]
                        parsed = parse_metric_value(raw_val, m_idx)
                        best_val = best_values_map.get((fs_idx, m_idx))
                        
                        is_best = parsed is not None and best_val is not None and abs(parsed - best_val) < 1e-5
                        if is_best and raw_val not in ["-", "N/A", ""]:
                            sub_vals.append(r"\textbf{" + str(raw_val) + "}")
                        else:
                            sub_vals.append(str(raw_val))
                            
                    latex_lines.append(" & ".join([r_label] + sub_vals) + r" \\")
                    
                latex_lines.extend([
                    r"\hline",
                    r"\end{tabular}%",
                    r"}"
                ])

            latex_lines.extend([
                r"\end{table*}"
            ])
            
            suffix = "_complete" if len(fraction_slows) == 6 else ""
            output_tex_path = os.path.join(base_path, dataset_dist_name, f"scheduler_table_{dataset_dist_name}{suffix}.tex")
            with open(output_tex_path, "w", encoding="utf-8") as f:
                f.write("\n".join(latex_lines))


def generate_efficiency_md(base_path: str, datasets: list, data_dists: list, strategies: list, fraction_slows: list, m_values: list):
    """Genera una tabla Markdown vertical agrupada por fraction_slow con métricas de rendimiento y eficiencia."""
    m_filtered = [m for m in m_values if m != 4]
    row_strategies = [{"type": "FedSaSync", "m": m} for m in m_filtered]
    if "FedAvg" in strategies:
        row_strategies.append({"type": "FedAvg"})
        
    TARGET_ACCS = [50.0, 58.0]
    TARGET_TIMES = [15.0, 25.0]
    metrics_names = ["Δloss/time", "Max Acc.", "TTA 50%", "TTA 58%", "Acc@15s", "Acc@25s"]
    num_metrics = len(metrics_names)
    
    def parse_metric_tuple(val_str, metric_idx):
        if val_str is None:
            return None
        val_clean = str(val_str).strip()
        if not val_clean or val_clean.startswith("N/A") or val_clean in ["-", ""]:
            return None
        try:
            val_clean = re.sub(r'<sup[^>]*>.*?</sup>', '', val_clean).strip()
            parts = re.split(r'\s*(?:\\pm|±)\s*', val_clean)
            cleaned_val = parts[0].replace("$", "").replace("%", "").replace("s", "").strip()
            if not cleaned_val:
                return None
            main_val = float(cleaned_val)
            std_val = 0.0
            if len(parts) >= 2:
                std_clean = parts[1].replace("$", "").replace("%", "").replace("s", "").strip()
                if std_clean:
                    try:
                        std_val = float(std_clean)
                    except ValueError:
                        std_val = 0.0
            return (main_val, std_val)
        except (ValueError, TypeError, IndexError):
            return None

    def is_lower_better(metric_idx):
        return metric_idx in [2, 3]

    for dataset in datasets:
        for data_dist in data_dists:
            dataset_dist_name = f"{dataset}_{data_dist}"

            table_rows_data = []
            row_labels = []
            
            for cfg in row_strategies:
                row_label = "FedAvg" if cfg["type"] == "FedAvg" else f"FedSaSync (M = {cfg['m']})"
                row_labels.append(row_label)
                
                row_values = []
                for fs in fraction_slows:
                    if cfg["type"] == "FedAvg":
                        d = os.path.join(base_path, dataset_dist_name, f"FedAvg_fs{fs}")
                    else:
                        d = os.path.join(base_path, dataset_dist_name, f"FedSaSync_fs{fs}_m{cfg['m']}")
                        
                    csv_path = os.path.join(d, "agg.csv")
                    json_path = os.path.join(d, "agg.json")
                    
                    eff, max_acc = "-", "-"
                    tta_results = ["N/A", "N/A"]
                    acc_at_times = ["N/A", "N/A"]
                    
                    jdata = {}
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, "r", encoding="utf-8") as jf:
                                jdata = json.load(jf)
                        except Exception:
                            pass

                    loss_per_sec_metrics = jdata.get("loss_per_sec_metrics", {})
                    if loss_per_sec_metrics.get("loss_per_sec_mean") is not None:
                        eff_mean = loss_per_sec_metrics["loss_per_sec_mean"]
                        eff_std = loss_per_sec_metrics.get("loss_per_sec_std", 0.0)
                        eff = f"{eff_mean:.4f} ± {eff_std:.4f}"

                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        if "eval_acc_mean" in df.columns and len(df) > 0:
                            acc_values = df["eval_acc_mean"].values.copy()
                            acc_std_values = df["eval_acc_std"].values if "eval_acc_std" in df.columns else np.zeros_like(acc_values)
                            if acc_values.max() <= 1.0:
                                acc_values = acc_values * 100.0
                                acc_std_values = acc_std_values * 100.0
                            max_idx = np.argmax(acc_values)
                            max_acc = f"{acc_values[max_idx]:.1f} ± {acc_std_values[max_idx]:.1f}%"

                    tta_metrics = jdata.get("tta_metrics", {})
                    for i, target in enumerate(TARGET_ACCS):
                        mean_key = f"tta{int(target)}_mean"
                        std_key = f"tta{int(target)}_std"
                        count_key = f"tta{int(target)}_success_count"
                        total_key = f"tta{int(target)}_total_runs"
                        
                        if mean_key in tta_metrics and tta_metrics[mean_key] is not None:
                            t_mean = tta_metrics[mean_key]
                            t_std = tta_metrics.get(std_key, 0.0)
                            success_count = tta_metrics.get(count_key, 5)
                            total_runs = tta_metrics.get(total_key, 5)
                            
                            if success_count == 0 or success_count < (total_runs * 0.6):
                                tta_results[i] = "N/A" if success_count == 0 else f"N/A<sup>{success_count}</sup>"
                            else:
                                sup_text = "" if success_count == total_runs else f"<sup>{success_count}</sup>"
                                tta_results[i] = f"{t_mean:.1f} ± {t_std:.1f}s{sup_text}"

                    acc_at_time_metrics = jdata.get("acc_at_time_metrics", {})
                    for i, t_target in enumerate(TARGET_TIMES):
                        mean_key = f"acc_at_{int(t_target)}s_mean"
                        std_key = f"acc_at_{int(t_target)}s_std"
                        if mean_key in acc_at_time_metrics and acc_at_time_metrics[mean_key] is not None:
                            acc_mean = acc_at_time_metrics[mean_key]
                            acc_std = acc_at_time_metrics.get(std_key, 0.0)
                            if acc_mean <= 1.0:
                                acc_mean *= 100.0
                                acc_std *= 100.0
                            acc_at_times[i] = f"{acc_mean:.1f} ± {acc_std:.1f}%"

                    row_values.extend([eff, max_acc, tta_results[0], tta_results[1], acc_at_times[0], acc_at_times[1]])
                table_rows_data.append(row_values)
                
            best_tuples_map = {}
            for fs_idx in range(len(fraction_slows)):
                for m_idx in range(num_metrics):
                    col_idx = fs_idx * num_metrics + m_idx
                    parsed_tuples = [parse_metric_tuple(r_vals[col_idx], m_idx) for r_vals in table_rows_data]
                    parsed_tuples = [t for t in parsed_tuples if t is not None]
                    
                    if parsed_tuples:
                        lower_better = is_lower_better(m_idx)
                        main_vals = [t[0] for t in parsed_tuples]
                        best_main = min(main_vals) if lower_better else max(main_vals)
                        tied_by_main = [t for t in parsed_tuples if abs(t[0] - best_main) < 1e-4]
                        best_std = min([t[1] for t in tied_by_main]) if tied_by_main else 0.0
                        best_tuples_map[(fs_idx, m_idx)] = (best_main, best_std)

            md_lines = [
                f"# Comprehensive Performance & Efficiency Metrics: {dataset_dist_name}\n"
            ]
            
            for fs_idx, fs in enumerate(fraction_slows):
                md_lines.append(f"### Slow Fraction = {fs}\n")
                
                headers = ["Strategy"] + metrics_names
                md_lines.append("| " + " | ".join(headers) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                
                for idx, r_label in enumerate(row_labels):
                    r_vals = table_rows_data[idx]
                    sub_vals = []
                    
                    for m_idx in range(num_metrics):
                        val_idx = fs_idx * num_metrics + m_idx
                        raw_val = r_vals[val_idx]
                        parsed_tup = parse_metric_tuple(raw_val, m_idx)
                        best_tup = best_tuples_map.get((fs_idx, m_idx))
                        
                        is_best = False
                        if parsed_tup is not None and best_tup is not None:
                            if abs(parsed_tup[0] - best_tup[0]) < 1e-4 and abs(parsed_tup[1] - best_tup[1]) < 1e-4:
                                is_best = True
                        
                        if is_best and raw_val not in ["-", "N/A", ""] and not str(raw_val).startswith("N/A"):
                            sub_vals.append(f"**{raw_val}**")
                        else:
                            sub_vals.append(str(raw_val))
                            
                    md_lines.append("| " + " | ".join([r_label] + sub_vals) + " |")
                
                md_lines.append("")  # Línea en blanco para separar secciones
                
            md_lines.append("*Note: Superscripts indicate the number of successful runs out of 5 repetitions when not all runs finished within the target.*")
            
            output_md_path = os.path.join(base_path, dataset_dist_name, f"efficiency_table_{dataset_dist_name}.md")
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_lines))


def generate_scheduler_metrics_md(base_path: str, datasets: list, data_dists: list, strategies: list, fraction_slows: list, m_values: list):
    """Genera una tabla Markdown vertical agrupada por fraction_slow para métricas de comunicación y scheduler."""
    m_filtered = [m for m in m_values if m != 4]
    row_strategies = [{"type": "FedSaSync", "m": m} for m in m_filtered]
    if "FedAvg" in strategies:
        row_strategies.append({"type": "FedAvg"})
        
    metrics_names = [
        "Polling Calls", 
        "Polling Overhead (%)", 
        "Server CPU Util. (%)", 
        "Agg. Rate (msgs/s)"
    ]
    num_metrics = len(metrics_names)
    
    def parse_metric_value(val_str, metric_idx):
        if val_str is None or str(val_str).startswith("N/A") or str(val_str) in ["-", ""]:
            return None
        try:
            cleaned = str(val_str).split(" ± ")[0]
            if "%" in cleaned:
                cleaned = cleaned.replace("%", "").strip()
            elif cleaned.endswith("s"):
                cleaned = cleaned[:-1].strip()
            return float(cleaned)
        except ValueError:
            return None

    def is_lower_better(metric_idx):
        return metric_idx in [0, 1, 2]

    for dataset in datasets:
        for data_dist in data_dists:
            dataset_dist_name = f"{dataset}_{data_dist}"

            table_rows_data = []
            row_labels = []
            
            for cfg in row_strategies:
                row_label = "FedAvg" if cfg["type"] == "FedAvg" else f"FedSaSync (M = {cfg['m']})"
                row_labels.append(row_label)
                
                row_values = []
                for fs in fraction_slows:
                    if cfg["type"] == "FedAvg":
                        d = os.path.join(base_path, dataset_dist_name, f"FedAvg_fs{fs}")
                    else:
                        d = os.path.join(base_path, dataset_dist_name, f"FedSaSync_fs{fs}_m{cfg['m']}")
                        
                    json_path = os.path.join(d, "agg.json")
                    metrics_results = ["-", "-", "-", "-"]
                    
                    jdata = {}
                    if os.path.exists(json_path):
                        try:
                            with open(json_path, "r", encoding="utf-8") as jf:
                                jdata = json.load(jf)
                        except Exception:
                            pass

                    scheduler_metrics = jdata.get("scheduler_metrics", {})
                    total_mean = scheduler_metrics.get("total_time_mean")
                    total_std = scheduler_metrics.get("total_time_std", 0.0)

                    # 1. Polling Calls
                    p_calls_mean = scheduler_metrics.get("polling_calls_count_mean")
                    p_calls_std = scheduler_metrics.get("polling_calls_count_std", 0.0)
                    if p_calls_mean is not None:
                        metrics_results[0] = f"{p_calls_mean:.1f} ± {p_calls_std:.1f}"

                    # 2. Polling Overhead (%)
                    pull_mean = scheduler_metrics.get("time_in_pull_sec_mean")
                    pull_std = scheduler_metrics.get("time_in_pull_sec_std", 0.0)
                    if pull_mean is not None and total_mean is not None and total_mean > 0:
                        overhead_mean = (pull_mean / total_mean) * 100.0
                        overhead_std = overhead_mean * np.sqrt((pull_std / pull_mean)**2 + (total_std / total_mean)**2) if pull_mean > 0 and total_std > 0 else 0.0
                        metrics_results[1] = f"{overhead_mean:.1f} ± {overhead_std:.1f}%"

                    # 3. Server CPU Util. (%)
                    cpu_time_mean = scheduler_metrics.get("server_cpu_time_sec_mean")
                    cpu_time_std = scheduler_metrics.get("server_cpu_time_sec_std", 0.0)
                    if cpu_time_mean is not None and total_mean is not None and total_mean > 0:
                        cpu_util_mean = (cpu_time_mean / total_mean) * 100.0
                        cpu_util_std = cpu_util_mean * np.sqrt((cpu_time_std / cpu_time_mean)**2 + (total_std / total_mean)**2) if cpu_time_mean > 0 and total_std > 0 else 0.0
                        metrics_results[2] = f"{cpu_util_mean:.1f} ± {cpu_util_std:.1f}%"

                    # 4. Agg. Rate (msgs/s)
                    msgs_mean = scheduler_metrics.get("messages_received_count_mean")
                    msgs_std = scheduler_metrics.get("messages_received_count_std", 0.0)
                    if msgs_mean is not None and total_mean is not None and total_mean > 0:
                        agg_rate_mean = msgs_mean / total_mean
                        agg_rate_std = agg_rate_mean * np.sqrt((msgs_std / msgs_mean)**2 + (total_std / total_mean)**2) if msgs_mean > 0 and total_std > 0 else 0.0
                        metrics_results[3] = f"{agg_rate_mean:.2f} ± {agg_rate_std:.2f}"

                    row_values.extend(metrics_results)
                table_rows_data.append(row_values)
                
            best_values_map = {}
            for fs_idx in range(len(fraction_slows)):
                for m_idx in range(num_metrics):
                    col_idx = fs_idx * num_metrics + m_idx
                    parsed_vals = []
                    for r_vals in table_rows_data:
                        val = parse_metric_value(r_vals[col_idx], m_idx)
                        if val is not None:
                            parsed_vals.append(val)
                    if parsed_vals:
                        if is_lower_better(m_idx):
                            best_values_map[(fs_idx, m_idx)] = min(parsed_vals)
                        else:
                            best_values_map[(fs_idx, m_idx)] = max(parsed_vals)

            md_lines = [
                f"# Scheduler and Overhead Metrics: {dataset_dist_name}\n"
            ]
            
            for fs_idx, fs in enumerate(fraction_slows):
                md_lines.append(f"### Slow Fraction = {fs}\n")
                
                headers = ["Strategy"] + metrics_names
                md_lines.append("| " + " | ".join(headers) + " |")
                md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                
                for idx, r_label in enumerate(row_labels):
                    r_vals = table_rows_data[idx]
                    sub_vals = []
                    
                    for m_idx in range(num_metrics):
                        val_idx = fs_idx * num_metrics + m_idx
                        raw_val = r_vals[val_idx]
                        parsed = parse_metric_value(raw_val, m_idx)
                        best_val = best_values_map.get((fs_idx, m_idx))
                        
                        is_best = parsed is not None and best_val is not None and abs(parsed - best_val) < 1e-5
                        if is_best and raw_val not in ["-", "N/A", ""]:
                            sub_vals.append(f"**{raw_val}**")
                        else:
                            sub_vals.append(str(raw_val))
                            
                    md_lines.append("| " + " | ".join([r_label] + sub_vals) + " |")
                
                md_lines.append("")
            
            output_md_path = os.path.join(base_path, dataset_dist_name, f"scheduler_table_{dataset_dist_name}.md")
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_lines))

if __name__ == "__main__":
    BASE_PATH = "_static"
    DATASETS = ["cifar10", "mnist"]
    DATA_DISTS = ["dirichlet", "iid"]
    STRATEGIES = ["FedAvg", "FedSaSync"]
    FRACTION_SLOWS = [0.0, 0.2, 0.5]
    M_VALUES = [8, 10, 12, 14, 16, 18, 20]

    # Execute all graphing scripts
    plot_participation_heatmap_split_fs(BASE_PATH, DATASETS, DATA_DISTS, STRATEGIES, FRACTION_SLOWS, M_VALUES, 3)
    plot_round_staleness_heatmap_split_fs(BASE_PATH, DATASETS, DATA_DISTS, STRATEGIES, FRACTION_SLOWS, M_VALUES, 3)
    plot_eval_loss_lines_split_fs(BASE_PATH, DATASETS, DATA_DISTS, STRATEGIES, FRACTION_SLOWS, M_VALUES, 3)
    plot_eval_acc_lines_split_fs(BASE_PATH, DATASETS, DATA_DISTS, STRATEGIES, FRACTION_SLOWS, M_VALUES, 3)

    if len(FRACTION_SLOWS) <= 3:
        generate_efficiency_table(BASE_PATH, DATASETS, DATA_DISTS, STRATEGIES, FRACTION_SLOWS, M_VALUES)
        generate_scheduler_metrics_table(BASE_PATH, DATASETS, DATA_DISTS, STRATEGIES, FRACTION_SLOWS, M_VALUES)
    else:
        generate_efficiency_md(BASE_PATH, DATASETS, DATA_DISTS, STRATEGIES, FRACTION_SLOWS, M_VALUES)
        generate_scheduler_metrics_md(BASE_PATH, DATASETS, DATA_DISTS, STRATEGIES, FRACTION_SLOWS, M_VALUES)
