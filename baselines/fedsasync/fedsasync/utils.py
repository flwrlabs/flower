"""FedSaSync: Semi-asynchronous Federated Learning in Flower."""
import csv
import json
import os
from logging import INFO

from flwr.common import MetricRecord, RecordDict, log
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords


def train_metrics_aggr_fn(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Personalized train_metrics_aggr_fn to delete client times."""
    train_times: list[float] = []
    for record in records:
        value = record.metric_records["metrics"]["train_time"]
        if isinstance(value, list):
            raise ValueError("train_time should never be a list")

        train_times.append(float(value))
        record.metric_records["metrics"].pop("train_time", None)
    mean_time = sum(train_times) / len(train_times)

    # Call original default function
    aggregated_metrics = aggregate_metricrecords(
        records,
        weighting_metric_name,
    )
    aggregated_metrics["train_time"] = mean_time
    aggregated_metrics["qtty_records"] = len(records)
    return aggregated_metrics

def save_logs(
    result,
    strategy_name: str,
    semiasync_deg: int,
    fraction_slow: float,
    dataset_name: str,
    total_rounds: int,
    run_id: int,
    client_staleness: dict,
    metrics: dict,
    data_distribution: str,
) -> None:
    """Save the federated result in a csv and summary statistics in a json."""
    dataset_map = {
        "uoft-cs/cifar10": "cifar10",
        "ylecun/mnist": "mnist",
    }
    dataset = dataset_map.get(dataset_name, dataset_name)

    if strategy_name == "FedSaSync":
        config_dir = f"{strategy_name}_fs{fraction_slow}_m{semiasync_deg}"
    else:
        config_dir = f"{strategy_name}_fs{fraction_slow}"

    base_dir = f"_static/{dataset}_{data_distribution}/{config_dir}"
    os.makedirs(base_dir, exist_ok=True)

    csv_path = f"{base_dir}/ex{run_id}.csv"
    summary_path = f"{base_dir}/summary_ex{run_id}.json"

    # --- 1. Export round-level metrics to CSV ---
    rounds = sorted(result.evaluate_metrics_clientapp.keys())
    
    header = None
    for rnd in rounds:
        ev = result.evaluate_metrics_clientapp.get(rnd, {}) or {}
        tr = result.train_metrics_clientapp.get(rnd, {}) or {}
        if ev or tr:
            header = list(ev.keys()) + [k for k in tr.keys() if k not in ev]
            break

    if header:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for rnd in rounds:
                eval_metrics = result.evaluate_metrics_clientapp.get(rnd, {}) or {}
                train_metrics = result.train_metrics_clientapp.get(rnd, {}) or {}

                if not eval_metrics and not train_metrics:
                    continue

                row = []
                for key in header:
                    if eval_metrics and key in eval_metrics:
                        row.append(eval_metrics[key])
                    elif train_metrics and key in train_metrics:
                        row.append(train_metrics[key])
                    else:
                        row.append("")

                writer.writerow(row)
        log(INFO, "CSV saved: %s", csv_path)
    else:
        log(INFO, "No valid metrics found to save CSV for %s", csv_path)

    # --- 2. Compute and export global summary statistics to JSON ---
    formatted_staleness_lists = {str(k): v for k, v in client_staleness.items()}

    total_staleness_per_client = {
        k: sum(values) for k, values in formatted_staleness_lists.items()
    }

    formatted_participation = {
        k: total_rounds - total_staleness
        for k, total_staleness in total_staleness_per_client.items()
    }

    aggregated_staleness = (
        [sum(round_values) for round_values in zip(*client_staleness.values())]
        if client_staleness
        else []
    )

    summary_data = {
        "client_participation": formatted_participation,
        "round_staleness": aggregated_staleness,
        "client_staleness": total_staleness_per_client,
        "metrics": metrics,
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)
    
    log(INFO, "Summary JSON saved: %s", summary_path)
