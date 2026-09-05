"""FedSaSync: Semi-asynchronous Federated Learning in Flower."""
import csv
import os
from logging import INFO

from flwr.common import MetricRecord, RecordDict, log
from flwr.serverapp.strategy.result import Result
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
    result: Result,
    strategy_name: str,
    semiasync_deg: int,
    number_slow: int,
    dataset_name: str,
) -> None:
    """Save the federated result in a csv.

    Utility function to write the federated final result on a csv, defined by
    dataset_name, strategy_name, number_slow, and semiasync_deg.

    Parameters
    ----------
    result : Result
        Result object containing metrics collected across rounds.
    strategy_name : str
        Strategy name used to build the output path.
    semiasync_deg : int
        Semi-asynchronous degree (M) of the experiment.
    number_slow : int
        Number of slow clients in the experiment.
    dataset_name : str
        Dataset name used for the experiment.
    """
    # Map dataset identifiers to short names used in paths
    dataset_map = {
        "uoft-cs/cifar10": "cifar10",
        "ylecun/mnist": "mnist",
    }
    dataset = dataset_map.get(dataset_name, dataset_name)

    # Build output path depending on strategy type
    if strategy_name == "FedSaSync":
        path = (
            f"_static/{dataset}/"
            f"{strategy_name}_fs{number_slow}_m{semiasync_deg}.csv"
        )
    else:
        path = (
            f"_static/{dataset}/"
            f"{strategy_name}_fs{number_slow}.csv"
        )

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Fixed column order for easier post-processing
        writer.writerow([
            "time",
            "loss",
            "train_time",
            "qtty_records",
        ])

        # Use rounds available in evaluation metrics
        rounds = sorted(result.evaluate_metrics_clientapp.keys())
        eval_metrics: MetricRecord | None
        train_metrics: MetricRecord | None

        for rnd in rounds:
            # Get evaluation metrics for this round (if available)
            eval_metrics = result.evaluate_metrics_clientapp.get(rnd, None)

            # Get training metrics for this round (if available)
            train_metrics = result.train_metrics_clientapp.get(rnd, None)

            if eval_metrics is None:
                continue
            if train_metrics is None:
                continue

            # Write a single row per round
            writer.writerow([
                eval_metrics.get("time", ""),
                eval_metrics.get("eval_loss", ""),
                train_metrics.get("train_time", ""),
                train_metrics.get("qtty_records", ""),
            ])

    # Log completion
    log(INFO, "CSV saved: %s", path)
