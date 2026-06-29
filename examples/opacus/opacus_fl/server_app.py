"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

import logging

from flwr.app import ArrayRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from opacus_fl.task import Net

# Opacus logger seems to change the flwr logger to DEBUG level. Set back to INFO
logging.getLogger("flwr").setLevel(logging.INFO)

app = ServerApp()


def weighted_mean(
    replies: list[RecordDict],
    metric_key: str,
    weight_key: str,
) -> float:
    weighted_value_sum = 0.0
    total_weight = 0.0

    for rd in replies:
        metrics: MetricRecord = rd["metrics"]
        weight = float(metrics.get(weight_key, 0.0))
        value = float(metrics.get(metric_key, 0.0))

        weighted_value_sum += weight * value
        total_weight += weight

    return weighted_value_sum / total_weight if total_weight > 0 else 0.0


def weighted_eval(replies: list[RecordDict], weight_key: str) -> MetricRecord:
    return MetricRecord(
        {
            "accuracy": weighted_mean(replies, "accuracy", weight_key),
            "loss": weighted_mean(replies, "loss", weight_key),
        }
    )


def aggregate_train(replies: list[RecordDict], weight_key: str) -> MetricRecord:
    epsilons = [
        float(rd["metrics"]["epsilon"]) for rd in replies if "epsilon" in rd["metrics"]
    ]

    noise_multiplier_weighted_avg = weighted_mean(
        replies, "noise_multiplier", weight_key
    )

    out = {"noise_multiplier_weighted_avg": noise_multiplier_weighted_avg}

    if epsilons:
        out["max_epsilon"] = max(epsilons)
        out["mean_epsilon"] = sum(epsilons) / len(epsilons)

    return MetricRecord(out)


@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds = int(context.run_config["num-server-rounds"])

    initial_arrays = ArrayRecord(Net().state_dict())

    strategy = FedAvg(
        evaluate_metrics_aggr_fn=weighted_eval,
        train_metrics_aggr_fn=aggregate_train,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
    )
    print(result)
