import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics
import numpy as np


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # 聚合metric
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


class GLFCServer(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
            failures: List[BaseException],
    ) -> Tuple[float, dict]:
        if not results:
            return 0.0, {}

        # 调用父类的聚合
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # 自定义聚合逻辑
        aggregated_metrics = weighted_average(
            [(m.num_examples, m.metrics) for _, m in results]
        )

        return loss, aggregated_metrics