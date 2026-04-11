"""Custom Flower strategies for the GHBM baseline."""

from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
from typing import Any

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg

from ghbm.utils import clone_state_dict

StateDict = OrderedDict[str, torch.Tensor]
RoundLogger = Callable[[str, int, MetricRecord | None], None]


class EvalLoggingFedAvg(FedAvg):
    """FedAvg with shared evaluation scheduling and per-round logging."""

    def __init__(
        self,
        *args: Any,
        num_rounds: int,
        evaluate_every: int = 1,
        final_eval_window: int = 100,
        round_logger: RoundLogger | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1, got {num_rounds}")
        if evaluate_every < 1:
            raise ValueError(f"evaluate_every must be >= 1, got {evaluate_every}")
        if final_eval_window < 1:
            raise ValueError(f"final_eval_window must be >= 1, got {final_eval_window}")
        self.num_rounds = num_rounds
        self.evaluate_every = evaluate_every
        self.final_eval_window = final_eval_window
        self.round_logger = round_logger

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid,
    ) -> Iterable[Message]:
        """Run federated evaluation on scheduled rounds and throughout the final
        tail."""
        if not self._should_evaluate(server_round):
            return []
        return super().configure_evaluate(server_round, arrays, config, grid)

    def _should_evaluate(self, server_round: int) -> bool:
        """Return whether this round should run federated evaluation."""
        final_eval_start = max(1, self.num_rounds - self.final_eval_window + 1)
        if server_round >= final_eval_start:
            return True
        return server_round % self.evaluate_every == 0

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        """Aggregate training metrics and log them round by round."""
        arrays, metrics = super().aggregate_train(server_round, replies)
        if self.round_logger is not None:
            self.round_logger("train", server_round, metrics)
        return arrays, metrics

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        """Aggregate evaluation metrics and log them round by round."""
        metrics = super().aggregate_evaluate(server_round, replies)
        if self.round_logger is not None:
            self.round_logger("eval", server_round, metrics)
        return metrics


class GHBMStrategy(EvalLoggingFedAvg):
    """FedAvg plus the tau-lag server momentum used by GHBM."""

    def __init__(self, *args: Any, tau: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if tau < 1:
            raise ValueError(f"tau must be >= 1, got {tau}")
        self.tau = tau
        self._past_models: deque[StateDict] = deque(maxlen=tau)
        self._current_model: StateDict | None = None
        self._current_momentum: StateDict | None = None

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid,
    ) -> Iterable[Message]:
        """Attach the current GHBM server momentum to each training message."""
        self._current_model = clone_state_dict(arrays.to_torch_state_dict())
        messages = list(super().configure_train(server_round, arrays, config, grid))
        if self._current_momentum is not None:
            momentum_record = ArrayRecord(torch_state_dict=self._current_momentum)
            for message in messages:
                message.content["server_momentum"] = momentum_record
        return messages

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        """Aggregate client updates and refresh the tau-lag server momentum."""
        arrays, metrics = super().aggregate_train(server_round, replies)
        if arrays is None or self._current_model is None:
            return arrays, metrics

        self._past_models.append(self._current_model)
        aggregated_state = clone_state_dict(arrays.to_torch_state_dict())
        if len(self._past_models) == self.tau:
            oldest_model = self._past_models[0]
            self._current_momentum = OrderedDict(
                (
                    name,
                    (oldest_model[name] - aggregated_state[name]) / self.tau,
                )
                for name in aggregated_state
            )
        else:
            self._current_momentum = None
        return arrays, metrics
