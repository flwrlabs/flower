"""flowertune-medical: A Flower / FlowerTune app."""

from collections.abc import Iterable
from logging import INFO, WARN
from typing import Optional

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg


class FlowerTuneLlm(FedAvg):
    """Customised FedAvg strategy implementation.

    This class behaves just like FedAvg but also tracks the communication
    costs associated with `train` over FL rounds.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training."""
        messages = super().configure_train(server_round, arrays, config, grid)

        # Track communication costs
        self.comm_tracker.track(messages)

        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        # Track communication costs
        self.comm_tracker.track(replies)

        arrays, metrics = super().aggregate_train(server_round, replies)

        return arrays, metrics


class CommunicationTracker:
    """Communication costs tracker over FL rounds."""

    def __init__(self):
        self.curr_comm_cost = 0.0

    def track(self, messages: Iterable[Message]):
        comm_cost = (
            sum(
                record.count_bytes()
                for msg in messages
                if msg.has_content()
                for record in msg.content.array_records.values()
            )
            / 1024**2
        )

        self.curr_comm_cost += comm_cost
        log(
            INFO,
            "Communication budget: used %.2f MB (+%.2f MB this round) / 200,000 MB",
            self.curr_comm_cost,
            comm_cost,
        )

        if self.curr_comm_cost > 2e5:
            log(
                WARN,
                "The accumulated communication cost has exceeded 200,000 MB. "
                "Please consider reducing it if you plan to participate "
                "FlowerTune LLM Leaderboard.",
            )
