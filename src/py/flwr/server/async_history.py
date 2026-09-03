"""
A wrapper around the flower History class that offers centralized and distributed metrics per timestamp instead of per round.
It also groups distributed_fit metrics per client instead of per client instead of per round.

losses_centralized: [ (timestamp1, value1) , .... ]

metrics_centralized: {
    "accuracy": [ (timestamp1, value1) , .... ]
}
metrics_distributed: {
    "client_ids": [ (timestamp1, [cid1, cid2, cid3]) ... ]
    "accuracy": [ (timestamp1, [value1, value2, value3]) , .... ]
} 
metrics_distributed_fit_async: {
    "accuracy": { 
        cid1: [
            (timestamp1, value1), 
            (timestamp2, value2), 
            (timestamp3, value3)
            ...
            ],
        ...
    }
    ...
}
# Metrics collected after each merge into the global model. (Global model evaluated centrally after merge.) 

DEPRECATED: This takes too much time and serializes the training process. Will be removed in the future.

metrics_centralized_async: {
    "accuracy": [ (timestamp1, value1) , .... ]
}
Note: value1 is collected at timestamp1 in metrics_distributed_fit.
"""
from flwr.server.history import History
from typing import Dict
from flwr.common.typing import Scalar
from threading import Lock

class AsyncHistory(History):

    def __init__(self) -> None:
        self.metrics_distributed_fit_async = {}
        self.metrics_centralized_async = {} # metrics aggregated after each merge into the global model.
        self.losses_centralized_async = []
        super().__init__()

    def add_metrics_distributed_fit_async(
        self, client_id: str, metrics: Dict[str, Scalar], timestamp: float
    ) -> None:
        """Add metrics entries (from distributed fit)."""
        lock = Lock()
        with lock:
            for key in metrics:
                if key not in self.metrics_distributed_fit_async:
                    self.metrics_distributed_fit_async[key] = {}
                if client_id not in self.metrics_distributed_fit_async[key]:
                    self.metrics_distributed_fit_async[key][client_id] = []
                self.metrics_distributed_fit_async[key][client_id].append((timestamp, metrics[key]))

    def add_metrics_centralized_async(self, metrics: Dict[str, Scalar], timestamp: float) -> None:
        """Add metrics entries (from centralized evaluation)."""
        lock = Lock()
        with lock:
            for metric in metrics:
                if metric not in self.metrics_centralized_async:
                    self.metrics_centralized_async[metric] = []
                self.metrics_centralized_async[metric].append((timestamp, metrics[metric]))

    def add_loss_centralized_async(self, timestamp: float, loss: float) -> None:
        """Add loss entries (from centralized evaluation)."""
        lock = Lock()
        with lock:
            self.losses_centralized_async.append((timestamp, loss))

    def add_loss_centralized(self, timestamp: float, loss: float) -> None:
        return super().add_loss_centralized(timestamp, loss)

    def add_loss_distributed(self, timestamp: float, loss: float) -> None:
        return super().add_loss_distributed(timestamp, loss)
    
    def add_metrics_centralized(self, timestamp: float, metrics: Dict[str, bool | bytes | float | int | str]) -> None:
        return super().add_metrics_centralized(timestamp, metrics)
    
    def add_metrics_distributed(self, timestamp: float, metrics: Dict[str, bool | bytes | float | int | str]) -> None:
        return super().add_metrics_distributed(timestamp, metrics)