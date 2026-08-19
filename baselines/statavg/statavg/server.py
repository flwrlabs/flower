"""statavg: A Flower Baseline."""

import os
import pickle
from logging import INFO
from pathlib import Path

from flwr.common import log
from flwr.server import Server

PROJECT_DIR = Path(os.path.abspath(__file__)).parent.parent


class ResultsSaverServer(Server):
    """Server to save history to disk."""

    def __init__(
        self,
        *,
        client_manager,
        strategy=None,
        results_saver_fn=None,
        run_config=None,
    ):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.results_saver_fn = results_saver_fn
        self.run_config = run_config

    def fit(self, num_rounds, timeout):
        """Run federated averaging for a number of rounds."""
        history, elapsed = super().fit(num_rounds, timeout)
        if self.results_saver_fn:
            log(INFO, "Results saver function provided. Executing")
            self.results_saver_fn(history, self.run_config)
        return history, elapsed


def save_results_and_clean_dir(history, run_config):
    """Save history and clean scaler dir."""
    results = {"history": history}
    results_path = PROJECT_DIR / run_config.results_save_dir / run_config.strategy_name
    results_path.mkdir(exist_ok=True, parents=True)
    with open(results_path / "results.pickle", "wb") as file:
        pickle.dump(results, file)
