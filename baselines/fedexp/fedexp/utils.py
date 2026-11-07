"""Utility functions for FedExp."""

import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from flwr.common import NDArrays
from flwr.server import History
from omegaconf import DictConfig
from torch.nn import Module


def plot_metric_from_history(
    hist: History,
    save_plot_path: str,
    suffix: Optional[str] = "",
    cfg: Optional[DictConfig] = None,
) -> None:
    """Plot the metrics from the history of the server.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : str
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    cfg : Optional[DictConfig]
        Optional dictionary containing the configuration of the experiment.
    """
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    rounds, values_accuracy = zip(*metric_dict["accuracy"])
    _, axs = plt.subplots()
    # Set the title
    axs.set_title(f"{cfg.strategy.algorithm} | {cfg.dataset_config.name} | Seed {cfg.seed}")
    axs.plot(np.asarray(rounds), np.asarray(values_accuracy))
    axs.set_ylabel("Accuracy")
    axs.set_xlabel("Rounds")
    fig_name = "_".join([metric_type, "metrics", suffix]) + ".png"
    plt.savefig(Path(save_plot_path) / Path(fig_name))
    plt.close()


def seed_everything(seed):
    """Seed everything for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_parameters(net: Module) -> NDArrays:
    """Get the parameters of the network."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
