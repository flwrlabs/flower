"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from flwr.server.history import History


def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    expected_maximum: float,
    suffix: Optional[str] = "",
) -> None:
    """Define a function to plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    expected_maximum : float
        The expected maximum accuracy from the original paper.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    rounds, values = zip(*metric_dict["accuracy"])
    fig = plt.figure()
    axis = fig.add_subplot(111)
    plt.plot(np.asarray(rounds), np.asarray(values), label="FedAvg")
    # Set expected graph for data
    plt.axhline(
        y=expected_maximum,
        color="r",
        linestyle="--",
        label=f"Paper's best result @{expected_maximum}",
    )
    # Set paper's results
    plt.axhline(
        y=0.99,
        color="silver",
        label="Paper's baseline @0.9900",
    )
    plt.ylim([0.97, 1])
    plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    # Set the apect ratio to 1.0
    xleft, xright = axis.get_xlim()
    ybottom, ytop = axis.get_ylim()
    axis.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * 1.0)

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()
