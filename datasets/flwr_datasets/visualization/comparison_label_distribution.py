# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Comparison of label distribution plotting."""

from typing import Any, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from flwr_datasets.common import EventType, event
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.visualization.constants import PLOT_TYPES
from flwr_datasets.visualization.label_distribution import plot_label_distributions


# pylint: disable=too-many-arguments,too-many-locals
# mypy: disable-error-code="call-overload"
def plot_comparison_label_distribution(  # pylint: disable=R0917
    partitioner_list: list[Partitioner],
    label_name: str | list[str],
    plot_type: Literal["bar", "heatmap"] = "bar",
    size_unit: Literal["percent", "absolute"] = "percent",
    max_num_partitions: int | None = 30,
    partition_id_axis: Literal["x", "y"] = "y",
    figsize: tuple[float, float] | None = None,
    subtitle: str = "Comparison of Per Partition Label Distribution",
    titles: list[str] | None = None,
    cmap: str | mcolors.Colormap | None = None,
    legend: bool = False,
    legend_title: str | None = None,
    verbose_labels: bool = True,
    plot_kwargs_list: list[dict[str, Any] | None] | None = None,
    legend_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure, list[Axes], list[pd.DataFrame]]:
    """Compare the label distribution across multiple partitioners.

    Parameters
    ----------
    partitioner_list : List[Partitioner]
        List of partitioners to be compared.
    label_name : Union[str, List[str]]
        Column name or list of column names identifying labels for each partitioner.
    plot_type :  Literal["bar", "heatmap"]
        Type of plot, either "bar" or "heatmap".
    size_unit : Literal["percent", "absolute"]
        "absolute" for raw counts, or "percent" to normalize values to 100%.
    max_num_partitions : Optional[int]
        Maximum number of partitions to include in the plot. If None, all partitions
        are included.
    partition_id_axis : Literal["x", "y"]
        Axis on which the partition IDs will be marked, either "x" or "y".
    figsize : Optional[Tuple[float, float]]
        Size of the figure. If None, a default size is calculated.
    subtitle : str
        Subtitle for the figure. Defaults to "Comparison of Per Partition Label
        Distribution"
    titles : Optional[List[str]]
        Titles for each subplot. If None, no titles are set.
    cmap : Optional[Union[str, mcolors.Colormap]]
        Colormap for determining the colorspace of the plot.
    legend : bool
        Whether to include a legend. If True, it will be included right-hand side after
        all the plots.
    legend_title : Optional[str]
        Title for the legend. If None, the defaults will be takes based on the type of
        plot.
    verbose_labels : bool
        Whether to use verbose versions of the labels.
    plot_kwargs_list: Optional[List[Optional[Dict[str, Any]]]]
        List of plot_kwargs. Any key value pair that can be passed to a plot function
        that are not supported directly. In case of the parameter doubling
        (e.g. specifying cmap here too) the chosen value will be taken from the
        explicit arguments (e.g. cmap specified as an argument to this function not
        the value in this dictionary).
    legend_kwargs: Optional[Dict[str, Any]]
        Any key value pair that can be passed to a figure.legend in case of bar plot or
        cbar_kws in case of heatmap that are not supported directly. In case of
        parameter doubling (e.g. specifying legend_title here too) the
        chosen value will be taken from the explicit arguments (e.g. legend_title
        specified as an argument to this function not the value in this dictionary).

    Returns
    -------
    fig : Figure
        The figure object containing the plots.
    axes_list : List[Axes]
        List of Axes objects for the plots.
    dataframe_list : List[pd.DataFrame]
        List of DataFrames used for each plot.

    Examples
    --------
    Compare the difference of using different alpha (concentration) parameters in
    DirichletPartitioner.

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>> from flwr_datasets.visualization import plot_comparison_label_distribution
    >>>
    >>> partitioner_list = []
    >>> alpha_list = [10_000.0, 100.0, 1.0, 0.1, 0.01, 0.00001]
    >>> for alpha in alpha_list:
    >>>     fds = FederatedDataset(
    >>>         dataset="cifar10",
    >>>         partitioners={
    >>>             "train": DirichletPartitioner(
    >>>                 num_partitions=20,
    >>>                 partition_by="label",
    >>>                 alpha=alpha,
    >>>                 min_partition_size=0,
    >>>             ),
    >>>         },
    >>>     )
    >>>     partitioner_list.append(fds.partitioners["train"])
    >>> fig, axes, dataframe_list = plot_comparison_label_distribution(
    >>>     partitioner_list=partitioner_list,
    >>>     label_name="label",
    >>>     titles=[f"Concentration = {alpha}" for alpha in alpha_list],
    >>> )
    """
    event(
        EventType.PLOT_COMPARISON_LABEL_DISTRIBUTION_CALLED,
        {
            "num_compare": len(partitioner_list),
            "plot_type": plot_type,
        },
    )
    num_partitioners = len(partitioner_list)
    if isinstance(label_name, str):
        label_name = [label_name] * num_partitioners
    elif isinstance(label_name, list):
        pass
    else:
        raise TypeError(
            f"Label name has to be of type List[str] or str but given "
            f"{type(label_name)}"
        )
    figsize = _initialize_comparison_figsize(figsize, num_partitioners)
    axes_sharing = _initialize_axis_sharing(size_unit, plot_type, partition_id_axis)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_partitioners,
        figsize=figsize,
        layout="constrained",
        **axes_sharing,
    )

    if titles is None:
        titles = ["" for _ in range(num_partitioners)]

    if plot_kwargs_list is None:
        plot_kwargs_list = [None] * num_partitioners

    dataframe_list = []
    for idx, (partitioner, single_label_name, plot_kwargs) in enumerate(
        zip(partitioner_list, label_name, plot_kwargs_list, strict=False)
    ):
        if idx == (num_partitioners - 1):
            *_, dataframe = plot_label_distributions(
                partitioner=partitioner,
                label_name=single_label_name,
                plot_type=plot_type,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                axis=axes[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=legend,
                legend_title=legend_title,
                verbose_labels=verbose_labels,
                plot_kwargs=plot_kwargs,
                legend_kwargs=legend_kwargs,
            )
            dataframe_list.append(dataframe)
        else:
            *_, dataframe = plot_label_distributions(
                partitioner=partitioner,
                label_name=single_label_name,
                plot_type=plot_type,
                size_unit=size_unit,
                partition_id_axis=partition_id_axis,
                axis=axes[idx],
                max_num_partitions=max_num_partitions,
                cmap=cmap,
                legend=False,
                plot_kwargs=plot_kwargs,
            )
            dataframe_list.append(dataframe)

    # Do not use the xlabel and ylabel on each subplot plot
    # (instead use global = per figure xlabel and ylabel)
    for idx, axis in enumerate(axes):
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_title(titles[idx])
    _set_tick_on_value_axes(axes, partition_id_axis, size_unit)

    # Set up figure xlabel and ylabel
    xlabel, ylabel = _initialize_comparison_xy_labels(
        plot_type, size_unit, partition_id_axis
    )
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(subtitle)

    fig.tight_layout()
    return fig, axes, dataframe_list


def _initialize_comparison_figsize(
    figsize: tuple[float, float] | None, num_partitioners: int
) -> tuple[float, float]:
    if figsize is not None:
        return figsize
    x_value = 4 + (num_partitioners - 1) * 2
    y_value = 4.8
    figsize = (x_value, y_value)
    return figsize


def _initialize_comparison_xy_labels(
    plot_type: Literal["bar", "heatmap"],
    size_unit: Literal["percent", "absolute"],
    partition_id_axis: Literal["x", "y"],
) -> tuple[str, str]:
    if plot_type == "bar":
        xlabel = "Partition ID"
        ylabel = "Class distribution" if size_unit == "percent" else "Class Count"
    elif plot_type == "heatmap":
        xlabel = "Partition ID"
        ylabel = "Label"
    else:
        raise ValueError(
            f"Invalid plot_type: {plot_type}. Must be one of {PLOT_TYPES}."
        )

    if partition_id_axis == "y":
        xlabel, ylabel = ylabel, xlabel

    return xlabel, ylabel


def _initialize_axis_sharing(
    size_unit: Literal["percent", "absolute"],
    plot_type: Literal["bar", "heatmap"],
    partition_id_axis: Literal["x", "y"],
) -> dict[str, bool]:
    # Do not intervene when the size_unit is percent and plot_type is heatmap
    if size_unit == "percent":
        return {}
    if plot_type == "heatmap":
        return {}
    if partition_id_axis == "x":
        return {"sharey": True}
    if partition_id_axis == "y":
        return {"sharex": True}
    return {"sharex": False, "sharey": False}


def _set_tick_on_value_axes(
    axes: list[Axes],
    partition_id_axis: Literal["x", "y"],
    size_unit: Literal["percent", "absolute"],
) -> None:
    if partition_id_axis == "x" and size_unit == "absolute":
        # Exclude this case due to sharing of y-axis (and thus y-ticks)
        # They must remain set and the number are displayed only on the first plot
        pass
    else:
        for axis in axes[1:]:
            axis.set_yticks([])
