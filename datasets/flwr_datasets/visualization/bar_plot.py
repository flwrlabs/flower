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
"""Label distribution bar plotting."""

from typing import Any

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _plot_bar(  # pylint: disable=R0912, R0913, R0914, R0917
    dataframe: pd.DataFrame,
    axis: Axes | None,
    figsize: tuple[float, float] | None,
    title: str,
    colormap: str | mcolors.Colormap | None,
    partition_id_axis: str,
    size_unit: str,
    legend: bool,
    legend_title: str | None,
    plot_kwargs: dict[str, Any] | None,
    legend_kwargs: dict[str, Any] | None,
) -> Axes:
    if axis is None:
        if figsize is None:
            figsize = _initialize_figsize(
                partition_id_axis=partition_id_axis, num_partitions=dataframe.shape[0]
            )
        _, axis = plt.subplots(figsize=figsize)

    # Handle plot_kwargs
    if plot_kwargs is None:
        plot_kwargs = {}

    kind = "bar" if partition_id_axis == "x" else "barh"
    if "kind" not in plot_kwargs:
        plot_kwargs["kind"] = kind

    # Handle non-optional parameters
    plot_kwargs["title"] = title

    # Handle optional parameters
    if colormap is not None:
        plot_kwargs["colormap"] = colormap
    elif "colormap" not in plot_kwargs:
        plot_kwargs["colormap"] = "RdYlGn"

    if "xlabel" not in plot_kwargs and "ylabel" not in plot_kwargs:
        xlabel, ylabel = _initialize_xy_labels(
            size_unit=size_unit, partition_id_axis=partition_id_axis
        )
        plot_kwargs["xlabel"] = xlabel
        plot_kwargs["ylabel"] = ylabel

    # Make the x ticks readable (they appear 90 degrees rotated by default)
    if "rot" not in plot_kwargs:
        plot_kwargs["rot"] = 0

    # Handle hard-coded parameters
    # Legend is handled separately (via axes.legend call not in the plot())
    if "legend" not in plot_kwargs:
        plot_kwargs["legend"] = False

    # Make the bar plot stacked
    if "stacked" not in plot_kwargs:
        plot_kwargs["stacked"] = True

    axis_df: Axes = dataframe.plot(
        ax=axis,
        **plot_kwargs,
    )
    assert axis_df is not None, "axis is None after plotting using DataFrame.plot()"

    if legend:
        if legend_kwargs is None:
            legend_kwargs = {}

        if legend_title is not None:
            legend_kwargs["title"] = legend_title
        elif "title" not in legend_kwargs:
            legend_kwargs["title"] = "Labels"

        if "loc" not in legend_kwargs:
            legend_kwargs["loc"] = "outside center right"

        if "bbox_to_anchor" not in legend_kwargs:
            max_len_label_str = max(len(str(column)) for column in dataframe.columns)
            shift = min(0.05 + max_len_label_str / 100, 0.15)
            legend_kwargs["bbox_to_anchor"] = (1.0 + shift, 0.5)

        handles, legend_labels = axis_df.get_legend_handles_labels()
        figure = axis_df.figure
        assert isinstance(figure, Figure), "figure extraction from axes is not a Figure"
        _ = figure.legend(
            handles=handles[::-1], labels=legend_labels[::-1], **legend_kwargs
        )

    # Heuristic to make the partition id on xticks non-overlapping
    if partition_id_axis == "x":
        xticklabels = axis_df.get_xticklabels()
        if len(xticklabels) > 20:
            # Make every other xtick label not visible
            for i, label in enumerate(xticklabels):
                if i % 2 == 1:
                    label.set_visible(False)
    return axis_df


def _initialize_figsize(
    partition_id_axis: str,
    num_partitions: int,
) -> tuple[float, float]:
    figsize = (0.0, 0.0)
    if partition_id_axis == "x":
        figsize = (6.4, 4.8)
    elif partition_id_axis == "y":
        figsize = (6.4, np.sqrt(num_partitions))
    return figsize


def _initialize_xy_labels(size_unit: str, partition_id_axis: str) -> tuple[str, str]:
    xlabel = "Partition ID"
    ylabel = "Count" if size_unit == "absolute" else "Percent %"

    if partition_id_axis == "y":
        xlabel, ylabel = ylabel, xlabel

    return xlabel, ylabel
