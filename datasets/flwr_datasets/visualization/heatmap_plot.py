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
"""Label distribution heatmap plotting."""

from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def _plot_heatmap(  # pylint: disable=R0913, R0917
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
                partition_id_axis=partition_id_axis,
                num_partitions=dataframe.shape[0],
                num_labels=dataframe.shape[1],
            )
        _, axis = plt.subplots(figsize=figsize)

    # Handle plot_kwargs
    if plot_kwargs is None:
        plot_kwargs = {}

    # Handle optional parameters
    if colormap is not None:
        plot_kwargs["cmap"] = colormap
    elif "cmap" not in plot_kwargs:
        plot_kwargs["cmap"] = sns.light_palette("seagreen", as_cmap=True)

    if "fmt" not in plot_kwargs:
        plot_kwargs["fmt"] = ",d" if size_unit == "absolute" else "0.2f"

    if legend_kwargs is None:
        legend_kwargs = {}
    if legend:
        plot_kwargs["cbar"] = True

        if legend_title is not None:
            legend_kwargs["label"] = legend_title
        else:
            legend_kwargs["label"] = _initialize_cbar_title(size_unit)
    else:
        plot_kwargs["cbar"] = False

    if partition_id_axis == "x":
        dataframe = dataframe.T

    sns.heatmap(
        dataframe,
        ax=axis,
        **plot_kwargs,
        cbar_kws=legend_kwargs,
    )
    axis.set_title(title)
    return axis


def _initialize_figsize(
    partition_id_axis: str,
    num_partitions: int,
    num_labels: int,
) -> tuple[float, float]:
    figsize = (0.0, 0.0)
    if partition_id_axis == "x":
        figsize = (3 * np.sqrt(num_partitions), np.sqrt(num_labels))
    elif partition_id_axis == "y":
        figsize = (3 * np.sqrt(num_labels), np.sqrt(num_partitions))

    return figsize


def _initialize_cbar_title(size_unit: str) -> str | None:
    return "Count" if size_unit == "absolute" else "Percent %"
