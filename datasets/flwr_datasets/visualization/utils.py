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
"""Plotting utils."""

from flwr_datasets.visualization.constants import AXIS_TYPES, PLOT_TYPES, SIZE_UNITS


def _validate_parameters(
    plot_type: str, size_unit: str, partition_id_axis: str
) -> None:
    if plot_type not in PLOT_TYPES:
        raise ValueError(
            f"Invalid plot_type: {plot_type}. Must be one of {PLOT_TYPES}."
        )
    if size_unit not in SIZE_UNITS:
        raise ValueError(
            f"Invalid size_unit: {size_unit}. Must be one of {SIZE_UNITS}."
        )
    if partition_id_axis not in AXIS_TYPES:
        raise ValueError(
            f"Invalid partition_id_axis: {partition_id_axis}. "
            f"Must be one of {AXIS_TYPES}."
        )
