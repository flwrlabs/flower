# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Utility functions for differential privacy."""


from logging import WARNING

import numpy as np

from flwr.common.logger import log
from flwr.compat.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.compat.common.typing import NDArrays, Parameters

KEY_CLIPPING_NORM = "clipping_norm"
KEY_NORM_BIT = "norm_bit"
CLIENTS_DISCREPANCY_WARNING = (
    "The number of clients returning parameters (%s)"
    " differs from the number of sampled clients (%s)."
    " This could impact the differential privacy guarantees,"
    " potentially leading to privacy leakage or inadequate noise calibration."
)


def get_norm(input_arrays: NDArrays) -> float:
    """Compute the L2 norm of the flattened input."""
    array_norms = [np.linalg.norm(array.flat) for array in input_arrays]
    # pylint: disable=consider-using-generator
    return float(np.sqrt(sum([norm**2 for norm in array_norms])))


def add_gaussian_noise_inplace(input_arrays: NDArrays, std_dev: float) -> None:
    """Add Gaussian noise to each element of the input arrays."""
    for array in input_arrays:
        noise = np.random.normal(0, std_dev, array.shape).astype(array.dtype)
        array += noise


def clip_inputs_inplace(input_arrays: NDArrays, clipping_norm: float) -> None:
    """Clip model update based on the clipping norm in-place.

    FlatClip method of the paper: https://arxiv.org/abs/1710.06963
    """
    input_norm = get_norm(input_arrays)
    scaling_factor = min(1, clipping_norm / input_norm)
    for array in input_arrays:
        array *= scaling_factor


def compute_stdv(
    noise_multiplier: float, clipping_norm: float, num_sampled_clients: int
) -> float:
    """Compute standard deviation for noise addition.

    Paper: https://arxiv.org/abs/1710.06963
    """
    return float((noise_multiplier * clipping_norm) / num_sampled_clients)


def compute_clip_model_update(
    param1: NDArrays, param2: NDArrays, clipping_norm: float
) -> None:
    """Compute model update (param1 - param2) and clip it.

    Then add the clipped value to param1."""
    model_update = [np.subtract(x, y) for (x, y) in zip(param1, param2, strict=True)]
    clip_inputs_inplace(model_update, clipping_norm)

    for i, _ in enumerate(param2):
        param1[i] = param2[i] + model_update[i]


def adaptive_clip_inputs_inplace(input_arrays: NDArrays, clipping_norm: float) -> bool:
    """Clip model update based on the clipping norm in-place.

    It returns true if scaling_factor < 1 which is used for norm_bit
    FlatClip method of the paper: https://arxiv.org/abs/1710.06963
    """
    input_norm = get_norm(input_arrays)
    scaling_factor = min(1, clipping_norm / input_norm)
    for array in input_arrays:
        array *= scaling_factor
    return scaling_factor < 1


def compute_adaptive_clip_model_update(
    param1: NDArrays, param2: NDArrays, clipping_norm: float
) -> bool:
    """Compute model update, clip it, then add the clipped value to param1.

    model update = param1 - param2
    Return the norm_bit
    """
    model_update = [np.subtract(x, y) for (x, y) in zip(param1, param2, strict=True)]
    norm_bit = adaptive_clip_inputs_inplace(model_update, clipping_norm)

    for i, _ in enumerate(param2):
        param1[i] = param2[i] + model_update[i]

    return norm_bit


def add_gaussian_noise_to_params(
    model_params: Parameters,
    noise_multiplier: float,
    clipping_norm: float,
    num_sampled_clients: int,
) -> Parameters:
    """Add gaussian noise to model parameters."""
    model_params_ndarrays = parameters_to_ndarrays(model_params)
    add_gaussian_noise_inplace(
        model_params_ndarrays,
        compute_stdv(noise_multiplier, clipping_norm, num_sampled_clients),
    )
    return ndarrays_to_parameters(model_params_ndarrays)


def compute_adaptive_noise_params(
    noise_multiplier: float,
    num_sampled_clients: float,
    clipped_count_stddev: float | None,
) -> tuple[float, float]:
    """Compute noising parameters for the adaptive clipping.

    Paper: https://arxiv.org/abs/1905.03871
    """
    if noise_multiplier > 0:
        if clipped_count_stddev is None:
            clipped_count_stddev = num_sampled_clients / 20
        if noise_multiplier >= 2 * clipped_count_stddev:
            raise ValueError(
                f"If not specified, `clipped_count_stddev` is set to "
                f"`num_sampled_clients`/20 by default. This value "
                f"({num_sampled_clients / 20}) is too low to achieve the "
                f"desired effective `noise_multiplier` ({noise_multiplier}). "
                f"Consider increasing `clipped_count_stddev` or decreasing "
                f"`noise_multiplier`."
            )
        noise_multiplier_value = (
            noise_multiplier ** (-2) - (2 * clipped_count_stddev) ** (-2)
        ) ** -0.5

        adding_noise = noise_multiplier_value / noise_multiplier
        if adding_noise >= 2:
            log(
                WARNING,
                "A significant amount of noise (%s) has to be "
                "added. Consider increasing `clipped_count_stddev` or "
                "`num_sampled_clients`.",
                adding_noise,
            )

    else:
        if clipped_count_stddev is None:
            clipped_count_stddev = 0.0
        noise_multiplier_value = 0.0

    return clipped_count_stddev, noise_multiplier_value


def add_localdp_gaussian_noise_to_params(
    model_params: Parameters, sensitivity: float, epsilon: float, delta: float
) -> Parameters:
    """Add local DP gaussian noise to model parameters."""
    model_params_ndarrays = parameters_to_ndarrays(model_params)
    add_gaussian_noise_inplace(
        model_params_ndarrays,
        sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon,
    )
    return ndarrays_to_parameters(model_params_ndarrays)
