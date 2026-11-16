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
"""Utility functions for performing operations on Numpy NDArrays."""


from typing import Any

import numpy as np
from numpy.typing import NDArray


def factor_combine(factor: int, parameters: list[NDArray[Any]]) -> list[NDArray[Any]]:
    """Combine factor with parameters."""
    return [np.array([factor])] + parameters


def factor_extract(
    parameters: list[NDArray[Any]],
) -> tuple[int, list[NDArray[Any]]]:
    """Extract factor from parameters."""
    return parameters[0][0], parameters[1:]


def get_parameters_shape(parameters: list[NDArray[Any]]) -> list[tuple[int, ...]]:
    """Get dimensions of each NDArray in parameters."""
    return [arr.shape for arr in parameters]


def parameters_addition(
    parameters1: list[NDArray[Any]], parameters2: list[NDArray[Any]]
) -> list[NDArray[Any]]:
    """Add two parameters."""
    return [parameters1[idx] + parameters2[idx] for idx in range(len(parameters1))]


def parameters_subtraction(
    parameters1: list[NDArray[Any]], parameters2: list[NDArray[Any]]
) -> list[NDArray[Any]]:
    """Subtract parameters from the other parameters."""
    return [parameters1[idx] - parameters2[idx] for idx in range(len(parameters1))]


def parameters_mod(parameters: list[NDArray[Any]], divisor: int) -> list[NDArray[Any]]:
    """Take mod of parameters with an integer divisor."""
    if bin(divisor).count("1") == 1:
        msk = divisor - 1
        return [parameters[idx] & msk for idx in range(len(parameters))]
    return [parameters[idx] % divisor for idx in range(len(parameters))]


def parameters_multiply(
    parameters: list[NDArray[Any]], multiplier: int | float
) -> list[NDArray[Any]]:
    """Multiply parameters by an integer/float multiplier."""
    return [parameters[idx] * multiplier for idx in range(len(parameters))]
