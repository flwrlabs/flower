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
"""Utility functions for model quantization."""


import numpy as np

from flwr.common import NDArrayFloat, NDArrayInt


def _stochastic_round(arr: NDArrayFloat) -> NDArrayInt:
    ret: NDArrayInt = np.ceil(arr).astype(np.int32)
    rand_arr = np.random.rand(*ret.shape)
    if len(ret.shape) == 0:
        if rand_arr < ret - arr:
            ret -= 1
    else:
        ret[rand_arr < ret - arr] -= 1
    return ret


def quantize(
    parameters: list[NDArrayFloat], clipping_range: float, target_range: int
) -> list[NDArrayInt]:
    """Quantize float Numpy arrays to integer Numpy arrays."""
    quantized_list: list[NDArrayInt] = []
    quantizer = target_range / (2 * clipping_range)
    for arr in parameters:
        # Stochastic quantization
        pre_quantized = (
            np.clip(arr, -clipping_range, clipping_range) + clipping_range
        ) * quantizer
        quantized = _stochastic_round(pre_quantized)
        quantized_list.append(quantized)
    return quantized_list


# Dequantize parameters to range [-clipping_range, clipping_range]
def dequantize(
    quantized_parameters: list[NDArrayInt],
    clipping_range: float,
    target_range: int,
) -> list[NDArrayFloat]:
    """Dequantize integer Numpy arrays to float Numpy arrays."""
    reverse_quantized_list: list[NDArrayFloat] = []
    quantizer = (2 * clipping_range) / target_range
    shift = -clipping_range
    for arr in quantized_parameters:
        recon_arr = arr.view(np.ndarray).astype(float)
        recon_arr = recon_arr * quantizer + shift
        reverse_quantized_list.append(recon_arr)
    return reverse_quantized_list
