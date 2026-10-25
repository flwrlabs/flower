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
"""Legacy common type definitions."""


from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float64]
NDArrays = list[NDArray]

Scalar = bool | bytes | float | int | str
Value = (
    bool
    | bytes
    | float
    | int
    | str
    | list[bool]
    | list[bytes]
    | list[float]
    | list[int]
    | list[str]
)
Metrics = dict[str, Scalar]
MetricsAggregationFn = Callable[[list[tuple[int, Metrics]]], Metrics]

Config = dict[str, Scalar]
Properties = dict[str, Scalar]


class Code(Enum):
    """Client status codes."""

    OK = 0
    GET_PROPERTIES_NOT_IMPLEMENTED = 1
    GET_PARAMETERS_NOT_IMPLEMENTED = 2
    FIT_NOT_IMPLEMENTED = 3
    EVALUATE_NOT_IMPLEMENTED = 4


@dataclass
class Status:
    """Client status."""

    code: Code
    message: str


@dataclass
class Parameters:
    """Model parameters."""

    tensors: list[bytes]
    tensor_type: str


@dataclass
class GetParametersIns:
    """Parameters request for a client."""

    config: Config


@dataclass
class GetParametersRes:
    """Response when asked to return parameters."""

    status: Status
    parameters: Parameters


@dataclass
class FitIns:
    """Fit instructions for a client."""

    parameters: Parameters
    config: dict[str, Scalar]


@dataclass
class FitRes:
    """Fit response from a client."""

    status: Status
    parameters: Parameters
    num_examples: int
    metrics: dict[str, Scalar]


@dataclass
class EvaluateIns:
    """Evaluate instructions for a client."""

    parameters: Parameters
    config: dict[str, Scalar]


@dataclass
class EvaluateRes:
    """Evaluate response from a client."""

    status: Status
    loss: float
    num_examples: int
    metrics: dict[str, Scalar]


@dataclass
class GetPropertiesIns:
    """Properties request for a client."""

    config: Config


@dataclass
class GetPropertiesRes:
    """Properties response from a client."""

    status: Status
    properties: Properties


@dataclass
class ReconnectIns:
    """ReconnectIns message from server to client."""

    seconds: int | None


@dataclass
class DisconnectRes:
    """DisconnectRes message from client to server."""

    reason: str


@dataclass
class ServerMessage:
    """ServerMessage is a container used to hold one instruction message."""

    get_properties_ins: GetPropertiesIns | None = None
    get_parameters_ins: GetParametersIns | None = None
    fit_ins: FitIns | None = None
    evaluate_ins: EvaluateIns | None = None


@dataclass
class ClientMessage:
    """ClientMessage is a container used to hold one result message."""

    get_properties_res: GetPropertiesRes | None = None
    get_parameters_res: GetParametersRes | None = None
    fit_res: FitRes | None = None
    evaluate_res: EvaluateRes | None = None
