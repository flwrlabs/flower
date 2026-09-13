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
"""Flower type definitions."""


from flwr.compat.common import typing as _compat_typing

# Compatibility shims to avoid breaking `from flwr.common.typing import [...]`
ClientMessage = _compat_typing.ClientMessage
Code = _compat_typing.Code
Config = _compat_typing.Config
DisconnectRes = _compat_typing.DisconnectRes
EvaluateIns = _compat_typing.EvaluateIns
EvaluateRes = _compat_typing.EvaluateRes
FitIns = _compat_typing.FitIns
FitRes = _compat_typing.FitRes
GetParametersIns = _compat_typing.GetParametersIns
GetParametersRes = _compat_typing.GetParametersRes
GetPropertiesIns = _compat_typing.GetPropertiesIns
GetPropertiesRes = _compat_typing.GetPropertiesRes
Metrics = _compat_typing.Metrics
MetricsAggregationFn = _compat_typing.MetricsAggregationFn
NDArray = _compat_typing.NDArray
NDArrayFloat = _compat_typing.NDArrayFloat
NDArrayInt = _compat_typing.NDArrayInt
NDArrays = _compat_typing.NDArrays
Parameters = _compat_typing.Parameters
Properties = _compat_typing.Properties
ReconnectIns = _compat_typing.ReconnectIns
Scalar = _compat_typing.Scalar
ServerMessage = _compat_typing.ServerMessage
Status = _compat_typing.Status
Value = _compat_typing.Value
