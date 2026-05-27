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
"""Common components shared between server and client."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flwr.app import Error, MessageType, Metadata
    from flwr.app.constants import DEFAULT_TTL
    from flwr.app.message import (
        Array,
        ArrayRecord,
        ConfigRecord,
        Context,
        Message,
        MetricRecord,
        RecordDict,
    )
    from flwr.app.typing import (
        ConfigRecordValues,
        ConfigScalar,
        ConfigScalarList,
        MetricRecordValues,
        MetricScalar,
        MetricScalarList,
    )
    from flwr.common.constant import MessageTypeLegacy
    from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH
    from flwr.common.logger import configure, log
    from flwr.common.parameter import (
        bytes_to_ndarray,
        ndarray_to_bytes,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.common.record.conversion_utils import array_from_numpy
    from flwr.common.telemetry import EventType, event
    from flwr.common.typing import (
        ClientMessage,
        Code,
        Config,
        DisconnectRes,
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        GetParametersIns,
        GetParametersRes,
        GetPropertiesIns,
        GetPropertiesRes,
        Metrics,
        MetricsAggregationFn,
        NDArray,
        NDArrays,
        Parameters,
        Properties,
        ReconnectIns,
        Scalar,
        ServerMessage,
        Status,
    )
    from flwr.compat.common.record import (
        ConfigsRecord,
        MetricsRecord,
        ParametersRecord,
        RecordSet,
    )
    from flwr.supercore.date import now

_LAZY_EXPORTS = {
    "Array": ("flwr.app.message", "Array"),
    "ArrayRecord": ("flwr.app.message", "ArrayRecord"),
    "ClientMessage": ("flwr.common.typing", "ClientMessage"),
    "Code": ("flwr.common.typing", "Code"),
    "Config": ("flwr.common.typing", "Config"),
    "ConfigRecord": ("flwr.app.message", "ConfigRecord"),
    "ConfigRecordValues": ("flwr.app", "ConfigRecordValues"),
    "ConfigScalar": ("flwr.app", "ConfigScalar"),
    "ConfigScalarList": ("flwr.app", "ConfigScalarList"),
    "ConfigsRecord": ("flwr.compat.common.record", "ConfigsRecord"),
    "Context": ("flwr.app.message", "Context"),
    "DEFAULT_TTL": ("flwr.app", "DEFAULT_TTL"),
    "DisconnectRes": ("flwr.common.typing", "DisconnectRes"),
    "Error": ("flwr.app", "Error"),
    "EvaluateIns": ("flwr.common.typing", "EvaluateIns"),
    "EvaluateRes": ("flwr.common.typing", "EvaluateRes"),
    "EventType": ("flwr.common.telemetry", "EventType"),
    "FitIns": ("flwr.common.typing", "FitIns"),
    "FitRes": ("flwr.common.typing", "FitRes"),
    "GRPC_MAX_MESSAGE_LENGTH": ("flwr.common.grpc", "GRPC_MAX_MESSAGE_LENGTH"),
    "GetParametersIns": ("flwr.common.typing", "GetParametersIns"),
    "GetParametersRes": ("flwr.common.typing", "GetParametersRes"),
    "GetPropertiesIns": ("flwr.common.typing", "GetPropertiesIns"),
    "GetPropertiesRes": ("flwr.common.typing", "GetPropertiesRes"),
    "Message": ("flwr.app.message", "Message"),
    "MessageType": ("flwr.app", "MessageType"),
    "MessageTypeLegacy": ("flwr.common.constant", "MessageTypeLegacy"),
    "Metadata": ("flwr.app", "Metadata"),
    "MetricRecord": ("flwr.app.message", "MetricRecord"),
    "MetricRecordValues": ("flwr.app", "MetricRecordValues"),
    "MetricScalar": ("flwr.app", "MetricScalar"),
    "MetricScalarList": ("flwr.app", "MetricScalarList"),
    "Metrics": ("flwr.common.typing", "Metrics"),
    "MetricsAggregationFn": ("flwr.common.typing", "MetricsAggregationFn"),
    "MetricsRecord": ("flwr.compat.common.record", "MetricsRecord"),
    "NDArray": ("flwr.common.typing", "NDArray"),
    "NDArrays": ("flwr.common.typing", "NDArrays"),
    "Parameters": ("flwr.common.typing", "Parameters"),
    "ParametersRecord": ("flwr.compat.common.record", "ParametersRecord"),
    "Properties": ("flwr.common.typing", "Properties"),
    "ReconnectIns": ("flwr.common.typing", "ReconnectIns"),
    "RecordDict": ("flwr.app.message", "RecordDict"),
    "RecordSet": ("flwr.compat.common.record", "RecordSet"),
    "Scalar": ("flwr.common.typing", "Scalar"),
    "ServerMessage": ("flwr.common.typing", "ServerMessage"),
    "Status": ("flwr.common.typing", "Status"),
    "array_from_numpy": ("flwr.common.record.conversion_utils", "array_from_numpy"),
    "bytes_to_ndarray": ("flwr.common.parameter", "bytes_to_ndarray"),
    "configure": ("flwr.common.logger", "configure"),
    "event": ("flwr.common.telemetry", "event"),
    "log": ("flwr.common.logger", "log"),
    "ndarray_to_bytes": ("flwr.common.parameter", "ndarray_to_bytes"),
    "ndarrays_to_parameters": ("flwr.common.parameter", "ndarrays_to_parameters"),
    "now": ("flwr.supercore.date", "now"),
    "parameters_to_ndarrays": ("flwr.common.parameter", "parameters_to_ndarrays"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve compatibility exports."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = [
    "Array",
    "ArrayRecord",
    "ClientMessage",
    "Code",
    "Config",
    "ConfigRecord",
    "ConfigRecordValues",
    "ConfigScalar",
    "ConfigScalarList",
    "ConfigsRecord",
    "Context",
    "DEFAULT_TTL",
    "DisconnectRes",
    "Error",
    "EvaluateIns",
    "EvaluateRes",
    "EventType",
    "FitIns",
    "FitRes",
    "GRPC_MAX_MESSAGE_LENGTH",
    "GetParametersIns",
    "GetParametersRes",
    "GetPropertiesIns",
    "GetPropertiesRes",
    "Message",
    "MessageType",
    "MessageTypeLegacy",
    "Metadata",
    "MetricRecord",
    "MetricRecordValues",
    "MetricScalar",
    "MetricScalarList",
    "Metrics",
    "MetricsAggregationFn",
    "MetricsRecord",
    "NDArray",
    "NDArrays",
    "Parameters",
    "ParametersRecord",
    "Properties",
    "ReconnectIns",
    "RecordDict",
    "RecordSet",
    "Scalar",
    "ServerMessage",
    "Status",
    "array_from_numpy",
    "bytes_to_ndarray",
    "configure",
    "event",
    "log",
    "ndarray_to_bytes",
    "ndarrays_to_parameters",
    "now",
    "parameters_to_ndarrays",
]
