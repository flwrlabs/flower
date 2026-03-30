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


from flwr.app.message_type import MessageType as MessageType

from ..app.error import Error as Error
from ..app.metadata import Metadata as Metadata
from ..supercore.date import now as now
from .constant import MessageTypeLegacy as MessageTypeLegacy
from .context import Context as Context
from .grpc import GRPC_MAX_MESSAGE_LENGTH
from .logger import configure as configure
from .logger import log as log
from .message import DEFAULT_TTL
from .message import Message as Message
from .parameter import bytes_to_ndarray as bytes_to_ndarray
from .parameter import ndarray_to_bytes as ndarray_to_bytes
from .parameter import ndarrays_to_parameters as ndarrays_to_parameters
from .parameter import parameters_to_ndarrays as parameters_to_ndarrays
from .record import Array as Array
from .record import ArrayRecord as ArrayRecord
from .record import ConfigRecord as ConfigRecord
from .record import ConfigsRecord as ConfigsRecord
from .record import MetricRecord as MetricRecord
from .record import MetricsRecord as MetricsRecord
from .record import ParametersRecord as ParametersRecord
from .record import RecordDict as RecordDict
from .record import RecordSet as RecordSet
from .record import array_from_numpy as array_from_numpy
from .telemetry import EventType as EventType
from .telemetry import event as event
from .typing import ClientMessage as ClientMessage
from .typing import Code as Code
from .typing import Config as Config
from .typing import ConfigRecordValues as ConfigRecordValues
from .typing import DisconnectRes as DisconnectRes
from .typing import EvaluateIns as EvaluateIns
from .typing import EvaluateRes as EvaluateRes
from .typing import FitIns as FitIns
from .typing import FitRes as FitRes
from .typing import GetParametersIns as GetParametersIns
from .typing import GetParametersRes as GetParametersRes
from .typing import GetPropertiesIns as GetPropertiesIns
from .typing import GetPropertiesRes as GetPropertiesRes
from .typing import MetricRecordValues as MetricRecordValues
from .typing import Metrics as Metrics
from .typing import MetricsAggregationFn as MetricsAggregationFn
from .typing import NDArray as NDArray
from .typing import NDArrays as NDArrays
from .typing import Parameters as Parameters
from .typing import Properties as Properties
from .typing import ReconnectIns as ReconnectIns
from .typing import Scalar as Scalar
from .typing import ServerMessage as ServerMessage
from .typing import Status as Status

__all__ = [
    "Array",
    "ArrayRecord",
    "ClientMessage",
    "Code",
    "Config",
    "ConfigRecord",
    "ConfigRecordValues",
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
