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
"""ProtoBuf serialization and deserialization."""


from typing import Any, cast

from flwr.app.user_config import UserConfig, UserConfigValue

# pylint: disable=E0611
from flwr.proto.fab_pb2 import Fab as ProtoFab
from flwr.proto.message_pb2 import Context as ProtoContext
from flwr.proto.message_pb2 import Message as ProtoMessage
from flwr.proto.recorddict_pb2 import Array as ProtoArray
from flwr.proto.recorddict_pb2 import ArrayRecord as ProtoArrayRecord
from flwr.proto.recorddict_pb2 import ConfigRecord as ProtoConfigRecord
from flwr.proto.recorddict_pb2 import ConfigRecordValue as ProtoConfigRecordValue
from flwr.proto.recorddict_pb2 import MetricRecord as ProtoMetricRecord
from flwr.proto.recorddict_pb2 import MetricRecordValue as ProtoMetricRecordValue
from flwr.proto.recorddict_pb2 import RecordDict as ProtoRecordDict
from flwr.proto.run_pb2 import Run as ProtoRun
from flwr.proto.run_pb2 import RunStatus as ProtoRunStatus
from flwr.proto.transport_pb2 import (
    ClientMessage,
    Code,
    Parameters,
    Reason,
    Scalar,
    ServerMessage,
    Status,
)

# pylint: enable=E0611
from . import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Context,
    MetricRecord,
    RecordDict,
    typing,
)
from .constant import INT64_MAX_VALUE
from .message import Message, make_message
from .serde_utils import (
    error_from_proto,
    error_to_proto,
    metadata_from_proto,
    metadata_to_proto,
    record_value_dict_from_proto,
    record_value_dict_to_proto,
)

#  === Parameters message ===


def parameters_to_proto(parameters: typing.Parameters) -> Parameters:
    """Serialize `Parameters` to ProtoBuf."""
    return Parameters(tensors=parameters.tensors, tensor_type=parameters.tensor_type)


def parameters_from_proto(msg: Parameters) -> typing.Parameters:
    """Deserialize `Parameters` from ProtoBuf."""
    tensors: list[bytes] = list(msg.tensors)
    return typing.Parameters(tensors=tensors, tensor_type=msg.tensor_type)


#  === ReconnectIns message ===


def reconnect_ins_to_proto(ins: typing.ReconnectIns) -> ServerMessage.ReconnectIns:
    """Serialize `ReconnectIns` to ProtoBuf."""
    if ins.seconds is not None:
        return ServerMessage.ReconnectIns(seconds=ins.seconds)
    return ServerMessage.ReconnectIns()


# === DisconnectRes message ===


def disconnect_res_from_proto(msg: ClientMessage.DisconnectRes) -> typing.DisconnectRes:
    """Deserialize `DisconnectRes` from ProtoBuf."""
    if msg.reason == Reason.RECONNECT:
        return typing.DisconnectRes(reason="RECONNECT")
    if msg.reason == Reason.POWER_DISCONNECTED:
        return typing.DisconnectRes(reason="POWER_DISCONNECTED")
    if msg.reason == Reason.WIFI_UNAVAILABLE:
        return typing.DisconnectRes(reason="WIFI_UNAVAILABLE")
    return typing.DisconnectRes(reason="UNKNOWN")


# === GetParameters messages ===


def get_parameters_ins_to_proto(
    ins: typing.GetParametersIns,
) -> ServerMessage.GetParametersIns:
    """Serialize `GetParametersIns` to ProtoBuf."""
    config = properties_to_proto(ins.config)
    return ServerMessage.GetParametersIns(config=config)


def get_parameters_ins_from_proto(
    msg: ServerMessage.GetParametersIns,
) -> typing.GetParametersIns:
    """Deserialize `GetParametersIns` from ProtoBuf."""
    config = properties_from_proto(msg.config)
    return typing.GetParametersIns(config=config)


def get_parameters_res_to_proto(
    res: typing.GetParametersRes,
) -> ClientMessage.GetParametersRes:
    """Serialize `GetParametersRes` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        return ClientMessage.GetParametersRes(status=status_msg)
    parameters_proto = parameters_to_proto(res.parameters)
    return ClientMessage.GetParametersRes(
        status=status_msg, parameters=parameters_proto
    )


def get_parameters_res_from_proto(
    msg: ClientMessage.GetParametersRes,
) -> typing.GetParametersRes:
    """Deserialize `GetParametersRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    parameters = parameters_from_proto(msg.parameters)
    return typing.GetParametersRes(status=status, parameters=parameters)


# === Fit messages ===


def fit_ins_to_proto(ins: typing.FitIns) -> ServerMessage.FitIns:
    """Serialize `FitIns` to ProtoBuf."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.FitIns(parameters=parameters_proto, config=config_msg)


def fit_ins_from_proto(msg: ServerMessage.FitIns) -> typing.FitIns:
    """Deserialize `FitIns` from ProtoBuf."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.FitIns(parameters=parameters, config=config)


def fit_res_to_proto(res: typing.FitRes) -> ClientMessage.FitRes:
    """Serialize `FitIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.FIT_NOT_IMPLEMENTED:
        return ClientMessage.FitRes(status=status_msg)
    parameters_proto = parameters_to_proto(res.parameters)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    return ClientMessage.FitRes(
        status=status_msg,
        parameters=parameters_proto,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def fit_res_from_proto(msg: ClientMessage.FitRes) -> typing.FitRes:
    """Deserialize `FitRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    parameters = parameters_from_proto(msg.parameters)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.FitRes(
        status=status,
        parameters=parameters,
        num_examples=msg.num_examples,
        metrics=metrics,
    )


# === GetProperties messages ===


def get_properties_ins_to_proto(
    ins: typing.GetPropertiesIns,
) -> ServerMessage.GetPropertiesIns:
    """Serialize `GetPropertiesIns` to ProtoBuf."""
    config = properties_to_proto(ins.config)
    return ServerMessage.GetPropertiesIns(config=config)


def get_properties_ins_from_proto(
    msg: ServerMessage.GetPropertiesIns,
) -> typing.GetPropertiesIns:
    """Deserialize `GetPropertiesIns` from ProtoBuf."""
    config = properties_from_proto(msg.config)
    return typing.GetPropertiesIns(config=config)


def get_properties_res_to_proto(
    res: typing.GetPropertiesRes,
) -> ClientMessage.GetPropertiesRes:
    """Serialize `GetPropertiesIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        return ClientMessage.GetPropertiesRes(status=status_msg)
    properties_msg = properties_to_proto(res.properties)
    return ClientMessage.GetPropertiesRes(status=status_msg, properties=properties_msg)


def get_properties_res_from_proto(
    msg: ClientMessage.GetPropertiesRes,
) -> typing.GetPropertiesRes:
    """Deserialize `GetPropertiesRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    properties = properties_from_proto(msg.properties)
    return typing.GetPropertiesRes(status=status, properties=properties)


# === Evaluate messages ===


def evaluate_ins_to_proto(ins: typing.EvaluateIns) -> ServerMessage.EvaluateIns:
    """Serialize `EvaluateIns` to ProtoBuf."""
    parameters_proto = parameters_to_proto(ins.parameters)
    config_msg = metrics_to_proto(ins.config)
    return ServerMessage.EvaluateIns(parameters=parameters_proto, config=config_msg)


def evaluate_ins_from_proto(msg: ServerMessage.EvaluateIns) -> typing.EvaluateIns:
    """Deserialize `EvaluateIns` from ProtoBuf."""
    parameters = parameters_from_proto(msg.parameters)
    config = metrics_from_proto(msg.config)
    return typing.EvaluateIns(parameters=parameters, config=config)


def evaluate_res_to_proto(res: typing.EvaluateRes) -> ClientMessage.EvaluateRes:
    """Serialize `EvaluateIns` to ProtoBuf."""
    status_msg = status_to_proto(res.status)
    if res.status.code == typing.Code.EVALUATE_NOT_IMPLEMENTED:
        return ClientMessage.EvaluateRes(status=status_msg)
    metrics_msg = None if res.metrics is None else metrics_to_proto(res.metrics)
    return ClientMessage.EvaluateRes(
        status=status_msg,
        loss=res.loss,
        num_examples=res.num_examples,
        metrics=metrics_msg,
    )


def evaluate_res_from_proto(msg: ClientMessage.EvaluateRes) -> typing.EvaluateRes:
    """Deserialize `EvaluateRes` from ProtoBuf."""
    status = status_from_proto(msg=msg.status)
    metrics = None if msg.metrics is None else metrics_from_proto(msg.metrics)
    return typing.EvaluateRes(
        status=status,
        loss=msg.loss,
        num_examples=msg.num_examples,
        metrics=metrics,
    )


# === Status messages ===


def status_to_proto(status: typing.Status) -> Status:
    """Serialize `Status` to ProtoBuf."""
    code = Code.OK
    if status.code == typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        code = Code.GET_PROPERTIES_NOT_IMPLEMENTED
    if status.code == typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        code = Code.GET_PARAMETERS_NOT_IMPLEMENTED
    if status.code == typing.Code.FIT_NOT_IMPLEMENTED:
        code = Code.FIT_NOT_IMPLEMENTED
    if status.code == typing.Code.EVALUATE_NOT_IMPLEMENTED:
        code = Code.EVALUATE_NOT_IMPLEMENTED
    return Status(code=code, message=status.message)


def status_from_proto(msg: Status) -> typing.Status:
    """Deserialize `Status` from ProtoBuf."""
    code = typing.Code.OK
    if msg.code == Code.GET_PROPERTIES_NOT_IMPLEMENTED:
        code = typing.Code.GET_PROPERTIES_NOT_IMPLEMENTED
    if msg.code == Code.GET_PARAMETERS_NOT_IMPLEMENTED:
        code = typing.Code.GET_PARAMETERS_NOT_IMPLEMENTED
    if msg.code == Code.FIT_NOT_IMPLEMENTED:
        code = typing.Code.FIT_NOT_IMPLEMENTED
    if msg.code == Code.EVALUATE_NOT_IMPLEMENTED:
        code = typing.Code.EVALUATE_NOT_IMPLEMENTED
    return typing.Status(code=code, message=msg.message)


# === Properties messages ===


def properties_to_proto(properties: typing.Properties) -> Any:
    """Serialize `Properties` to ProtoBuf."""
    proto = {}
    for key in properties:
        proto[key] = scalar_to_proto(properties[key])
    return proto


def properties_from_proto(proto: Any) -> typing.Properties:
    """Deserialize `Properties` from ProtoBuf."""
    properties = {}
    for k in proto:
        properties[k] = scalar_from_proto(proto[k])
    return properties


# === Metrics messages ===


def metrics_to_proto(metrics: typing.Metrics) -> Any:
    """Serialize `Metrics` to ProtoBuf."""
    proto = {}
    for key in metrics:
        proto[key] = scalar_to_proto(metrics[key])
    return proto


def metrics_from_proto(proto: Any) -> typing.Metrics:
    """Deserialize `Metrics` from ProtoBuf."""
    metrics = {}
    for k in proto:
        metrics[k] = scalar_from_proto(proto[k])
    return metrics


# === Scalar messages ===


def scalar_to_proto(scalar: typing.Scalar) -> Scalar:
    """Serialize `Scalar` to ProtoBuf."""
    if isinstance(scalar, bool):
        return Scalar(bool=scalar)

    if isinstance(scalar, bytes):
        return Scalar(bytes=scalar)

    if isinstance(scalar, float):
        return Scalar(double=scalar)

    if isinstance(scalar, int):
        # Use uint64 for integers larger than the maximum value of sint64
        if scalar > INT64_MAX_VALUE:
            return Scalar(uint64=scalar)
        return Scalar(sint64=scalar)

    if isinstance(scalar, str):
        return Scalar(string=scalar)

    raise ValueError(
        f"Accepted types: {bool, bytes, float, int, str} (but not {type(scalar)})"
    )


def scalar_from_proto(scalar_msg: Scalar) -> typing.Scalar:
    """Deserialize `Scalar` from ProtoBuf."""
    scalar_field = scalar_msg.WhichOneof("scalar")
    scalar = getattr(scalar_msg, cast(str, scalar_field))
    return cast(typing.Scalar, scalar)


# === Record messages ===


def array_to_proto(array: Array) -> ProtoArray:
    """Serialize Array to ProtoBuf."""
    return ProtoArray(
        dtype=array.dtype,
        shape=array.shape,
        stype=array.stype,
        data=array.data,
    )


def array_from_proto(array_proto: ProtoArray) -> Array:
    """Deserialize Array from ProtoBuf."""
    return Array(
        dtype=array_proto.dtype,
        shape=tuple(array_proto.shape),
        stype=array_proto.stype,
        data=array_proto.data,
    )


def array_record_to_proto(record: ArrayRecord) -> ProtoArrayRecord:
    """Serialize ArrayRecord to ProtoBuf."""
    return ProtoArrayRecord(
        items=[
            ProtoArrayRecord.Item(key=k, value=array_to_proto(v))
            for k, v in record.items()
        ]
    )


def array_record_from_proto(
    record_proto: ProtoArrayRecord,
) -> ArrayRecord:
    """Deserialize ArrayRecord from ProtoBuf."""
    return ArrayRecord(
        array_dict={
            item.key: array_from_proto(item.value) for item in record_proto.items
        },
        keep_input=False,
    )


def metric_record_to_proto(record: MetricRecord) -> ProtoMetricRecord:
    """Serialize MetricRecord to ProtoBuf."""
    protos = record_value_dict_to_proto(record, [float, int], ProtoMetricRecordValue)
    return ProtoMetricRecord(
        items=[ProtoMetricRecord.Item(key=k, value=v) for k, v in protos.items()]
    )


def metric_record_from_proto(record_proto: ProtoMetricRecord) -> MetricRecord:
    """Deserialize MetricRecord from ProtoBuf."""
    protos = {item.key: item.value for item in record_proto.items}
    return MetricRecord(
        metric_dict=cast(
            dict[str, typing.MetricRecordValues],
            record_value_dict_from_proto(protos),
        ),
        keep_input=False,
    )


def config_record_to_proto(record: ConfigRecord) -> ProtoConfigRecord:
    """Serialize ConfigRecord to ProtoBuf."""
    protos = record_value_dict_to_proto(
        record,
        [bool, int, float, str, bytes],
        ProtoConfigRecordValue,
    )
    return ProtoConfigRecord(
        items=[ProtoConfigRecord.Item(key=k, value=v) for k, v in protos.items()]
    )


def config_record_from_proto(record_proto: ProtoConfigRecord) -> ConfigRecord:
    """Deserialize ConfigRecord from ProtoBuf."""
    protos = {item.key: item.value for item in record_proto.items}
    return ConfigRecord(
        config_dict=cast(
            dict[str, typing.ConfigRecordValues],
            record_value_dict_from_proto(protos),
        ),
        keep_input=False,
    )


# === RecordDict message ===


def recorddict_to_proto(recorddict: RecordDict) -> ProtoRecordDict:
    """Serialize RecordDict to ProtoBuf."""
    item_cls = ProtoRecordDict.Item
    items: list[ProtoRecordDict.Item] = []
    for k, v in recorddict.items():
        if isinstance(v, ArrayRecord):
            items += [item_cls(key=k, array_record=array_record_to_proto(v))]
        elif isinstance(v, MetricRecord):
            items += [item_cls(key=k, metric_record=metric_record_to_proto(v))]
        elif isinstance(v, ConfigRecord):
            items += [item_cls(key=k, config_record=config_record_to_proto(v))]
        else:
            raise ValueError(f"Unsupported record type: {type(v)}")
    return ProtoRecordDict(items=items)


def recorddict_from_proto(recorddict_proto: ProtoRecordDict) -> RecordDict:
    """Deserialize RecordDict from ProtoBuf."""
    ret = RecordDict()
    for item in recorddict_proto.items:
        field = item.WhichOneof("value")
        if field == "array_record":
            ret[item.key] = array_record_from_proto(item.array_record)
        elif field == "metric_record":
            ret[item.key] = metric_record_from_proto(item.metric_record)
        elif field == "config_record":
            ret[item.key] = config_record_from_proto(item.config_record)
        else:
            raise ValueError(f"Unsupported record type: {field}")
    return ret


# === FAB ===


def fab_to_proto(fab: typing.Fab) -> ProtoFab:
    """Create a proto Fab object from a Python Fab."""
    return ProtoFab(
        hash_str=fab.hash_str, content=fab.content, verifications=fab.verifications
    )


def fab_from_proto(fab: ProtoFab) -> typing.Fab:
    """Create a Python Fab object from a proto Fab."""
    return typing.Fab(fab.hash_str, fab.content, dict(fab.verifications))


# === User configs ===


def user_config_to_proto(user_config: UserConfig) -> Any:
    """Serialize `UserConfig` to ProtoBuf."""
    proto = {}
    for key, value in user_config.items():
        proto[key] = user_config_value_to_proto(value)
    return proto


def user_config_from_proto(proto: Any) -> UserConfig:
    """Deserialize `UserConfig` from ProtoBuf."""
    metrics = {}
    for key, value in proto.items():
        metrics[key] = user_config_value_from_proto(value)
    return metrics


def user_config_value_to_proto(user_config_value: UserConfigValue) -> Scalar:
    """Serialize `UserConfigValue` to ProtoBuf."""
    if isinstance(user_config_value, bool):
        return Scalar(bool=user_config_value)

    if isinstance(user_config_value, float):
        return Scalar(double=user_config_value)

    if isinstance(user_config_value, int):
        return Scalar(sint64=user_config_value)

    if isinstance(user_config_value, str):
        return Scalar(string=user_config_value)

    raise ValueError(
        f"Accepted types: {bool, float, int, str} (but not {type(user_config_value)})"
    )


def user_config_value_from_proto(scalar_msg: Scalar) -> UserConfigValue:
    """Deserialize `UserConfigValue` from ProtoBuf."""
    scalar_field = scalar_msg.WhichOneof("scalar")
    scalar = getattr(scalar_msg, cast(str, scalar_field))
    return cast(UserConfigValue, scalar)


# === Message messages ===


def message_to_proto(message: Message) -> ProtoMessage:
    """Serialize `Message` to ProtoBuf."""
    proto = ProtoMessage(
        metadata=metadata_to_proto(message.metadata),
        content=(
            recorddict_to_proto(message.content) if message.has_content() else None
        ),
        error=error_to_proto(message.error) if message.has_error() else None,
    )
    return proto


def message_from_proto(message_proto: ProtoMessage) -> Message:
    """Deserialize `Message` from ProtoBuf."""
    return make_message(
        metadata=metadata_from_proto(message_proto.metadata),
        content=(
            recorddict_from_proto(message_proto.content)
            if message_proto.HasField("content")
            else None
        ),
        error=(
            error_from_proto(message_proto.error)
            if message_proto.HasField("error")
            else None
        ),
    )


# === Context messages ===


def context_to_proto(context: Context) -> ProtoContext:
    """Serialize `Context` to ProtoBuf."""
    proto = ProtoContext(
        run_id=context.run_id,
        node_id=context.node_id,
        node_config=user_config_to_proto(context.node_config),
        state=recorddict_to_proto(context.state),
        run_config=user_config_to_proto(context.run_config),
    )
    return proto


def context_from_proto(context_proto: ProtoContext) -> Context:
    """Deserialize `Context` from ProtoBuf."""
    context = Context(
        run_id=context_proto.run_id,
        node_id=context_proto.node_id,
        node_config=user_config_from_proto(context_proto.node_config),
        state=recorddict_from_proto(context_proto.state),
        run_config=user_config_from_proto(context_proto.run_config),
    )
    return context


# === Run messages ===


def run_to_proto(run: typing.Run) -> ProtoRun:
    """Serialize `Run` to ProtoBuf."""
    proto = ProtoRun(
        run_id=run.run_id,
        fab_id=run.fab_id,
        fab_version=run.fab_version,
        fab_hash=run.fab_hash,
        override_config=user_config_to_proto(run.override_config),
        pending_at=run.pending_at,
        starting_at=run.starting_at,
        running_at=run.running_at,
        finished_at=run.finished_at,
        status=run_status_to_proto(run.status),
        flwr_aid=run.flwr_aid,
        federation=run.federation,
        bytes_sent=run.bytes_sent,
        bytes_recv=run.bytes_recv,
        clientapp_runtime=run.clientapp_runtime,
        run_type=run.run_type,
    )
    if run.primary_task_id is not None:
        proto.primary_task_id = run.primary_task_id
    return proto


def run_from_proto(run_proto: ProtoRun) -> typing.Run:
    """Deserialize `Run` from ProtoBuf."""
    run = typing.Run(
        run_id=run_proto.run_id,
        fab_id=run_proto.fab_id,
        fab_version=run_proto.fab_version,
        fab_hash=run_proto.fab_hash,
        override_config=user_config_from_proto(run_proto.override_config),
        pending_at=run_proto.pending_at,
        starting_at=run_proto.starting_at,
        running_at=run_proto.running_at,
        finished_at=run_proto.finished_at,
        status=run_status_from_proto(run_proto.status),
        flwr_aid=run_proto.flwr_aid,
        federation=run_proto.federation,
        primary_task_id=(
            run_proto.primary_task_id if run_proto.HasField("primary_task_id") else None
        ),
        bytes_sent=run_proto.bytes_sent,
        bytes_recv=run_proto.bytes_recv,
        clientapp_runtime=run_proto.clientapp_runtime,
        run_type=run_proto.run_type,
    )
    return run


# === Run status ===


def run_status_to_proto(run_status: typing.RunStatus) -> ProtoRunStatus:
    """Serialize `RunStatus` to ProtoBuf."""
    return ProtoRunStatus(
        status=run_status.status,
        sub_status=run_status.sub_status,
        details=run_status.details,
    )


def run_status_from_proto(run_status_proto: ProtoRunStatus) -> typing.RunStatus:
    """Deserialize `RunStatus` from ProtoBuf."""
    return typing.RunStatus(
        status=run_status_proto.status,
        sub_status=run_status_proto.sub_status,
        details=run_status_proto.details,
    )
