// Copyright 2024 Flower Labs GmbH. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
import {
  Parameters as ProtoParams,
  Scalar as ProtoScalar,
  Status as ProtoStatus,
  ClientMessage_GetPropertiesRes as ProtoClientMessage_GetPropertiesRes,
  ClientMessage_GetParametersRes as ProtoClientMessage_GetParametersRes,
  ClientMessage_FitRes as ProtoClientMessage_FitRes,
  ClientMessage_EvaluateRes as ProtoClientMessage_EvaluateRes,
  ServerMessage_GetPropertiesIns as ProtoServerMessage_GetPropertiesIns,
  ServerMessage_FitIns as ProtoServerMessage_FitIns,
  ServerMessage_EvaluateIns as ProtoServerMessage_EvaluateIns,
  Scalar,
} from "../protos/flwr/proto/transport";
import { Error as ProtoError } from "../protos/flwr/proto/error";
import { Task, TaskIns, TaskRes } from "../protos/flwr/proto/task";
import { Node } from "../protos/flwr/proto/node";
import {
  Parameters as LocalParams,
  Scalar as LocalScalar,
  Status as LocalStatus,
  GetParametersRes as LocalGetParametersRes,
  FitIns as LocalFitIns,
  FitRes as LocalFitRes,
  EvaluateIns as LocalEvaluateIns,
  EvaluateRes as LocalEvaluateRes,
  GetPropertiesIns as LocalGetPropertiesIns,
  GetPropertiesRes as LocalGetPropertiesRes,
  Properties as LocalProperties,
  Message,
  Metadata,
  Error as LocalError,
  UserConfigValue,
  UserConfig,
} from "./typing";
import {
  RecordSet,
  ConfigsRecord,
  MetricsRecord,
  MetricsRecordValue,
  ParametersRecord,
  ArrayData,
} from "./recordset";
import {
  RecordSet as ProtoRecordSet,
  ConfigsRecord as ProtoConfigsRecord,
  ConfigsRecordValue as ProtoConfigsRecordValue,
  MetricsRecord as ProtoMetricsRecord,
  MetricsRecordValue as ProtoMetricsRecordValue,
  ParametersRecord as ProtoParametersRecord,
  Array$ as ProtoArray,
} from "../protos/flwr/proto/recordset";

// Parameter conversions
export const parametersToProto = (params: LocalParams): ProtoParams => {
  return { tensors: params.tensors, tensorType: params.tensorType } as ProtoParams;
};

export const parametersFromProto = (protoParams: ProtoParams): LocalParams => {
  return { tensors: protoParams.tensors, tensorType: protoParams.tensorType } as LocalParams;
};

// Scalar conversions
export const scalarToProto = (scalar: LocalScalar): ProtoScalar => {
  if (typeof scalar === "string") {
    return { scalar: { oneofKind: "string", string: scalar } } as ProtoScalar;
  } else if (typeof scalar === "boolean") {
    return { scalar: { oneofKind: "bool", bool: scalar } } as ProtoScalar;
  } else if (typeof scalar === "bigint") {
    return { scalar: { oneofKind: "sint64", sint64: scalar } } as ProtoScalar;
  } else if (typeof scalar === "number") {
    return { scalar: { oneofKind: "double", double: scalar } } as ProtoScalar;
  } else if (scalar instanceof Uint8Array) {
    return { scalar: { oneofKind: "bytes", bytes: scalar } } as ProtoScalar;
  }
  throw new Error("Unsupported scalar type");
};

export const scalarFromProto = (protoScalar: ProtoScalar): LocalScalar => {
  switch (protoScalar.scalar?.oneofKind) {
    case "double":
      return protoScalar.scalar.double as number;
    case "sint64":
      return protoScalar.scalar.sint64 as bigint;
    case "bool":
      return protoScalar.scalar.bool as boolean;
    case "string":
      return protoScalar.scalar.string as string;
    case "bytes":
      return protoScalar.scalar.bytes as Uint8Array;
    default:
      throw new Error("Unknown scalar type");
  }
};

// Metrics conversions
export const metricsToProto = (
  metrics: Record<string, LocalScalar>,
): Record<string, ProtoScalar> => {
  const protoMetrics: Record<string, ProtoScalar> = {};
  for (const key in metrics) {
    protoMetrics[key] = scalarToProto(metrics[key]);
  }
  return protoMetrics;
};

export const metricsFromProto = (
  protoMetrics: Record<string, ProtoScalar>,
): Record<string, LocalScalar> => {
  const metrics: Record<string, LocalScalar> = {};
  for (const key in protoMetrics) {
    metrics[key] = scalarFromProto(protoMetrics[key]);
  }
  return metrics;
};

// GetParametersRes conversions
export const parameterResToProto = (
  res: LocalGetParametersRes,
): ProtoClientMessage_GetParametersRes => {
  return {
    parameters: parametersToProto(res.parameters),
    status: statusToProto(res.status),
  };
};

// FitIns conversions
export const fitInsFromProto = (fitInsMsg: ProtoServerMessage_FitIns): LocalFitIns => {
  return {
    parameters: parametersFromProto(fitInsMsg.parameters!),
    config: metricsFromProto(fitInsMsg.config),
  };
};

// FitRes conversions
export const fitResToProto = (res: LocalFitRes): ProtoClientMessage_FitRes => {
  return {
    parameters: parametersToProto(res.parameters),
    numExamples: BigInt(res.numExamples),
    metrics: Object.keys(res.metrics).length > 0 ? metricsToProto(res.metrics) : {},
    status: statusToProto(res.status),
  };
};

// EvaluateIns conversions
export const evaluateInsFromProto = (
  evaluateInsMsg: ProtoServerMessage_EvaluateIns,
): LocalEvaluateIns => {
  return {
    parameters: parametersFromProto(evaluateInsMsg.parameters!),
    config: metricsFromProto(evaluateInsMsg.config),
  };
};

// EvaluateRes conversions
export const evaluateResToProto = (res: LocalEvaluateRes): ProtoClientMessage_EvaluateRes => {
  return {
    loss: res.loss,
    numExamples: BigInt(res.numExamples),
    metrics: Object.keys(res.metrics).length > 0 ? metricsToProto(res.metrics) : {},
    status: statusToProto(res.status),
  };
};

// Status conversions
export const statusToProto = (status: LocalStatus): ProtoStatus => {
  return {
    code: status.code,
    message: status.message,
  };
};

// GetPropertiesIns conversions
export const getPropertiesInsFromProto = (
  getPropertiesMsg: ProtoServerMessage_GetPropertiesIns,
): LocalGetPropertiesIns => {
  return {
    config: propertiesFromProto(getPropertiesMsg.config),
  };
};

// GetPropertiesRes conversions
export const getPropertiesResToProto = (
  res: LocalGetPropertiesRes,
): ProtoClientMessage_GetPropertiesRes => {
  return {
    properties: propertiesToProto(res.properties),
    status: statusToProto(res.status),
  };
};

// Properties conversions
export const propertiesFromProto = (
  protoProperties: Record<string, ProtoScalar>,
): LocalProperties => {
  const properties: LocalProperties = {};
  for (const key in protoProperties) {
    properties[key] = scalarFromProto(protoProperties[key]);
  }
  return properties;
};

export const propertiesToProto = (properties: LocalProperties): Record<string, ProtoScalar> => {
  const protoProperties: Record<string, ProtoScalar> = {};
  for (const key in properties) {
    protoProperties[key] = scalarToProto(properties[key]);
  }
  return protoProperties;
};

function recordValueToProto(value: any): ProtoMetricsRecordValue | ProtoConfigsRecordValue {
  if (typeof value === "number") {
    return { value: { oneofKind: "double", double: value } };
  } else if (typeof value === "bigint") {
    return { value: { oneofKind: "sint64", sint64: value } };
  } else if (typeof value === "boolean") {
    return { value: { oneofKind: "bool", bool: value } };
  } else if (typeof value === "string") {
    return { value: { oneofKind: "string", string: value } };
  } else if (value instanceof Uint8Array) {
    return { value: { oneofKind: "bytes", bytes: value } };
  } else if (Array.isArray(value)) {
    if (typeof value[0] === "number") {
      return { value: { oneofKind: "doubleList", doubleList: { vals: value } } };
    } else if (typeof value[0] === "bigint") {
      return { value: { oneofKind: "sintList", sintList: { vals: value } } };
    } else if (typeof value[0] === "boolean") {
      return { value: { oneofKind: "boolList", boolList: { vals: value } } };
    } else if (typeof value[0] === "string") {
      return { value: { oneofKind: "stringList", stringList: { vals: value } } };
    } else if (value[0] instanceof Uint8Array) {
      return { value: { oneofKind: "bytesList", bytesList: { vals: value } } };
    }
  }
  throw new TypeError("Unsupported value type");
}

// Helper for converting Protobuf messages back into values
function recordValueFromProto(proto: ProtoMetricsRecordValue | ProtoConfigsRecordValue): any {
  switch (proto.value.oneofKind) {
    case "double":
      return proto.value.double;
    case "sint64":
      return proto.value.sint64;
    case "bool":
      return proto.value.bool;
    case "string":
      return proto.value.string;
    case "bytes":
      return proto.value.bytes;
    case "doubleList":
      return proto.value.doubleList.vals;
    case "sintList":
      return proto.value.sintList.vals;
    case "boolList":
      return proto.value.boolList.vals;
    case "stringList":
      return proto.value.stringList.vals;
    case "bytesList":
      return proto.value.bytesList.vals;
    default:
      throw new Error("Unknown value kind");
  }
}

function arrayToProto(array: ArrayData): ProtoArray {
  return {
    dtype: array.dtype,
    shape: array.shape,
    stype: array.stype,
    data: array.data,
  };
}

function arrayFromProto(proto: ProtoArray): ArrayData {
  return new ArrayData(proto.dtype, proto.shape, proto.stype, proto.data);
}

function parametersRecordToProto(record: ParametersRecord): ProtoParametersRecord {
  return {
    dataKeys: Object.keys(record),
    dataValues: Object.values(record).map(arrayToProto),
  };
}

function parametersRecordFromProto(proto: ProtoParametersRecord): ParametersRecord {
  const arrayDict = Object.fromEntries(
    proto.dataKeys.map((k, i) => [k, arrayFromProto(proto.dataValues[i])]),
  );

  // Create a new instance of ParametersRecord and populate it with the arrayDict
  return new ParametersRecord(arrayDict);
}

function metricsRecordToProto(record: MetricsRecord): ProtoMetricsRecord {
  const data = Object.fromEntries(
    Object.entries(record).map(([k, v]) => [k, recordValueToProto(v) as ProtoMetricsRecordValue]),
  );
  return { data };
}

function metricsRecordFromProto(proto: ProtoMetricsRecord): MetricsRecord {
  const metrics = Object.fromEntries(
    Object.entries(proto.data).map(([k, v]) => [k, recordValueFromProto(v) as MetricsRecordValue]),
  );
  return new MetricsRecord(metrics);
}

function configsRecordToProto(record: ConfigsRecord): ProtoConfigsRecord {
  const data = Object.fromEntries(
    Object.entries(record).map(([k, v]) => [k, recordValueToProto(v) as ProtoConfigsRecordValue]),
  );
  return { data };
}

function configsRecordFromProto(proto: ProtoConfigsRecord): ConfigsRecord {
  const config = Object.fromEntries(
    Object.entries(proto.data).map(([k, v]) => [k, recordValueFromProto(v)]),
  );
  return new ConfigsRecord(config);
}

export function recordSetToProto(recordset: RecordSet): ProtoRecordSet {
  const parameters = Object.fromEntries(
    Object.entries(recordset.parametersRecords).map(([k, v]) => [
      k,
      parametersRecordToProto(v), // Nested dictionary (string -> Record<string, ArrayData>)
    ]),
  );
  const metrics = Object.fromEntries(
    Object.entries(recordset.metricsRecords).map(([k, v]) => [k, metricsRecordToProto(v)]),
  );
  const configs = Object.fromEntries(
    Object.entries(recordset.configsRecords).map(([k, v]) => [k, configsRecordToProto(v)]),
  );
  return { parameters, metrics, configs };
}

export function recordSetFromProto(proto: ProtoRecordSet): RecordSet {
  const parametersRecords = Object.fromEntries(
    Object.entries(proto.parameters).map(([k, v]) => [k, parametersRecordFromProto(v)]),
  );
  const metricsRecords = Object.fromEntries(
    Object.entries(proto.metrics).map(([k, v]) => [k, metricsRecordFromProto(v)]),
  );
  const configsRecords = Object.fromEntries(
    Object.entries(proto.configs).map(([k, v]) => [k, configsRecordFromProto(v)]),
  );
  return new RecordSet(parametersRecords, metricsRecords, configsRecords);
}

export const messageFromTaskIns = (taskIns: TaskIns): Message => {
  let metadata = {
    runId: taskIns.runId,
    messageId: taskIns.taskId,
    srcNodeId: taskIns.task?.producer?.nodeId,
    dstNodeId: taskIns.task?.consumer?.nodeId,
    replyToMessage: taskIns.task?.ancestry ? taskIns.task?.ancestry[0] : "",
    groupId: taskIns.groupId,
    ttl: taskIns.task?.ttl,
    messageType: taskIns.task?.taskType,
  } as Metadata;

  let message = new Message(
    metadata,
    taskIns.task?.recordset ? recordSetFromProto(taskIns.task.recordset) : null,
    taskIns.task?.error ? ({ code: Number(taskIns.task.error.code), reason: taskIns.task.error.reason } as LocalError) : null,
  );

  if (taskIns.task?.createdAt) {
    message.metadata.createdAt = taskIns.task?.createdAt;
  }
  return message;
};

export const messageToTaskRes = (message: Message): TaskRes => {
  const md = message.metadata;
  const taskRes = TaskRes.create();
  taskRes.taskId = "",
    taskRes.groupId = md.groupId;
  taskRes.runId = md.runId;

  let task = Task.create();

  let producer = Node.create();
  producer.nodeId = md.srcNodeId;
  producer.anonymous = false;
  task.producer = producer;

  let consumer = Node.create();
  consumer.nodeId = BigInt(0);
  consumer.anonymous = true;
  task.consumer = consumer;

  task.createdAt = md.createdAt;
  task.ttl = md.ttl;
  task.ancestry = md.replyToMessage !== "" ? [md.replyToMessage] : [];
  task.taskType = md.messageType;
  task.recordset = message.content === null ? undefined : recordSetToProto(message.content);
  task.error = message.error === null ? undefined : ({ code: BigInt(message.error.code), reason: message.error.reason } as ProtoError);

  taskRes.task = task;
  return taskRes;


  // return {
  //   taskId: "",
  //   groupId: md.groupId,
  //   runId: md.runId,
  //   task: {
  //     producer: { nodeId: md.srcNodeId, anonymous: false } as Node,
  //     consumer: { nodeId: BigInt(0), anonymous: true } as Node,
  //     createdAt: md.createdAt,
  //     ttl: md.ttl,
  //     ancestry: md.replyToMessage ? [md.replyToMessage] : [],
  //     taskType: md.messageType,
  //     recordset: message.content ? recordSetToProto(message.content) : null,
  //     error: message.error ? ({ code: BigInt(message.error.code), reason: message.error.reason } as ProtoError) : null,
  //   } as Task,
  // } as TaskRes;
};

export const userConfigFromProto = (proto: Record<string, any>): UserConfig => {
  let metrics: UserConfig = {};

  Object.entries(proto).forEach(([key, value]: [string, Scalar]) => {
    metrics[key] = userConfigValueFromProto(value);
  });

  return metrics;
};

export const userConfigValueToProto = (userConfigValue: UserConfigValue): Scalar => {
  switch (typeof userConfigValue) {
    case "string":
      return { scalar: { oneofKind: "string", string: userConfigValue } } as Scalar;
    case "number":
      return { scalar: { oneofKind: "double", double: userConfigValue } } as Scalar;
    case "bigint":
      return { scalar: { oneofKind: "sint64", sint64: userConfigValue } } as Scalar;
    case "boolean":
      return { scalar: { oneofKind: "bool", bool: userConfigValue } } as Scalar;
    default:
      throw new Error(
        `Accepted types: {bool, float, int, str} (but not ${typeof userConfigValue})`,
      );
  }
};

export const userConfigValueFromProto = (scalarMsg: Scalar): UserConfigValue => {
  switch (scalarMsg.scalar.oneofKind) {
    case "string":
      return scalarMsg.scalar.string as UserConfigValue;
    case "bool":
      return scalarMsg.scalar.bool as UserConfigValue;
    case "sint64":
      return scalarMsg.scalar.sint64 as UserConfigValue;
    case "double":
      return scalarMsg.scalar.double as UserConfigValue;
    default:
      throw new Error(
        `Accepted types: {bool, float, int, str} (but not ${scalarMsg.scalar.oneofKind})`,
      );
  }
};
