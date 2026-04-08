"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.userConfigValueFromProto = exports.userConfigValueToProto = exports.userConfigFromProto = exports.messageToTaskRes = exports.messageFromTaskIns = exports.propertiesToProto = exports.propertiesFromProto = exports.getPropertiesResToProto = exports.getPropertiesInsFromProto = exports.statusToProto = exports.evaluateResToProto = exports.evaluateInsFromProto = exports.fitResToProto = exports.fitInsFromProto = exports.parameterResToProto = exports.metricsFromProto = exports.metricsToProto = exports.scalarFromProto = exports.scalarToProto = exports.parametersFromProto = exports.parametersToProto = void 0;
exports.recordSetToProto = recordSetToProto;
exports.recordSetFromProto = recordSetFromProto;
const task_1 = require("../protos/flwr/proto/task");
const node_1 = require("../protos/flwr/proto/node");
const typing_1 = require("./typing");
const recordset_1 = require("./recordset");
// Parameter conversions
const parametersToProto = (params) => {
    return { tensors: params.tensors, tensorType: params.tensorType };
};
exports.parametersToProto = parametersToProto;
const parametersFromProto = (protoParams) => {
    return { tensors: protoParams.tensors, tensorType: protoParams.tensorType };
};
exports.parametersFromProto = parametersFromProto;
// Scalar conversions
const scalarToProto = (scalar) => {
    if (typeof scalar === "string") {
        return { scalar: { oneofKind: "string", string: scalar } };
    }
    else if (typeof scalar === "boolean") {
        return { scalar: { oneofKind: "bool", bool: scalar } };
    }
    else if (typeof scalar === "bigint") {
        return { scalar: { oneofKind: "sint64", sint64: scalar } };
    }
    else if (typeof scalar === "number") {
        return { scalar: { oneofKind: "double", double: scalar } };
    }
    else if (scalar instanceof Uint8Array) {
        return { scalar: { oneofKind: "bytes", bytes: scalar } };
    }
    throw new Error("Unsupported scalar type");
};
exports.scalarToProto = scalarToProto;
const scalarFromProto = (protoScalar) => {
    switch (protoScalar.scalar?.oneofKind) {
        case "double":
            return protoScalar.scalar.double;
        case "sint64":
            return protoScalar.scalar.sint64;
        case "bool":
            return protoScalar.scalar.bool;
        case "string":
            return protoScalar.scalar.string;
        case "bytes":
            return protoScalar.scalar.bytes;
        default:
            throw new Error("Unknown scalar type");
    }
};
exports.scalarFromProto = scalarFromProto;
// Metrics conversions
const metricsToProto = (metrics) => {
    const protoMetrics = {};
    for (const key in metrics) {
        protoMetrics[key] = (0, exports.scalarToProto)(metrics[key]);
    }
    return protoMetrics;
};
exports.metricsToProto = metricsToProto;
const metricsFromProto = (protoMetrics) => {
    const metrics = {};
    for (const key in protoMetrics) {
        metrics[key] = (0, exports.scalarFromProto)(protoMetrics[key]);
    }
    return metrics;
};
exports.metricsFromProto = metricsFromProto;
// GetParametersRes conversions
const parameterResToProto = (res) => {
    return {
        parameters: (0, exports.parametersToProto)(res.parameters),
        status: (0, exports.statusToProto)(res.status),
    };
};
exports.parameterResToProto = parameterResToProto;
// FitIns conversions
const fitInsFromProto = (fitInsMsg) => {
    return {
        parameters: (0, exports.parametersFromProto)(fitInsMsg.parameters),
        config: (0, exports.metricsFromProto)(fitInsMsg.config),
    };
};
exports.fitInsFromProto = fitInsFromProto;
// FitRes conversions
const fitResToProto = (res) => {
    return {
        parameters: (0, exports.parametersToProto)(res.parameters),
        numExamples: BigInt(res.numExamples),
        metrics: Object.keys(res.metrics).length > 0 ? (0, exports.metricsToProto)(res.metrics) : {},
        status: (0, exports.statusToProto)(res.status),
    };
};
exports.fitResToProto = fitResToProto;
// EvaluateIns conversions
const evaluateInsFromProto = (evaluateInsMsg) => {
    return {
        parameters: (0, exports.parametersFromProto)(evaluateInsMsg.parameters),
        config: (0, exports.metricsFromProto)(evaluateInsMsg.config),
    };
};
exports.evaluateInsFromProto = evaluateInsFromProto;
// EvaluateRes conversions
const evaluateResToProto = (res) => {
    return {
        loss: res.loss,
        numExamples: BigInt(res.numExamples),
        metrics: Object.keys(res.metrics).length > 0 ? (0, exports.metricsToProto)(res.metrics) : {},
        status: (0, exports.statusToProto)(res.status),
    };
};
exports.evaluateResToProto = evaluateResToProto;
// Status conversions
const statusToProto = (status) => {
    return {
        code: status.code,
        message: status.message,
    };
};
exports.statusToProto = statusToProto;
// GetPropertiesIns conversions
const getPropertiesInsFromProto = (getPropertiesMsg) => {
    return {
        config: (0, exports.propertiesFromProto)(getPropertiesMsg.config),
    };
};
exports.getPropertiesInsFromProto = getPropertiesInsFromProto;
// GetPropertiesRes conversions
const getPropertiesResToProto = (res) => {
    return {
        properties: (0, exports.propertiesToProto)(res.properties),
        status: (0, exports.statusToProto)(res.status),
    };
};
exports.getPropertiesResToProto = getPropertiesResToProto;
// Properties conversions
const propertiesFromProto = (protoProperties) => {
    const properties = {};
    for (const key in protoProperties) {
        properties[key] = (0, exports.scalarFromProto)(protoProperties[key]);
    }
    return properties;
};
exports.propertiesFromProto = propertiesFromProto;
const propertiesToProto = (properties) => {
    const protoProperties = {};
    for (const key in properties) {
        protoProperties[key] = (0, exports.scalarToProto)(properties[key]);
    }
    return protoProperties;
};
exports.propertiesToProto = propertiesToProto;
function recordValueToProto(value) {
    if (typeof value === "number") {
        return { value: { oneofKind: "double", double: value } };
    }
    else if (typeof value === "bigint") {
        return { value: { oneofKind: "sint64", sint64: value } };
    }
    else if (typeof value === "boolean") {
        return { value: { oneofKind: "bool", bool: value } };
    }
    else if (typeof value === "string") {
        return { value: { oneofKind: "string", string: value } };
    }
    else if (value instanceof Uint8Array) {
        return { value: { oneofKind: "bytes", bytes: value } };
    }
    else if (Array.isArray(value)) {
        if (typeof value[0] === "number") {
            return { value: { oneofKind: "doubleList", doubleList: { vals: value } } };
        }
        else if (typeof value[0] === "bigint") {
            return { value: { oneofKind: "sintList", sintList: { vals: value } } };
        }
        else if (typeof value[0] === "boolean") {
            return { value: { oneofKind: "boolList", boolList: { vals: value } } };
        }
        else if (typeof value[0] === "string") {
            return { value: { oneofKind: "stringList", stringList: { vals: value } } };
        }
        else if (value[0] instanceof Uint8Array) {
            return { value: { oneofKind: "bytesList", bytesList: { vals: value } } };
        }
    }
    throw new TypeError("Unsupported value type");
}
// Helper for converting Protobuf messages back into values
function recordValueFromProto(proto) {
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
function arrayToProto(array) {
    return {
        dtype: array.dtype,
        shape: array.shape,
        stype: array.stype,
        data: array.data,
    };
}
function arrayFromProto(proto) {
    return new recordset_1.ArrayData(proto.dtype, proto.shape, proto.stype, proto.data);
}
function parametersRecordToProto(record) {
    return {
        dataKeys: Object.keys(record),
        dataValues: Object.values(record).map(arrayToProto),
    };
}
function parametersRecordFromProto(proto) {
    const arrayDict = Object.fromEntries(proto.dataKeys.map((k, i) => [k, arrayFromProto(proto.dataValues[i])]));
    // Create a new instance of ParametersRecord and populate it with the arrayDict
    return new recordset_1.ParametersRecord(arrayDict);
}
function metricsRecordToProto(record) {
    const data = Object.fromEntries(Object.entries(record).map(([k, v]) => [k, recordValueToProto(v)]));
    return { data };
}
function metricsRecordFromProto(proto) {
    const metrics = Object.fromEntries(Object.entries(proto.data).map(([k, v]) => [k, recordValueFromProto(v)]));
    return new recordset_1.MetricsRecord(metrics);
}
function configsRecordToProto(record) {
    const data = Object.fromEntries(Object.entries(record).map(([k, v]) => [k, recordValueToProto(v)]));
    return { data };
}
function configsRecordFromProto(proto) {
    const config = Object.fromEntries(Object.entries(proto.data).map(([k, v]) => [k, recordValueFromProto(v)]));
    return new recordset_1.ConfigsRecord(config);
}
function recordSetToProto(recordset) {
    const parameters = Object.fromEntries(Object.entries(recordset.parametersRecords).map(([k, v]) => [
        k,
        parametersRecordToProto(v), // Nested dictionary (string -> Record<string, ArrayData>)
    ]));
    const metrics = Object.fromEntries(Object.entries(recordset.metricsRecords).map(([k, v]) => [k, metricsRecordToProto(v)]));
    const configs = Object.fromEntries(Object.entries(recordset.configsRecords).map(([k, v]) => [k, configsRecordToProto(v)]));
    return { parameters, metrics, configs };
}
function recordSetFromProto(proto) {
    const parametersRecords = Object.fromEntries(Object.entries(proto.parameters).map(([k, v]) => [k, parametersRecordFromProto(v)]));
    const metricsRecords = Object.fromEntries(Object.entries(proto.metrics).map(([k, v]) => [k, metricsRecordFromProto(v)]));
    const configsRecords = Object.fromEntries(Object.entries(proto.configs).map(([k, v]) => [k, configsRecordFromProto(v)]));
    return new recordset_1.RecordSet(parametersRecords, metricsRecords, configsRecords);
}
const messageFromTaskIns = (taskIns) => {
    let metadata = {
        runId: taskIns.runId,
        messageId: taskIns.taskId,
        srcNodeId: taskIns.task?.producer?.nodeId,
        dstNodeId: taskIns.task?.consumer?.nodeId,
        replyToMessage: taskIns.task?.ancestry ? taskIns.task?.ancestry[0] : "",
        groupId: taskIns.groupId,
        ttl: taskIns.task?.ttl,
        messageType: taskIns.task?.taskType,
    };
    let message = new typing_1.Message(metadata, taskIns.task?.recordset ? recordSetFromProto(taskIns.task.recordset) : null, taskIns.task?.error ? { code: Number(taskIns.task.error.code), reason: taskIns.task.error.reason } : null);
    if (taskIns.task?.createdAt) {
        message.metadata.createdAt = taskIns.task?.createdAt;
    }
    return message;
};
exports.messageFromTaskIns = messageFromTaskIns;
const messageToTaskRes = (message) => {
    const md = message.metadata;
    const taskRes = task_1.TaskRes.create();
    taskRes.taskId = "",
        taskRes.groupId = md.groupId;
    taskRes.runId = md.runId;
    let task = task_1.Task.create();
    let producer = node_1.Node.create();
    producer.nodeId = md.srcNodeId;
    producer.anonymous = false;
    task.producer = producer;
    let consumer = node_1.Node.create();
    consumer.nodeId = BigInt(0);
    consumer.anonymous = true;
    task.consumer = consumer;
    task.createdAt = md.createdAt;
    task.ttl = md.ttl;
    task.ancestry = md.replyToMessage !== "" ? [md.replyToMessage] : [];
    task.taskType = md.messageType;
    task.recordset = message.content === null ? undefined : recordSetToProto(message.content);
    task.error = message.error === null ? undefined : { code: BigInt(message.error.code), reason: message.error.reason };
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
exports.messageToTaskRes = messageToTaskRes;
const userConfigFromProto = (proto) => {
    let metrics = {};
    Object.entries(proto).forEach(([key, value]) => {
        metrics[key] = (0, exports.userConfigValueFromProto)(value);
    });
    return metrics;
};
exports.userConfigFromProto = userConfigFromProto;
const userConfigValueToProto = (userConfigValue) => {
    switch (typeof userConfigValue) {
        case "string":
            return { scalar: { oneofKind: "string", string: userConfigValue } };
        case "number":
            return { scalar: { oneofKind: "double", double: userConfigValue } };
        case "bigint":
            return { scalar: { oneofKind: "sint64", sint64: userConfigValue } };
        case "boolean":
            return { scalar: { oneofKind: "bool", bool: userConfigValue } };
        default:
            throw new Error(`Accepted types: {bool, float, int, str} (but not ${typeof userConfigValue})`);
    }
};
exports.userConfigValueToProto = userConfigValueToProto;
const userConfigValueFromProto = (scalarMsg) => {
    switch (scalarMsg.scalar.oneofKind) {
        case "string":
            return scalarMsg.scalar.string;
        case "bool":
            return scalarMsg.scalar.bool;
        case "sint64":
            return scalarMsg.scalar.sint64;
        case "double":
            return scalarMsg.scalar.double;
        default:
            throw new Error(`Accepted types: {bool, float, int, str} (but not ${scalarMsg.scalar.oneofKind})`);
    }
};
exports.userConfigValueFromProto = userConfigValueFromProto;
