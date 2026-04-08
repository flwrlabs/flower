"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TaskRes = exports.TaskIns = exports.Task = void 0;
const runtime_1 = require("@protobuf-ts/runtime");
const runtime_2 = require("@protobuf-ts/runtime");
const runtime_3 = require("@protobuf-ts/runtime");
const runtime_4 = require("@protobuf-ts/runtime");
const error_1 = require("./error");
const recordset_1 = require("./recordset");
const node_1 = require("./node");
// @generated message type with reflection information, may provide speed optimized methods
class Task$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.Task", [
            { no: 1, name: "producer", kind: "message", T: () => node_1.Node },
            { no: 2, name: "consumer", kind: "message", T: () => node_1.Node },
            { no: 3, name: "created_at", kind: "scalar", T: 1 /*ScalarType.DOUBLE*/ },
            { no: 4, name: "delivered_at", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 5, name: "pushed_at", kind: "scalar", T: 1 /*ScalarType.DOUBLE*/ },
            { no: 6, name: "ttl", kind: "scalar", T: 1 /*ScalarType.DOUBLE*/ },
            { no: 7, name: "ancestry", kind: "scalar", repeat: 2 /*RepeatType.UNPACKED*/, T: 9 /*ScalarType.STRING*/ },
            { no: 8, name: "task_type", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 9, name: "recordset", kind: "message", T: () => recordset_1.RecordSet },
            { no: 10, name: "error", kind: "message", T: () => error_1.Error }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.createdAt = 0;
        message.deliveredAt = "";
        message.pushedAt = 0;
        message.ttl = 0;
        message.ancestry = [];
        message.taskType = "";
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* flwr.proto.Node producer */ 1:
                    message.producer = node_1.Node.internalBinaryRead(reader, reader.uint32(), options, message.producer);
                    break;
                case /* flwr.proto.Node consumer */ 2:
                    message.consumer = node_1.Node.internalBinaryRead(reader, reader.uint32(), options, message.consumer);
                    break;
                case /* double created_at */ 3:
                    message.createdAt = reader.double();
                    break;
                case /* string delivered_at */ 4:
                    message.deliveredAt = reader.string();
                    break;
                case /* double pushed_at */ 5:
                    message.pushedAt = reader.double();
                    break;
                case /* double ttl */ 6:
                    message.ttl = reader.double();
                    break;
                case /* repeated string ancestry */ 7:
                    message.ancestry.push(reader.string());
                    break;
                case /* string task_type */ 8:
                    message.taskType = reader.string();
                    break;
                case /* flwr.proto.RecordSet recordset */ 9:
                    message.recordset = recordset_1.RecordSet.internalBinaryRead(reader, reader.uint32(), options, message.recordset);
                    break;
                case /* flwr.proto.Error error */ 10:
                    message.error = error_1.Error.internalBinaryRead(reader, reader.uint32(), options, message.error);
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_2.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* flwr.proto.Node producer = 1; */
        if (message.producer)
            node_1.Node.internalBinaryWrite(message.producer, writer.tag(1, runtime_1.WireType.LengthDelimited).fork(), options).join();
        /* flwr.proto.Node consumer = 2; */
        if (message.consumer)
            node_1.Node.internalBinaryWrite(message.consumer, writer.tag(2, runtime_1.WireType.LengthDelimited).fork(), options).join();
        /* double created_at = 3; */
        if (message.createdAt !== 0)
            writer.tag(3, runtime_1.WireType.Bit64).double(message.createdAt);
        /* string delivered_at = 4; */
        if (message.deliveredAt !== "")
            writer.tag(4, runtime_1.WireType.LengthDelimited).string(message.deliveredAt);
        /* double pushed_at = 5; */
        if (message.pushedAt !== 0)
            writer.tag(5, runtime_1.WireType.Bit64).double(message.pushedAt);
        /* double ttl = 6; */
        if (message.ttl !== 0)
            writer.tag(6, runtime_1.WireType.Bit64).double(message.ttl);
        /* repeated string ancestry = 7; */
        for (let i = 0; i < message.ancestry.length; i++)
            writer.tag(7, runtime_1.WireType.LengthDelimited).string(message.ancestry[i]);
        /* string task_type = 8; */
        if (message.taskType !== "")
            writer.tag(8, runtime_1.WireType.LengthDelimited).string(message.taskType);
        /* flwr.proto.RecordSet recordset = 9; */
        if (message.recordset)
            recordset_1.RecordSet.internalBinaryWrite(message.recordset, writer.tag(9, runtime_1.WireType.LengthDelimited).fork(), options).join();
        /* flwr.proto.Error error = 10; */
        if (message.error)
            error_1.Error.internalBinaryWrite(message.error, writer.tag(10, runtime_1.WireType.LengthDelimited).fork(), options).join();
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.Task
 */
exports.Task = new Task$Type();
// @generated message type with reflection information, may provide speed optimized methods
class TaskIns$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.TaskIns", [
            { no: 1, name: "task_id", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 2, name: "group_id", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 3, name: "run_id", kind: "scalar", T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 4, name: "task", kind: "message", T: () => exports.Task }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.taskId = "";
        message.groupId = "";
        message.runId = 0n;
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* string task_id */ 1:
                    message.taskId = reader.string();
                    break;
                case /* string group_id */ 2:
                    message.groupId = reader.string();
                    break;
                case /* uint64 run_id */ 3:
                    message.runId = reader.uint64().toBigInt();
                    break;
                case /* flwr.proto.Task task */ 4:
                    message.task = exports.Task.internalBinaryRead(reader, reader.uint32(), options, message.task);
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_2.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* string task_id = 1; */
        if (message.taskId !== "")
            writer.tag(1, runtime_1.WireType.LengthDelimited).string(message.taskId);
        /* string group_id = 2; */
        if (message.groupId !== "")
            writer.tag(2, runtime_1.WireType.LengthDelimited).string(message.groupId);
        /* uint64 run_id = 3; */
        if (message.runId !== 0n)
            writer.tag(3, runtime_1.WireType.Varint).uint64(message.runId);
        /* flwr.proto.Task task = 4; */
        if (message.task)
            exports.Task.internalBinaryWrite(message.task, writer.tag(4, runtime_1.WireType.LengthDelimited).fork(), options).join();
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.TaskIns
 */
exports.TaskIns = new TaskIns$Type();
// @generated message type with reflection information, may provide speed optimized methods
class TaskRes$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.TaskRes", [
            { no: 1, name: "task_id", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 2, name: "group_id", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 3, name: "run_id", kind: "scalar", T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 4, name: "task", kind: "message", T: () => exports.Task }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.taskId = "";
        message.groupId = "";
        message.runId = 0n;
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* string task_id */ 1:
                    message.taskId = reader.string();
                    break;
                case /* string group_id */ 2:
                    message.groupId = reader.string();
                    break;
                case /* uint64 run_id */ 3:
                    message.runId = reader.uint64().toBigInt();
                    break;
                case /* flwr.proto.Task task */ 4:
                    message.task = exports.Task.internalBinaryRead(reader, reader.uint32(), options, message.task);
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_2.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* string task_id = 1; */
        if (message.taskId !== "")
            writer.tag(1, runtime_1.WireType.LengthDelimited).string(message.taskId);
        /* string group_id = 2; */
        if (message.groupId !== "")
            writer.tag(2, runtime_1.WireType.LengthDelimited).string(message.groupId);
        /* uint64 run_id = 3; */
        if (message.runId !== 0n)
            writer.tag(3, runtime_1.WireType.Varint).uint64(message.runId);
        /* flwr.proto.Task task = 4; */
        if (message.task)
            exports.Task.internalBinaryWrite(message.task, writer.tag(4, runtime_1.WireType.LengthDelimited).fork(), options).join();
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.TaskRes
 */
exports.TaskRes = new TaskRes$Type();
