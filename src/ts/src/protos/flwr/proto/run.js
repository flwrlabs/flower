"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.GetRunStatusResponse = exports.GetRunStatusRequest = exports.UpdateRunStatusResponse = exports.UpdateRunStatusRequest = exports.GetRunResponse = exports.GetRunRequest = exports.CreateRunResponse = exports.CreateRunRequest = exports.RunStatus = exports.Run = void 0;
const runtime_1 = require("@protobuf-ts/runtime");
const runtime_2 = require("@protobuf-ts/runtime");
const runtime_3 = require("@protobuf-ts/runtime");
const runtime_4 = require("@protobuf-ts/runtime");
const node_1 = require("./node");
const fab_1 = require("./fab");
const transport_1 = require("./transport");
// @generated message type with reflection information, may provide speed optimized methods
class Run$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.Run", [
            { no: 1, name: "run_id", kind: "scalar", T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 2, name: "fab_id", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 3, name: "fab_version", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 4, name: "override_config", kind: "map", K: 9 /*ScalarType.STRING*/, V: { kind: "message", T: () => transport_1.Scalar } },
            { no: 5, name: "fab_hash", kind: "scalar", T: 9 /*ScalarType.STRING*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.runId = 0n;
        message.fabId = "";
        message.fabVersion = "";
        message.overrideConfig = {};
        message.fabHash = "";
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* uint64 run_id */ 1:
                    message.runId = reader.uint64().toBigInt();
                    break;
                case /* string fab_id */ 2:
                    message.fabId = reader.string();
                    break;
                case /* string fab_version */ 3:
                    message.fabVersion = reader.string();
                    break;
                case /* map<string, flwr.proto.Scalar> override_config */ 4:
                    this.binaryReadMap4(message.overrideConfig, reader, options);
                    break;
                case /* string fab_hash */ 5:
                    message.fabHash = reader.string();
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
    binaryReadMap4(map, reader, options) {
        let len = reader.uint32(), end = reader.pos + len, key, val;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case 1:
                    key = reader.string();
                    break;
                case 2:
                    val = transport_1.Scalar.internalBinaryRead(reader, reader.uint32(), options);
                    break;
                default: throw new globalThis.Error("unknown map entry field for field flwr.proto.Run.override_config");
            }
        }
        map[key ?? ""] = val ?? transport_1.Scalar.create();
    }
    internalBinaryWrite(message, writer, options) {
        /* uint64 run_id = 1; */
        if (message.runId !== 0n)
            writer.tag(1, runtime_1.WireType.Varint).uint64(message.runId);
        /* string fab_id = 2; */
        if (message.fabId !== "")
            writer.tag(2, runtime_1.WireType.LengthDelimited).string(message.fabId);
        /* string fab_version = 3; */
        if (message.fabVersion !== "")
            writer.tag(3, runtime_1.WireType.LengthDelimited).string(message.fabVersion);
        /* map<string, flwr.proto.Scalar> override_config = 4; */
        for (let k of globalThis.Object.keys(message.overrideConfig)) {
            writer.tag(4, runtime_1.WireType.LengthDelimited).fork().tag(1, runtime_1.WireType.LengthDelimited).string(k);
            writer.tag(2, runtime_1.WireType.LengthDelimited).fork();
            transport_1.Scalar.internalBinaryWrite(message.overrideConfig[k], writer, options);
            writer.join().join();
        }
        /* string fab_hash = 5; */
        if (message.fabHash !== "")
            writer.tag(5, runtime_1.WireType.LengthDelimited).string(message.fabHash);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.Run
 */
exports.Run = new Run$Type();
// @generated message type with reflection information, may provide speed optimized methods
class RunStatus$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.RunStatus", [
            { no: 1, name: "status", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 2, name: "sub_status", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 3, name: "details", kind: "scalar", T: 9 /*ScalarType.STRING*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.status = "";
        message.subStatus = "";
        message.details = "";
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* string status */ 1:
                    message.status = reader.string();
                    break;
                case /* string sub_status */ 2:
                    message.subStatus = reader.string();
                    break;
                case /* string details */ 3:
                    message.details = reader.string();
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
        /* string status = 1; */
        if (message.status !== "")
            writer.tag(1, runtime_1.WireType.LengthDelimited).string(message.status);
        /* string sub_status = 2; */
        if (message.subStatus !== "")
            writer.tag(2, runtime_1.WireType.LengthDelimited).string(message.subStatus);
        /* string details = 3; */
        if (message.details !== "")
            writer.tag(3, runtime_1.WireType.LengthDelimited).string(message.details);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.RunStatus
 */
exports.RunStatus = new RunStatus$Type();
// @generated message type with reflection information, may provide speed optimized methods
class CreateRunRequest$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.CreateRunRequest", [
            { no: 1, name: "fab_id", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 2, name: "fab_version", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 3, name: "override_config", kind: "map", K: 9 /*ScalarType.STRING*/, V: { kind: "message", T: () => transport_1.Scalar } },
            { no: 4, name: "fab", kind: "message", T: () => fab_1.Fab }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.fabId = "";
        message.fabVersion = "";
        message.overrideConfig = {};
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* string fab_id */ 1:
                    message.fabId = reader.string();
                    break;
                case /* string fab_version */ 2:
                    message.fabVersion = reader.string();
                    break;
                case /* map<string, flwr.proto.Scalar> override_config */ 3:
                    this.binaryReadMap3(message.overrideConfig, reader, options);
                    break;
                case /* flwr.proto.Fab fab */ 4:
                    message.fab = fab_1.Fab.internalBinaryRead(reader, reader.uint32(), options, message.fab);
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
    binaryReadMap3(map, reader, options) {
        let len = reader.uint32(), end = reader.pos + len, key, val;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case 1:
                    key = reader.string();
                    break;
                case 2:
                    val = transport_1.Scalar.internalBinaryRead(reader, reader.uint32(), options);
                    break;
                default: throw new globalThis.Error("unknown map entry field for field flwr.proto.CreateRunRequest.override_config");
            }
        }
        map[key ?? ""] = val ?? transport_1.Scalar.create();
    }
    internalBinaryWrite(message, writer, options) {
        /* string fab_id = 1; */
        if (message.fabId !== "")
            writer.tag(1, runtime_1.WireType.LengthDelimited).string(message.fabId);
        /* string fab_version = 2; */
        if (message.fabVersion !== "")
            writer.tag(2, runtime_1.WireType.LengthDelimited).string(message.fabVersion);
        /* map<string, flwr.proto.Scalar> override_config = 3; */
        for (let k of globalThis.Object.keys(message.overrideConfig)) {
            writer.tag(3, runtime_1.WireType.LengthDelimited).fork().tag(1, runtime_1.WireType.LengthDelimited).string(k);
            writer.tag(2, runtime_1.WireType.LengthDelimited).fork();
            transport_1.Scalar.internalBinaryWrite(message.overrideConfig[k], writer, options);
            writer.join().join();
        }
        /* flwr.proto.Fab fab = 4; */
        if (message.fab)
            fab_1.Fab.internalBinaryWrite(message.fab, writer.tag(4, runtime_1.WireType.LengthDelimited).fork(), options).join();
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.CreateRunRequest
 */
exports.CreateRunRequest = new CreateRunRequest$Type();
// @generated message type with reflection information, may provide speed optimized methods
class CreateRunResponse$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.CreateRunResponse", [
            { no: 1, name: "run_id", kind: "scalar", T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
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
                case /* uint64 run_id */ 1:
                    message.runId = reader.uint64().toBigInt();
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
        /* uint64 run_id = 1; */
        if (message.runId !== 0n)
            writer.tag(1, runtime_1.WireType.Varint).uint64(message.runId);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.CreateRunResponse
 */
exports.CreateRunResponse = new CreateRunResponse$Type();
// @generated message type with reflection information, may provide speed optimized methods
class GetRunRequest$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.GetRunRequest", [
            { no: 1, name: "node", kind: "message", T: () => node_1.Node },
            { no: 2, name: "run_id", kind: "scalar", T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
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
                case /* flwr.proto.Node node */ 1:
                    message.node = node_1.Node.internalBinaryRead(reader, reader.uint32(), options, message.node);
                    break;
                case /* uint64 run_id */ 2:
                    message.runId = reader.uint64().toBigInt();
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
        /* flwr.proto.Node node = 1; */
        if (message.node)
            node_1.Node.internalBinaryWrite(message.node, writer.tag(1, runtime_1.WireType.LengthDelimited).fork(), options).join();
        /* uint64 run_id = 2; */
        if (message.runId !== 0n)
            writer.tag(2, runtime_1.WireType.Varint).uint64(message.runId);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.GetRunRequest
 */
exports.GetRunRequest = new GetRunRequest$Type();
// @generated message type with reflection information, may provide speed optimized methods
class GetRunResponse$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.GetRunResponse", [
            { no: 1, name: "run", kind: "message", T: () => exports.Run }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* flwr.proto.Run run */ 1:
                    message.run = exports.Run.internalBinaryRead(reader, reader.uint32(), options, message.run);
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
        /* flwr.proto.Run run = 1; */
        if (message.run)
            exports.Run.internalBinaryWrite(message.run, writer.tag(1, runtime_1.WireType.LengthDelimited).fork(), options).join();
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.GetRunResponse
 */
exports.GetRunResponse = new GetRunResponse$Type();
// @generated message type with reflection information, may provide speed optimized methods
class UpdateRunStatusRequest$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.UpdateRunStatusRequest", [
            { no: 1, name: "run_id", kind: "scalar", T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 2, name: "run_status", kind: "message", T: () => exports.RunStatus }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
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
                case /* uint64 run_id */ 1:
                    message.runId = reader.uint64().toBigInt();
                    break;
                case /* flwr.proto.RunStatus run_status */ 2:
                    message.runStatus = exports.RunStatus.internalBinaryRead(reader, reader.uint32(), options, message.runStatus);
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
        /* uint64 run_id = 1; */
        if (message.runId !== 0n)
            writer.tag(1, runtime_1.WireType.Varint).uint64(message.runId);
        /* flwr.proto.RunStatus run_status = 2; */
        if (message.runStatus)
            exports.RunStatus.internalBinaryWrite(message.runStatus, writer.tag(2, runtime_1.WireType.LengthDelimited).fork(), options).join();
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.UpdateRunStatusRequest
 */
exports.UpdateRunStatusRequest = new UpdateRunStatusRequest$Type();
// @generated message type with reflection information, may provide speed optimized methods
class UpdateRunStatusResponse$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.UpdateRunStatusResponse", []);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        return target ?? this.create();
    }
    internalBinaryWrite(message, writer, options) {
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.UpdateRunStatusResponse
 */
exports.UpdateRunStatusResponse = new UpdateRunStatusResponse$Type();
// @generated message type with reflection information, may provide speed optimized methods
class GetRunStatusRequest$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.GetRunStatusRequest", [
            { no: 1, name: "node", kind: "message", T: () => node_1.Node },
            { no: 2, name: "run_ids", kind: "scalar", repeat: 1 /*RepeatType.PACKED*/, T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.runIds = [];
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* flwr.proto.Node node */ 1:
                    message.node = node_1.Node.internalBinaryRead(reader, reader.uint32(), options, message.node);
                    break;
                case /* repeated uint64 run_ids */ 2:
                    if (wireType === runtime_1.WireType.LengthDelimited)
                        for (let e = reader.int32() + reader.pos; reader.pos < e;)
                            message.runIds.push(reader.uint64().toBigInt());
                    else
                        message.runIds.push(reader.uint64().toBigInt());
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
        /* flwr.proto.Node node = 1; */
        if (message.node)
            node_1.Node.internalBinaryWrite(message.node, writer.tag(1, runtime_1.WireType.LengthDelimited).fork(), options).join();
        /* repeated uint64 run_ids = 2; */
        if (message.runIds.length) {
            writer.tag(2, runtime_1.WireType.LengthDelimited).fork();
            for (let i = 0; i < message.runIds.length; i++)
                writer.uint64(message.runIds[i]);
            writer.join();
        }
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.GetRunStatusRequest
 */
exports.GetRunStatusRequest = new GetRunStatusRequest$Type();
// @generated message type with reflection information, may provide speed optimized methods
class GetRunStatusResponse$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.GetRunStatusResponse", [
            { no: 1, name: "run_status_dict", kind: "map", K: 4 /*ScalarType.UINT64*/, V: { kind: "message", T: () => exports.RunStatus } }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.runStatusDict = {};
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* map<uint64, flwr.proto.RunStatus> run_status_dict */ 1:
                    this.binaryReadMap1(message.runStatusDict, reader, options);
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
    binaryReadMap1(map, reader, options) {
        let len = reader.uint32(), end = reader.pos + len, key, val;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case 1:
                    key = reader.uint64().toString();
                    break;
                case 2:
                    val = exports.RunStatus.internalBinaryRead(reader, reader.uint32(), options);
                    break;
                default: throw new globalThis.Error("unknown map entry field for field flwr.proto.GetRunStatusResponse.run_status_dict");
            }
        }
        map[key ?? "0"] = val ?? exports.RunStatus.create();
    }
    internalBinaryWrite(message, writer, options) {
        /* map<uint64, flwr.proto.RunStatus> run_status_dict = 1; */
        for (let k of globalThis.Object.keys(message.runStatusDict)) {
            writer.tag(1, runtime_1.WireType.LengthDelimited).fork().tag(1, runtime_1.WireType.Varint).uint64(k);
            writer.tag(2, runtime_1.WireType.LengthDelimited).fork();
            exports.RunStatus.internalBinaryWrite(message.runStatusDict[k], writer, options);
            writer.join().join();
        }
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.GetRunStatusResponse
 */
exports.GetRunStatusResponse = new GetRunStatusResponse$Type();
