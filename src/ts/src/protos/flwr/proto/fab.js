"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.GetFabResponse = exports.GetFabRequest = exports.Fab = void 0;
const runtime_1 = require("@protobuf-ts/runtime");
const runtime_2 = require("@protobuf-ts/runtime");
const runtime_3 = require("@protobuf-ts/runtime");
const runtime_4 = require("@protobuf-ts/runtime");
const node_1 = require("./node");
// @generated message type with reflection information, may provide speed optimized methods
class Fab$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.Fab", [
            { no: 1, name: "hash_str", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 2, name: "content", kind: "scalar", T: 12 /*ScalarType.BYTES*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.hashStr = "";
        message.content = new Uint8Array(0);
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* string hash_str */ 1:
                    message.hashStr = reader.string();
                    break;
                case /* bytes content */ 2:
                    message.content = reader.bytes();
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
        /* string hash_str = 1; */
        if (message.hashStr !== "")
            writer.tag(1, runtime_1.WireType.LengthDelimited).string(message.hashStr);
        /* bytes content = 2; */
        if (message.content.length)
            writer.tag(2, runtime_1.WireType.LengthDelimited).bytes(message.content);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.Fab
 */
exports.Fab = new Fab$Type();
// @generated message type with reflection information, may provide speed optimized methods
class GetFabRequest$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.GetFabRequest", [
            { no: 1, name: "node", kind: "message", T: () => node_1.Node },
            { no: 2, name: "hash_str", kind: "scalar", T: 9 /*ScalarType.STRING*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.hashStr = "";
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
                case /* string hash_str */ 2:
                    message.hashStr = reader.string();
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
        /* string hash_str = 2; */
        if (message.hashStr !== "")
            writer.tag(2, runtime_1.WireType.LengthDelimited).string(message.hashStr);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.GetFabRequest
 */
exports.GetFabRequest = new GetFabRequest$Type();
// @generated message type with reflection information, may provide speed optimized methods
class GetFabResponse$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.GetFabResponse", [
            { no: 1, name: "fab", kind: "message", T: () => exports.Fab }
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
                case /* flwr.proto.Fab fab */ 1:
                    message.fab = exports.Fab.internalBinaryRead(reader, reader.uint32(), options, message.fab);
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
        /* flwr.proto.Fab fab = 1; */
        if (message.fab)
            exports.Fab.internalBinaryWrite(message.fab, writer.tag(1, runtime_1.WireType.LengthDelimited).fork(), options).join();
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_2.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.GetFabResponse
 */
exports.GetFabResponse = new GetFabResponse$Type();
