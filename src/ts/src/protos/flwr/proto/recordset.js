"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RecordSet = exports.ConfigsRecord = exports.MetricsRecord = exports.ParametersRecord = exports.ConfigsRecordValue = exports.MetricsRecordValue = exports.Array$ = exports.BytesList = exports.StringList = exports.BoolList = exports.UintList = exports.SintList = exports.DoubleList = void 0;
const runtime_1 = require("@protobuf-ts/runtime");
const runtime_2 = require("@protobuf-ts/runtime");
const runtime_3 = require("@protobuf-ts/runtime");
const runtime_4 = require("@protobuf-ts/runtime");
// @generated message type with reflection information, may provide speed optimized methods
class DoubleList$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.DoubleList", [
            { no: 1, name: "vals", kind: "scalar", repeat: 1 /*RepeatType.PACKED*/, T: 1 /*ScalarType.DOUBLE*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.vals = [];
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* repeated double vals */ 1:
                    if (wireType === runtime_2.WireType.LengthDelimited)
                        for (let e = reader.int32() + reader.pos; reader.pos < e;)
                            message.vals.push(reader.double());
                    else
                        message.vals.push(reader.double());
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* repeated double vals = 1; */
        if (message.vals.length) {
            writer.tag(1, runtime_2.WireType.LengthDelimited).fork();
            for (let i = 0; i < message.vals.length; i++)
                writer.double(message.vals[i]);
            writer.join();
        }
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.DoubleList
 */
exports.DoubleList = new DoubleList$Type();
// @generated message type with reflection information, may provide speed optimized methods
class SintList$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.SintList", [
            { no: 1, name: "vals", kind: "scalar", repeat: 1 /*RepeatType.PACKED*/, T: 18 /*ScalarType.SINT64*/, L: 0 /*LongType.BIGINT*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.vals = [];
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* repeated sint64 vals */ 1:
                    if (wireType === runtime_2.WireType.LengthDelimited)
                        for (let e = reader.int32() + reader.pos; reader.pos < e;)
                            message.vals.push(reader.sint64().toBigInt());
                    else
                        message.vals.push(reader.sint64().toBigInt());
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* repeated sint64 vals = 1; */
        if (message.vals.length) {
            writer.tag(1, runtime_2.WireType.LengthDelimited).fork();
            for (let i = 0; i < message.vals.length; i++)
                writer.sint64(message.vals[i]);
            writer.join();
        }
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.SintList
 */
exports.SintList = new SintList$Type();
// @generated message type with reflection information, may provide speed optimized methods
class UintList$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.UintList", [
            { no: 1, name: "vals", kind: "scalar", repeat: 1 /*RepeatType.PACKED*/, T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.vals = [];
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* repeated uint64 vals */ 1:
                    if (wireType === runtime_2.WireType.LengthDelimited)
                        for (let e = reader.int32() + reader.pos; reader.pos < e;)
                            message.vals.push(reader.uint64().toBigInt());
                    else
                        message.vals.push(reader.uint64().toBigInt());
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* repeated uint64 vals = 1; */
        if (message.vals.length) {
            writer.tag(1, runtime_2.WireType.LengthDelimited).fork();
            for (let i = 0; i < message.vals.length; i++)
                writer.uint64(message.vals[i]);
            writer.join();
        }
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.UintList
 */
exports.UintList = new UintList$Type();
// @generated message type with reflection information, may provide speed optimized methods
class BoolList$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.BoolList", [
            { no: 1, name: "vals", kind: "scalar", repeat: 1 /*RepeatType.PACKED*/, T: 8 /*ScalarType.BOOL*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.vals = [];
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* repeated bool vals */ 1:
                    if (wireType === runtime_2.WireType.LengthDelimited)
                        for (let e = reader.int32() + reader.pos; reader.pos < e;)
                            message.vals.push(reader.bool());
                    else
                        message.vals.push(reader.bool());
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* repeated bool vals = 1; */
        if (message.vals.length) {
            writer.tag(1, runtime_2.WireType.LengthDelimited).fork();
            for (let i = 0; i < message.vals.length; i++)
                writer.bool(message.vals[i]);
            writer.join();
        }
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.BoolList
 */
exports.BoolList = new BoolList$Type();
// @generated message type with reflection information, may provide speed optimized methods
class StringList$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.StringList", [
            { no: 1, name: "vals", kind: "scalar", repeat: 2 /*RepeatType.UNPACKED*/, T: 9 /*ScalarType.STRING*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.vals = [];
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* repeated string vals */ 1:
                    message.vals.push(reader.string());
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* repeated string vals = 1; */
        for (let i = 0; i < message.vals.length; i++)
            writer.tag(1, runtime_2.WireType.LengthDelimited).string(message.vals[i]);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.StringList
 */
exports.StringList = new StringList$Type();
// @generated message type with reflection information, may provide speed optimized methods
class BytesList$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.BytesList", [
            { no: 1, name: "vals", kind: "scalar", repeat: 2 /*RepeatType.UNPACKED*/, T: 12 /*ScalarType.BYTES*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.vals = [];
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* repeated bytes vals */ 1:
                    message.vals.push(reader.bytes());
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* repeated bytes vals = 1; */
        for (let i = 0; i < message.vals.length; i++)
            writer.tag(1, runtime_2.WireType.LengthDelimited).bytes(message.vals[i]);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.BytesList
 */
exports.BytesList = new BytesList$Type();
// @generated message type with reflection information, may provide speed optimized methods
class Array$$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.Array", [
            { no: 1, name: "dtype", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 2, name: "shape", kind: "scalar", repeat: 1 /*RepeatType.PACKED*/, T: 5 /*ScalarType.INT32*/ },
            { no: 3, name: "stype", kind: "scalar", T: 9 /*ScalarType.STRING*/ },
            { no: 4, name: "data", kind: "scalar", T: 12 /*ScalarType.BYTES*/ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.dtype = "";
        message.shape = [];
        message.stype = "";
        message.data = new Uint8Array(0);
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* string dtype */ 1:
                    message.dtype = reader.string();
                    break;
                case /* repeated int32 shape */ 2:
                    if (wireType === runtime_2.WireType.LengthDelimited)
                        for (let e = reader.int32() + reader.pos; reader.pos < e;)
                            message.shape.push(reader.int32());
                    else
                        message.shape.push(reader.int32());
                    break;
                case /* string stype */ 3:
                    message.stype = reader.string();
                    break;
                case /* bytes data */ 4:
                    message.data = reader.bytes();
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* string dtype = 1; */
        if (message.dtype !== "")
            writer.tag(1, runtime_2.WireType.LengthDelimited).string(message.dtype);
        /* repeated int32 shape = 2; */
        if (message.shape.length) {
            writer.tag(2, runtime_2.WireType.LengthDelimited).fork();
            for (let i = 0; i < message.shape.length; i++)
                writer.int32(message.shape[i]);
            writer.join();
        }
        /* string stype = 3; */
        if (message.stype !== "")
            writer.tag(3, runtime_2.WireType.LengthDelimited).string(message.stype);
        /* bytes data = 4; */
        if (message.data.length)
            writer.tag(4, runtime_2.WireType.LengthDelimited).bytes(message.data);
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.Array
 */
exports.Array$ = new Array$$Type();
// @generated message type with reflection information, may provide speed optimized methods
class MetricsRecordValue$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.MetricsRecordValue", [
            { no: 1, name: "double", kind: "scalar", oneof: "value", T: 1 /*ScalarType.DOUBLE*/ },
            { no: 2, name: "sint64", kind: "scalar", oneof: "value", T: 18 /*ScalarType.SINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 3, name: "uint64", kind: "scalar", oneof: "value", T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 21, name: "double_list", kind: "message", oneof: "value", T: () => exports.DoubleList },
            { no: 22, name: "sint_list", kind: "message", oneof: "value", T: () => exports.SintList },
            { no: 23, name: "uint_list", kind: "message", oneof: "value", T: () => exports.UintList }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.value = { oneofKind: undefined };
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* double double */ 1:
                    message.value = {
                        oneofKind: "double",
                        double: reader.double()
                    };
                    break;
                case /* sint64 sint64 */ 2:
                    message.value = {
                        oneofKind: "sint64",
                        sint64: reader.sint64().toBigInt()
                    };
                    break;
                case /* uint64 uint64 */ 3:
                    message.value = {
                        oneofKind: "uint64",
                        uint64: reader.uint64().toBigInt()
                    };
                    break;
                case /* flwr.proto.DoubleList double_list */ 21:
                    message.value = {
                        oneofKind: "doubleList",
                        doubleList: exports.DoubleList.internalBinaryRead(reader, reader.uint32(), options, message.value.doubleList)
                    };
                    break;
                case /* flwr.proto.SintList sint_list */ 22:
                    message.value = {
                        oneofKind: "sintList",
                        sintList: exports.SintList.internalBinaryRead(reader, reader.uint32(), options, message.value.sintList)
                    };
                    break;
                case /* flwr.proto.UintList uint_list */ 23:
                    message.value = {
                        oneofKind: "uintList",
                        uintList: exports.UintList.internalBinaryRead(reader, reader.uint32(), options, message.value.uintList)
                    };
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* double double = 1; */
        if (message.value.oneofKind === "double")
            writer.tag(1, runtime_2.WireType.Bit64).double(message.value.double);
        /* sint64 sint64 = 2; */
        if (message.value.oneofKind === "sint64")
            writer.tag(2, runtime_2.WireType.Varint).sint64(message.value.sint64);
        /* uint64 uint64 = 3; */
        if (message.value.oneofKind === "uint64")
            writer.tag(3, runtime_2.WireType.Varint).uint64(message.value.uint64);
        /* flwr.proto.DoubleList double_list = 21; */
        if (message.value.oneofKind === "doubleList")
            exports.DoubleList.internalBinaryWrite(message.value.doubleList, writer.tag(21, runtime_2.WireType.LengthDelimited).fork(), options).join();
        /* flwr.proto.SintList sint_list = 22; */
        if (message.value.oneofKind === "sintList")
            exports.SintList.internalBinaryWrite(message.value.sintList, writer.tag(22, runtime_2.WireType.LengthDelimited).fork(), options).join();
        /* flwr.proto.UintList uint_list = 23; */
        if (message.value.oneofKind === "uintList")
            exports.UintList.internalBinaryWrite(message.value.uintList, writer.tag(23, runtime_2.WireType.LengthDelimited).fork(), options).join();
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.MetricsRecordValue
 */
exports.MetricsRecordValue = new MetricsRecordValue$Type();
// @generated message type with reflection information, may provide speed optimized methods
class ConfigsRecordValue$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.ConfigsRecordValue", [
            { no: 1, name: "double", kind: "scalar", oneof: "value", T: 1 /*ScalarType.DOUBLE*/ },
            { no: 2, name: "sint64", kind: "scalar", oneof: "value", T: 18 /*ScalarType.SINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 3, name: "uint64", kind: "scalar", oneof: "value", T: 4 /*ScalarType.UINT64*/, L: 0 /*LongType.BIGINT*/ },
            { no: 4, name: "bool", kind: "scalar", oneof: "value", T: 8 /*ScalarType.BOOL*/ },
            { no: 5, name: "string", kind: "scalar", oneof: "value", T: 9 /*ScalarType.STRING*/ },
            { no: 6, name: "bytes", kind: "scalar", oneof: "value", T: 12 /*ScalarType.BYTES*/ },
            { no: 21, name: "double_list", kind: "message", oneof: "value", T: () => exports.DoubleList },
            { no: 22, name: "sint_list", kind: "message", oneof: "value", T: () => exports.SintList },
            { no: 23, name: "uint_list", kind: "message", oneof: "value", T: () => exports.UintList },
            { no: 24, name: "bool_list", kind: "message", oneof: "value", T: () => exports.BoolList },
            { no: 25, name: "string_list", kind: "message", oneof: "value", T: () => exports.StringList },
            { no: 26, name: "bytes_list", kind: "message", oneof: "value", T: () => exports.BytesList }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.value = { oneofKind: undefined };
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* double double */ 1:
                    message.value = {
                        oneofKind: "double",
                        double: reader.double()
                    };
                    break;
                case /* sint64 sint64 */ 2:
                    message.value = {
                        oneofKind: "sint64",
                        sint64: reader.sint64().toBigInt()
                    };
                    break;
                case /* uint64 uint64 */ 3:
                    message.value = {
                        oneofKind: "uint64",
                        uint64: reader.uint64().toBigInt()
                    };
                    break;
                case /* bool bool */ 4:
                    message.value = {
                        oneofKind: "bool",
                        bool: reader.bool()
                    };
                    break;
                case /* string string */ 5:
                    message.value = {
                        oneofKind: "string",
                        string: reader.string()
                    };
                    break;
                case /* bytes bytes */ 6:
                    message.value = {
                        oneofKind: "bytes",
                        bytes: reader.bytes()
                    };
                    break;
                case /* flwr.proto.DoubleList double_list */ 21:
                    message.value = {
                        oneofKind: "doubleList",
                        doubleList: exports.DoubleList.internalBinaryRead(reader, reader.uint32(), options, message.value.doubleList)
                    };
                    break;
                case /* flwr.proto.SintList sint_list */ 22:
                    message.value = {
                        oneofKind: "sintList",
                        sintList: exports.SintList.internalBinaryRead(reader, reader.uint32(), options, message.value.sintList)
                    };
                    break;
                case /* flwr.proto.UintList uint_list */ 23:
                    message.value = {
                        oneofKind: "uintList",
                        uintList: exports.UintList.internalBinaryRead(reader, reader.uint32(), options, message.value.uintList)
                    };
                    break;
                case /* flwr.proto.BoolList bool_list */ 24:
                    message.value = {
                        oneofKind: "boolList",
                        boolList: exports.BoolList.internalBinaryRead(reader, reader.uint32(), options, message.value.boolList)
                    };
                    break;
                case /* flwr.proto.StringList string_list */ 25:
                    message.value = {
                        oneofKind: "stringList",
                        stringList: exports.StringList.internalBinaryRead(reader, reader.uint32(), options, message.value.stringList)
                    };
                    break;
                case /* flwr.proto.BytesList bytes_list */ 26:
                    message.value = {
                        oneofKind: "bytesList",
                        bytesList: exports.BytesList.internalBinaryRead(reader, reader.uint32(), options, message.value.bytesList)
                    };
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* double double = 1; */
        if (message.value.oneofKind === "double")
            writer.tag(1, runtime_2.WireType.Bit64).double(message.value.double);
        /* sint64 sint64 = 2; */
        if (message.value.oneofKind === "sint64")
            writer.tag(2, runtime_2.WireType.Varint).sint64(message.value.sint64);
        /* uint64 uint64 = 3; */
        if (message.value.oneofKind === "uint64")
            writer.tag(3, runtime_2.WireType.Varint).uint64(message.value.uint64);
        /* bool bool = 4; */
        if (message.value.oneofKind === "bool")
            writer.tag(4, runtime_2.WireType.Varint).bool(message.value.bool);
        /* string string = 5; */
        if (message.value.oneofKind === "string")
            writer.tag(5, runtime_2.WireType.LengthDelimited).string(message.value.string);
        /* bytes bytes = 6; */
        if (message.value.oneofKind === "bytes")
            writer.tag(6, runtime_2.WireType.LengthDelimited).bytes(message.value.bytes);
        /* flwr.proto.DoubleList double_list = 21; */
        if (message.value.oneofKind === "doubleList")
            exports.DoubleList.internalBinaryWrite(message.value.doubleList, writer.tag(21, runtime_2.WireType.LengthDelimited).fork(), options).join();
        /* flwr.proto.SintList sint_list = 22; */
        if (message.value.oneofKind === "sintList")
            exports.SintList.internalBinaryWrite(message.value.sintList, writer.tag(22, runtime_2.WireType.LengthDelimited).fork(), options).join();
        /* flwr.proto.UintList uint_list = 23; */
        if (message.value.oneofKind === "uintList")
            exports.UintList.internalBinaryWrite(message.value.uintList, writer.tag(23, runtime_2.WireType.LengthDelimited).fork(), options).join();
        /* flwr.proto.BoolList bool_list = 24; */
        if (message.value.oneofKind === "boolList")
            exports.BoolList.internalBinaryWrite(message.value.boolList, writer.tag(24, runtime_2.WireType.LengthDelimited).fork(), options).join();
        /* flwr.proto.StringList string_list = 25; */
        if (message.value.oneofKind === "stringList")
            exports.StringList.internalBinaryWrite(message.value.stringList, writer.tag(25, runtime_2.WireType.LengthDelimited).fork(), options).join();
        /* flwr.proto.BytesList bytes_list = 26; */
        if (message.value.oneofKind === "bytesList")
            exports.BytesList.internalBinaryWrite(message.value.bytesList, writer.tag(26, runtime_2.WireType.LengthDelimited).fork(), options).join();
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.ConfigsRecordValue
 */
exports.ConfigsRecordValue = new ConfigsRecordValue$Type();
// @generated message type with reflection information, may provide speed optimized methods
class ParametersRecord$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.ParametersRecord", [
            { no: 1, name: "data_keys", kind: "scalar", repeat: 2 /*RepeatType.UNPACKED*/, T: 9 /*ScalarType.STRING*/ },
            { no: 2, name: "data_values", kind: "message", repeat: 1 /*RepeatType.PACKED*/, T: () => exports.Array$ }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.dataKeys = [];
        message.dataValues = [];
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* repeated string data_keys */ 1:
                    message.dataKeys.push(reader.string());
                    break;
                case /* repeated flwr.proto.Array data_values */ 2:
                    message.dataValues.push(exports.Array$.internalBinaryRead(reader, reader.uint32(), options));
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
            }
        }
        return message;
    }
    internalBinaryWrite(message, writer, options) {
        /* repeated string data_keys = 1; */
        for (let i = 0; i < message.dataKeys.length; i++)
            writer.tag(1, runtime_2.WireType.LengthDelimited).string(message.dataKeys[i]);
        /* repeated flwr.proto.Array data_values = 2; */
        for (let i = 0; i < message.dataValues.length; i++)
            exports.Array$.internalBinaryWrite(message.dataValues[i], writer.tag(2, runtime_2.WireType.LengthDelimited).fork(), options).join();
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.ParametersRecord
 */
exports.ParametersRecord = new ParametersRecord$Type();
// @generated message type with reflection information, may provide speed optimized methods
class MetricsRecord$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.MetricsRecord", [
            { no: 1, name: "data", kind: "map", K: 9 /*ScalarType.STRING*/, V: { kind: "message", T: () => exports.MetricsRecordValue } }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.data = {};
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* map<string, flwr.proto.MetricsRecordValue> data */ 1:
                    this.binaryReadMap1(message.data, reader, options);
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
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
                    key = reader.string();
                    break;
                case 2:
                    val = exports.MetricsRecordValue.internalBinaryRead(reader, reader.uint32(), options);
                    break;
                default: throw new globalThis.Error("unknown map entry field for field flwr.proto.MetricsRecord.data");
            }
        }
        map[key ?? ""] = val ?? exports.MetricsRecordValue.create();
    }
    internalBinaryWrite(message, writer, options) {
        /* map<string, flwr.proto.MetricsRecordValue> data = 1; */
        for (let k of globalThis.Object.keys(message.data)) {
            writer.tag(1, runtime_2.WireType.LengthDelimited).fork().tag(1, runtime_2.WireType.LengthDelimited).string(k);
            writer.tag(2, runtime_2.WireType.LengthDelimited).fork();
            exports.MetricsRecordValue.internalBinaryWrite(message.data[k], writer, options);
            writer.join().join();
        }
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.MetricsRecord
 */
exports.MetricsRecord = new MetricsRecord$Type();
// @generated message type with reflection information, may provide speed optimized methods
class ConfigsRecord$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.ConfigsRecord", [
            { no: 1, name: "data", kind: "map", K: 9 /*ScalarType.STRING*/, V: { kind: "message", T: () => exports.ConfigsRecordValue } }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.data = {};
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* map<string, flwr.proto.ConfigsRecordValue> data */ 1:
                    this.binaryReadMap1(message.data, reader, options);
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
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
                    key = reader.string();
                    break;
                case 2:
                    val = exports.ConfigsRecordValue.internalBinaryRead(reader, reader.uint32(), options);
                    break;
                default: throw new globalThis.Error("unknown map entry field for field flwr.proto.ConfigsRecord.data");
            }
        }
        map[key ?? ""] = val ?? exports.ConfigsRecordValue.create();
    }
    internalBinaryWrite(message, writer, options) {
        /* map<string, flwr.proto.ConfigsRecordValue> data = 1; */
        for (let k of globalThis.Object.keys(message.data)) {
            writer.tag(1, runtime_2.WireType.LengthDelimited).fork().tag(1, runtime_2.WireType.LengthDelimited).string(k);
            writer.tag(2, runtime_2.WireType.LengthDelimited).fork();
            exports.ConfigsRecordValue.internalBinaryWrite(message.data[k], writer, options);
            writer.join().join();
        }
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.ConfigsRecord
 */
exports.ConfigsRecord = new ConfigsRecord$Type();
// @generated message type with reflection information, may provide speed optimized methods
class RecordSet$Type extends runtime_4.MessageType {
    constructor() {
        super("flwr.proto.RecordSet", [
            { no: 1, name: "parameters", kind: "map", K: 9 /*ScalarType.STRING*/, V: { kind: "message", T: () => exports.ParametersRecord } },
            { no: 2, name: "metrics", kind: "map", K: 9 /*ScalarType.STRING*/, V: { kind: "message", T: () => exports.MetricsRecord } },
            { no: 3, name: "configs", kind: "map", K: 9 /*ScalarType.STRING*/, V: { kind: "message", T: () => exports.ConfigsRecord } }
        ]);
    }
    create(value) {
        const message = globalThis.Object.create((this.messagePrototype));
        message.parameters = {};
        message.metrics = {};
        message.configs = {};
        if (value !== undefined)
            (0, runtime_3.reflectionMergePartial)(this, message, value);
        return message;
    }
    internalBinaryRead(reader, length, options, target) {
        let message = target ?? this.create(), end = reader.pos + length;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case /* map<string, flwr.proto.ParametersRecord> parameters */ 1:
                    this.binaryReadMap1(message.parameters, reader, options);
                    break;
                case /* map<string, flwr.proto.MetricsRecord> metrics */ 2:
                    this.binaryReadMap2(message.metrics, reader, options);
                    break;
                case /* map<string, flwr.proto.ConfigsRecord> configs */ 3:
                    this.binaryReadMap3(message.configs, reader, options);
                    break;
                default:
                    let u = options.readUnknownField;
                    if (u === "throw")
                        throw new globalThis.Error(`Unknown field ${fieldNo} (wire type ${wireType}) for ${this.typeName}`);
                    let d = reader.skip(wireType);
                    if (u !== false)
                        (u === true ? runtime_1.UnknownFieldHandler.onRead : u)(this.typeName, message, fieldNo, wireType, d);
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
                    key = reader.string();
                    break;
                case 2:
                    val = exports.ParametersRecord.internalBinaryRead(reader, reader.uint32(), options);
                    break;
                default: throw new globalThis.Error("unknown map entry field for field flwr.proto.RecordSet.parameters");
            }
        }
        map[key ?? ""] = val ?? exports.ParametersRecord.create();
    }
    binaryReadMap2(map, reader, options) {
        let len = reader.uint32(), end = reader.pos + len, key, val;
        while (reader.pos < end) {
            let [fieldNo, wireType] = reader.tag();
            switch (fieldNo) {
                case 1:
                    key = reader.string();
                    break;
                case 2:
                    val = exports.MetricsRecord.internalBinaryRead(reader, reader.uint32(), options);
                    break;
                default: throw new globalThis.Error("unknown map entry field for field flwr.proto.RecordSet.metrics");
            }
        }
        map[key ?? ""] = val ?? exports.MetricsRecord.create();
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
                    val = exports.ConfigsRecord.internalBinaryRead(reader, reader.uint32(), options);
                    break;
                default: throw new globalThis.Error("unknown map entry field for field flwr.proto.RecordSet.configs");
            }
        }
        map[key ?? ""] = val ?? exports.ConfigsRecord.create();
    }
    internalBinaryWrite(message, writer, options) {
        /* map<string, flwr.proto.ParametersRecord> parameters = 1; */
        for (let k of globalThis.Object.keys(message.parameters)) {
            writer.tag(1, runtime_2.WireType.LengthDelimited).fork().tag(1, runtime_2.WireType.LengthDelimited).string(k);
            writer.tag(2, runtime_2.WireType.LengthDelimited).fork();
            exports.ParametersRecord.internalBinaryWrite(message.parameters[k], writer, options);
            writer.join().join();
        }
        /* map<string, flwr.proto.MetricsRecord> metrics = 2; */
        for (let k of globalThis.Object.keys(message.metrics)) {
            writer.tag(2, runtime_2.WireType.LengthDelimited).fork().tag(1, runtime_2.WireType.LengthDelimited).string(k);
            writer.tag(2, runtime_2.WireType.LengthDelimited).fork();
            exports.MetricsRecord.internalBinaryWrite(message.metrics[k], writer, options);
            writer.join().join();
        }
        /* map<string, flwr.proto.ConfigsRecord> configs = 3; */
        for (let k of globalThis.Object.keys(message.configs)) {
            writer.tag(3, runtime_2.WireType.LengthDelimited).fork().tag(1, runtime_2.WireType.LengthDelimited).string(k);
            writer.tag(2, runtime_2.WireType.LengthDelimited).fork();
            exports.ConfigsRecord.internalBinaryWrite(message.configs[k], writer, options);
            writer.join().join();
        }
        let u = options.writeUnknownFields;
        if (u !== false)
            (u == true ? runtime_1.UnknownFieldHandler.onWrite : u)(this.typeName, message, writer);
        return writer;
    }
}
/**
 * @generated MessageType for protobuf message flwr.proto.RecordSet
 */
exports.RecordSet = new RecordSet$Type();
