"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Message = exports.MessageType = exports.ErrorCode = exports.Code = void 0;
const DEFAULT_TTL = 3600;
var Code;
(function (Code) {
    Code[Code["OK"] = 0] = "OK";
    Code[Code["GET_PROPERTIES_NOT_IMPLEMENTED"] = 1] = "GET_PROPERTIES_NOT_IMPLEMENTED";
    Code[Code["GET_PARAMETERS_NOT_IMPLEMENTED"] = 2] = "GET_PARAMETERS_NOT_IMPLEMENTED";
    Code[Code["FIT_NOT_IMPLEMENTED"] = 3] = "FIT_NOT_IMPLEMENTED";
    Code[Code["EVALUATE_NOT_IMPLEMENTED"] = 4] = "EVALUATE_NOT_IMPLEMENTED";
})(Code || (exports.Code = Code = {}));
var ErrorCode;
(function (ErrorCode) {
    ErrorCode[ErrorCode["UNKNOWN"] = 0] = "UNKNOWN";
    ErrorCode[ErrorCode["LOAD_CLIENT_APP_EXCEPTION"] = 1] = "LOAD_CLIENT_APP_EXCEPTION";
    ErrorCode[ErrorCode["CLIENT_APP_RAISED_EXCEPTION"] = 2] = "CLIENT_APP_RAISED_EXCEPTION";
    ErrorCode[ErrorCode["NODE_UNAVAILABLE"] = 3] = "NODE_UNAVAILABLE";
})(ErrorCode || (exports.ErrorCode = ErrorCode = {}));
var MessageType;
(function (MessageType) {
    MessageType["TRAIN"] = "train";
    MessageType["EVALUATE"] = "evaluate";
    MessageType["QUERY"] = "query";
})(MessageType || (exports.MessageType = MessageType = {}));
class Message {
    metadata;
    content;
    error;
    constructor(metadata, content, error) {
        if (!content && !error) {
            throw "Either `content` or `error` must be set, but not both.";
        }
        // Here we divide by 1000 because Python's time.time() is in s while
        // here it is in ms by default
        metadata.createdAt = (new Date()).valueOf() / 1000;
        this.metadata = metadata;
        this.content = content;
        this.error = error;
    }
    createErrorReply = (error, ttl = null) => {
        if (ttl) {
            console.warn("A custom TTL was set, but note that the SuperLink does not enforce the TTL yet. The SuperLink will start enforcing the TTL in a future version of Flower.");
        }
        const ttl_ = ttl ? ttl : DEFAULT_TTL;
        let message = new Message(createReplyMetadata(this, ttl_), null, error);
        if (!ttl) {
            ttl = this.metadata.ttl - (message.metadata.createdAt - this.metadata.createdAt);
            message.metadata.ttl = ttl;
        }
        return message;
    };
    createReply = (content, ttl = null) => {
        if (ttl) {
            console.warn("A custom TTL was set, but note that the SuperLink does not enforce the TTL yet. The SuperLink will start enforcing the TTL in a future version of Flower.");
        }
        const ttl_ = ttl !== null ? ttl : DEFAULT_TTL;
        let message = new Message(createReplyMetadata(this, ttl_), content, null);
        if (!ttl) {
            ttl = this.metadata.ttl - (message.metadata.createdAt - this.metadata.createdAt);
            message.metadata.ttl = ttl;
        }
        return message;
    };
}
exports.Message = Message;
const createReplyMetadata = (msg, ttl) => {
    return {
        runId: msg.metadata.runId,
        messageId: "",
        srcNodeId: msg.metadata.dstNodeId,
        dstNodeId: msg.metadata.srcNodeId,
        replyToMessage: msg.metadata.messageId,
        groupId: msg.metadata.groupId,
        ttl: ttl,
        messageType: msg.metadata.messageType,
    };
};
