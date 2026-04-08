"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.validateOutMessage = exports.handleLegacyMessageFromMsgType = exports.handleControlMessage = void 0;
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
const client_1 = require("./client");
const transport_1 = require("../protos/flwr/proto/transport");
const recordset_1 = require("./recordset");
const recordset_compat_1 = require("./recordset_compat");
const reconnect = (reconnectMsg) => {
    let reason = transport_1.Reason.ACK;
    let sleepDuration = BigInt(0);
    if (reconnectMsg.seconds !== BigInt(0)) {
        reason = transport_1.Reason.RECONNECT;
        sleepDuration = reconnectMsg.seconds;
    }
    const disconnectRes = {
        reason,
    };
    return [
        {
            msg: { oneofKind: "disconnectRes", disconnectRes },
        },
        sleepDuration,
    ];
};
const handleControlMessage = (message) => {
    if (message.metadata.messageType === "reconnet") {
        let recordset = message.content;
        let seconds = recordset?.configsRecords["config"]["seconds"];
        const reconnectMsg = reconnect({ seconds: seconds });
        const disconnectMsg = reconnectMsg[0];
        const sleepDuration = reconnectMsg[1];
        if (disconnectMsg.msg.oneofKind === "disconnectRes") {
            let reason = disconnectMsg.msg.disconnectRes.reason;
            let recordset = new recordset_1.RecordSet();
            recordset.configsRecords["config"] = new recordset_1.ConfigsRecord({ reason: reason });
            let outMessage = message.createReply(recordset);
            return [outMessage, Number(sleepDuration)];
        }
    }
    return [null, 0];
};
exports.handleControlMessage = handleControlMessage;
const handleLegacyMessageFromMsgType = (client_fn, message, context) => {
    let client = client_fn(context);
    client.setContext(context);
    let messageType = message.metadata.messageType;
    let outRecordset;
    switch (messageType) {
        case "get_properties": {
            const getPropertiesRes = (0, client_1.maybeCallGetProperties)(client, (0, recordset_compat_1.recordSetToGetPropertiesIns)(message.content));
            outRecordset = (0, recordset_compat_1.getPropertiesResToRecordSet)(getPropertiesRes);
            break;
        }
        case "get_parameters": {
            const getParametersRes = (0, client_1.maybeCallGetParameters)(client, (0, recordset_compat_1.recordSetToGetParametersIns)(message.content));
            outRecordset = (0, recordset_compat_1.getParametersResToRecordSet)(getParametersRes, false);
            break;
        }
        case "train": {
            const fitRes = (0, client_1.maybeCallFit)(client, (0, recordset_compat_1.recordSetToFitIns)(message.content, true));
            outRecordset = (0, recordset_compat_1.fitResToRecordSet)(fitRes, false);
            break;
        }
        case "evaluate": {
            const evaluateRes = (0, client_1.maybeCallEvaluate)(client, (0, recordset_compat_1.recordSetToEvaluateIns)(message.content, true));
            outRecordset = (0, recordset_compat_1.evaluateResToRecordSet)(evaluateRes);
            break;
        }
        default: {
            throw `Invalid message type: ${messageType}`;
        }
    }
    return message.createReply(outRecordset);
};
exports.handleLegacyMessageFromMsgType = handleLegacyMessageFromMsgType;
const validateOutMessage = (outMessage, inMessageMetadata) => {
    let outMeta = outMessage.metadata;
    let inMeta = inMessageMetadata;
    if (outMeta.runId === inMeta.runId &&
        outMeta.messageId === "" &&
        outMeta.srcNodeId === inMeta.dstNodeId &&
        outMeta.dstNodeId === inMeta.srcNodeId &&
        outMeta.replyToMessage === inMeta.messageId &&
        outMeta.groupId === inMeta.groupId &&
        outMeta.messageType === inMeta.messageType &&
        outMeta.createdAt > inMeta.createdAt) {
        return true;
    }
    return false;
};
exports.validateOutMessage = validateOutMessage;
