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
  Client,
  maybeCallEvaluate,
  maybeCallFit,
  maybeCallGetParameters,
  maybeCallGetProperties,
} from "./client";
import {
  ClientMessage as ProtoClientMessage,
  Reason as ProtoReason,
  ServerMessage_ReconnectIns as ProtoServerMessage_ReconnectIns,
  ClientMessage_DisconnectRes as ProtoClientMessage_DisconnectRes,
  ServerMessage_ReconnectIns,
} from "../protos/flwr/proto/transport";
import { Message, Context, Metadata } from "./typing";
import { RecordSet, ConfigsRecord } from "./recordset";
import {
  getParametersResToRecordSet,
  getPropertiesResToRecordSet,
  recordSetToFitIns,
  recordSetToGetParametersIns,
  recordSetToGetPropertiesIns,
  fitResToRecordSet,
  recordSetToEvaluateIns,
  evaluateResToRecordSet,
} from "./recordset_compat";

const reconnect = (reconnectMsg: ProtoServerMessage_ReconnectIns): [ProtoClientMessage, bigint] => {
  let reason = ProtoReason.ACK;
  let sleepDuration = BigInt(0);
  if (reconnectMsg.seconds !== BigInt(0)) {
    reason = ProtoReason.RECONNECT;
    sleepDuration = reconnectMsg.seconds;
  }

  const disconnectRes: ProtoClientMessage_DisconnectRes = {
    reason,
  };

  return [
    {
      msg: { oneofKind: "disconnectRes", disconnectRes },
    } as ProtoClientMessage,
    sleepDuration,
  ];
};

export const handleControlMessage = (message: Message): [Message | null, number] => {
  if (message.metadata.messageType === "reconnet") {
    let recordset = message.content;
    let seconds = recordset?.configsRecords["config"]["seconds"]!;
    const reconnectMsg = reconnect({ seconds: seconds as bigint } as ServerMessage_ReconnectIns);
    const disconnectMsg = reconnectMsg[0];
    const sleepDuration = reconnectMsg[1];
    if (disconnectMsg.msg.oneofKind === "disconnectRes") {
      let reason = disconnectMsg.msg.disconnectRes.reason as number;
      let recordset = new RecordSet();
      recordset.configsRecords["config"] = new ConfigsRecord({ reason: reason });
      let outMessage = message.createReply(recordset);
      return [outMessage, Number(sleepDuration)];
    }
  }

  return [null, 0];
};

export const handleLegacyMessageFromMsgType = (
  client_fn: (context: Context) => Client,
  message: Message,
  context: Context,
): Message => {
  let client = client_fn(context);
  client.setContext(context);

  let messageType = message.metadata.messageType;
  let outRecordset: RecordSet;

  switch (messageType) {
    case "get_properties": {
      const getPropertiesRes = maybeCallGetProperties(
        client,
        recordSetToGetPropertiesIns(message.content!),
      );
      outRecordset = getPropertiesResToRecordSet(getPropertiesRes);
      break;
    }
    case "get_parameters": {
      const getParametersRes = maybeCallGetParameters(
        client,
        recordSetToGetParametersIns(message.content!),
      );
      outRecordset = getParametersResToRecordSet(getParametersRes, false);
      break;
    }
    case "train": {
      const fitRes = maybeCallFit(client, recordSetToFitIns(message.content!, true));
      outRecordset = fitResToRecordSet(fitRes, false);
      break;
    }
    case "evaluate": {
      const evaluateRes = maybeCallEvaluate(client, recordSetToEvaluateIns(message.content!, true));
      outRecordset = evaluateResToRecordSet(evaluateRes);
      break;
    }
    default: {
      throw `Invalid message type: ${messageType}`;
    }
  }

  return message.createReply(outRecordset);
};

export const validateOutMessage = (outMessage: Message, inMessageMetadata: Metadata) => {
  let outMeta = outMessage.metadata;
  let inMeta = inMessageMetadata;
  if (
    outMeta.runId === inMeta.runId &&
    outMeta.messageId === "" &&
    outMeta.srcNodeId === inMeta.dstNodeId &&
    outMeta.dstNodeId === inMeta.srcNodeId &&
    outMeta.replyToMessage === inMeta.messageId &&
    outMeta.groupId === inMeta.groupId &&
    outMeta.messageType === inMeta.messageType &&
    outMeta.createdAt > inMeta.createdAt
  ) {
    return true;
  }
  return false;
};
