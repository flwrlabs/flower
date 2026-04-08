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
import { handleLegacyMessageFromMsgType, validateOutMessage } from "./message_handler";
import {
  GetPropertiesRes,
  Message,
  Metadata,
  Context,
  GetPropertiesIns,
  Code,
  Error as LocalError,
} from "./typing";
import { Client } from "./client";
import { RecordSet } from "./recordset";
import { getPropertiesInsToRecordSet, getPropertiesResToRecordSet } from "./recordset_compat";

function removeCreatedAtField(metadata: Metadata): Partial<Metadata> {
  const { createdAt, ...rest } = metadata;
  return rest;
}

// Mock ClientWithoutProps and ClientWithProps
class ClientWithoutProps extends Client {
  getParameters() {
    return {
      status: { code: Code.OK, message: "Success" },
      parameters: { tensors: [], tensorType: "" },
    };
  }

  fit() {
    return {
      status: { code: Code.OK, message: "Success" },
      parameters: { tensors: [], tensorType: "" },
      numExamples: 1,
      metrics: {},
    };
  }

  evaluate() {
    return {
      status: { code: Code.OK, message: "Success" },
      loss: 1.0,
      numExamples: 1,
      metrics: {},
    };
  }
}

class ClientWithProps extends ClientWithoutProps {
  getProperties() {
    return {
      status: { code: Code.OK, message: "Success" },
      properties: { str_prop: "val", int_prop: 1 },
    };
  }
}

// Helper function to create the client_fn
const getClientFn = (client: any) => (context: Context) => client;

describe("Message Handler Tests", () => {
  const createMessage = (messageType: string, content: RecordSet) => {
    return new Message(
      {
        runId: BigInt(123),
        messageId: "abc123",
        groupId: "some-group-id",
        srcNodeId: BigInt(0),
        dstNodeId: BigInt(1123),
        replyToMessage: "",
        ttl: 10,
        messageType,
        createdAt: 0,
      },
      content,
      {} as LocalError,
    );
  };

  const context: Context = {
    nodeId: BigInt(1123),
    nodeConfig: {},
    state: new RecordSet(),
    runConfig: {},
  };

  test("Client without get_properties", () => {
    const client = new ClientWithoutProps({} as Context);
    const recordset = getPropertiesInsToRecordSet({} as GetPropertiesIns);
    const message = createMessage("get_properties", recordset);

    const actualMessage = handleLegacyMessageFromMsgType(getClientFn(client), message, context);

    const expectedGetPropertiesRes: GetPropertiesRes = {
      status: {
        code: Code.GET_PROPERTIES_NOT_IMPLEMENTED,
        message: "Client does not implement `get_properties`",
      },
      properties: {},
    };
    const expectedRs = getPropertiesResToRecordSet(expectedGetPropertiesRes);
    const expectedMessage = new Message(
      {
        ...message.metadata,
        messageId: "",
        srcNodeId: BigInt(1123),
        dstNodeId: BigInt(0),
        replyToMessage: message.metadata.messageId,
        ttl: actualMessage.metadata.ttl,
      },
      expectedRs,
      {} as LocalError,
    );

    expect(actualMessage.content).toEqual(expectedMessage.content);
    expect(removeCreatedAtField(actualMessage.metadata)).toMatchObject(
      removeCreatedAtField(expectedMessage.metadata),
    );
    // expect(actualMessage.metadata.createdAt).toBeGreaterThan(message.metadata.createdAt);
  });

  test("Client with get_properties", () => {
    const client = new ClientWithProps({} as Context);
    const recordset = getPropertiesInsToRecordSet({} as GetPropertiesIns);
    const message = createMessage("get_properties", recordset);

    const actualMessage = handleLegacyMessageFromMsgType(getClientFn(client), message, context);

    const expectedGetPropertiesRes: GetPropertiesRes = {
      status: { code: Code.OK, message: "Success" },
      properties: { str_prop: "val", int_prop: 1 },
    };
    const expectedRs = getPropertiesResToRecordSet(expectedGetPropertiesRes);
    const expectedMessage = new Message(
      {
        ...message.metadata,
        messageId: "",
        srcNodeId: BigInt(1123),
        dstNodeId: BigInt(0),
        replyToMessage: message.metadata.messageId,
        ttl: actualMessage.metadata.ttl,
      },
      expectedRs,
      {} as LocalError,
    );

    expect(actualMessage.content).toEqual(expectedMessage.content);
    expect(removeCreatedAtField(actualMessage.metadata)).toMatchObject(
      removeCreatedAtField(expectedMessage.metadata),
    );
    // expect(actualMessage.metadata.createdAt).toBeGreaterThan(message.metadata.createdAt);
  });
});

describe("Message Validation", () => {
  let inMetadata: Metadata;
  let validOutMetadata: Metadata;

  beforeEach(() => {
    inMetadata = {
      runId: BigInt(123),
      messageId: "qwerty",
      srcNodeId: BigInt(10),
      dstNodeId: BigInt(20),
      replyToMessage: "",
      groupId: "group1",
      ttl: 100,
      messageType: "train",
      createdAt: Date.now() - 10,
    };

    validOutMetadata = {
      runId: BigInt(123),
      messageId: "",
      srcNodeId: BigInt(20),
      dstNodeId: BigInt(10),
      replyToMessage: "qwerty",
      groupId: "group1",
      ttl: 100,
      messageType: "train",
      createdAt: Date.now(),
    };
  });

  test("Valid message", () => {
    const validMessage: Message = new Message(validOutMetadata, new RecordSet(), {} as LocalError);
    expect(validateOutMessage(validMessage, inMetadata)).toBe(true);
  });

  test("Invalid message run_id", () => {
    const msg: Message = new Message(validOutMetadata, new RecordSet(), {} as LocalError);

    const invalidMetadata = {
      runId: BigInt(12), // Different runId
      messageId: "qwerty",
      srcNodeId: BigInt(10),
      dstNodeId: BigInt(20),
      replyToMessage: "",
      groupId: "group1",
      ttl: 100,
      messageType: "train",
      createdAt: Date.now() - 10,
    };
    expect(validateOutMessage(msg, invalidMetadata)).toBe(false);
  });
});
