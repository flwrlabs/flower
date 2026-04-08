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
import { RecordSet } from "./recordset";
import { Client } from "./client";

const DEFAULT_TTL = 3600;

export type Scalar = boolean | number | bigint | string | Uint8Array;
export type Config = { [index: string]: Scalar };
export type Properties = { [index: string]: Scalar };
export type Metrics = { [index: string]: Scalar };

export type UserConfigValue = boolean | bigint | number | string;
export type UserConfig = { [index: string]: UserConfigValue };

export type ClientFn = (cid: string) => Client;
export type ClientFnExt = (context: Context) => Client;

export type ClientAppCallable = (msg: Message, context: Context) => Message;
export type Mod = (msg: Message, context: Context, call_next: ClientAppCallable) => Message;

export enum Code {
  OK = 0,
  GET_PROPERTIES_NOT_IMPLEMENTED = 1,
  GET_PARAMETERS_NOT_IMPLEMENTED = 2,
  FIT_NOT_IMPLEMENTED = 3,
  EVALUATE_NOT_IMPLEMENTED = 4,
}

export enum ErrorCode {
  UNKNOWN = 0,
  LOAD_CLIENT_APP_EXCEPTION = 1,
  CLIENT_APP_RAISED_EXCEPTION = 2,
  NODE_UNAVAILABLE = 3,
}

export interface Status {
  code: Code;
  message: string;
}

export interface Parameters {
  tensors: Uint8Array[];
  tensorType: string;
}

export interface GetPropertiesIns {
  config: Config;
}

export interface GetPropertiesRes {
  status: Status;
  properties: Properties;
}

export interface GetParametersIns {
  config: Config;
}

export interface GetParametersRes {
  status: Status;
  parameters: Parameters;
}

export interface FitIns {
  parameters: Parameters;
  config: Config;
}

export interface FitRes {
  status: Status;
  parameters: Parameters;
  numExamples: number;
  metrics: Metrics;
}

export interface EvaluateIns {
  parameters: Parameters;
  config: Config;
}

export interface EvaluateRes {
  status: Status;
  loss: number;
  numExamples: number;
  metrics: Metrics;
}

export interface ServerMessage {
  getPropertiesIns: GetPropertiesIns | null;
  getParametersIns: GetParametersIns | null;
  fitIns: FitIns | null;
  evaluateIns: EvaluateIns | null;
}

export interface ClientMessage {
  getPropertiesRes: GetPropertiesRes | null;
  getParametersRes: GetParametersRes | null;
  fitRes: FitRes | null;
  evaluateRes: EvaluateRes | null;
}

export interface Run {
  runId: bigint;
  fabId: string;
  fabVersion: string;
  fabHash: string;
  overrideConfig: UserConfig;
}

export interface Fab {
  hashStr: string;
  content: Uint8Array;
}

export interface Context {
  nodeId: bigint;
  nodeConfig: UserConfig;
  state: RecordSet;
  runConfig: UserConfig;
}

export interface Metadata {
  runId: bigint;
  messageId: string;
  srcNodeId: bigint;
  dstNodeId: bigint;
  replyToMessage: string;
  groupId: string;
  ttl: number;
  messageType: string;
  createdAt: number;
}

export interface Error {
  code: number;
  reason: string | null;
}


export enum MessageType {
  TRAIN = "train",
  EVALUATE = "evaluate",
  QUERY = "query",
}

export class Message {
  metadata: Metadata;
  content: RecordSet | null;
  error: Error | null;

  constructor(metadata: Metadata, content: RecordSet | null, error: Error | null) {
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

  createErrorReply = (error: Error, ttl: number | null = null) => {
    if (ttl) {
      console.warn(
        "A custom TTL was set, but note that the SuperLink does not enforce the TTL yet. The SuperLink will start enforcing the TTL in a future version of Flower.",
      );
    }
    const ttl_ = ttl ? ttl : DEFAULT_TTL;
    let message = new Message(createReplyMetadata(this, ttl_), null, error);

    if (!ttl) {
      ttl = this.metadata.ttl - (message.metadata.createdAt - this.metadata.createdAt);
      message.metadata.ttl = ttl;
    }
    return message;
  };

  createReply = (content: RecordSet, ttl: number | null = null) => {
    if (ttl) {
      console.warn(
        "A custom TTL was set, but note that the SuperLink does not enforce the TTL yet. The SuperLink will start enforcing the TTL in a future version of Flower.",
      );
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

const createReplyMetadata = (msg: Message, ttl: number) => {
  return {
    runId: msg.metadata.runId,
    messageId: "",
    srcNodeId: msg.metadata.dstNodeId,
    dstNodeId: msg.metadata.srcNodeId,
    replyToMessage: msg.metadata.messageId,
    groupId: msg.metadata.groupId,
    ttl: ttl,
    messageType: msg.metadata.messageType,
  } as Metadata;
};
