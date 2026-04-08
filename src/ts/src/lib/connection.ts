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
import * as fs from "fs";
import { FinishedUnaryCall, RpcInterceptor } from "@protobuf-ts/runtime-rpc";

import { createChannel, GRPC_MAX_MESSAGE_LENGTH } from "./grpc";
import {
  PingRequest,
  PingResponse,
  CreateNodeRequest,
  CreateNodeResponse,
  DeleteNodeRequest,
  PullTaskInsRequest,
  PullTaskInsResponse,
  PushTaskResRequest,
} from "../protos/flwr/proto/fleet";
import { GetRunRequest, GetRunResponse } from "../protos/flwr/proto/run";
import { GetFabRequest, GetFabResponse } from "../protos/flwr/proto/fab";
import { Node } from "../protos/flwr/proto/node";
import { TaskIns } from "../protos/flwr/proto/task";
import { Metadata, Message, Run, Fab } from "./typing";
import { ec } from "elliptic";
import { startPingLoop, createStopEvent } from "./heartbeat";
import {
  PING_CALL_TIMEOUT,
  PING_RANDOM_RANGE,
  PING_BASE_MULTIPLIER,
  PING_DEFAULT_INTERVAL,
} from "./constants";
import { validateOutMessage } from "./message_handler";
import { getTaskIns, validateTaskIns } from "./task_handler";
import { messageFromTaskIns, messageToTaskRes, userConfigFromProto } from "./serde";
import { RetryInvoker } from "./retry_invoker";
import { AuthenticateClientInterceptor } from "./client_interceptor";
import { FleetClient } from "../protos/flwr/proto/fleet.client";

type GrpcRequestResponseReturnType = [
  () => Promise<Message | null>,
  (message: Message) => Promise<void>,
  () => Promise<bigint | null>,
  () => Promise<void>,
  (run_id: bigint) => Promise<Run>,
  (fabHash: string) => Promise<Fab>,
];

export async function grpcRequestResponse(
  serverAddress: string,
  insecure: boolean,
  retryInvoker: RetryInvoker,
  maxMessageLength: number = GRPC_MAX_MESSAGE_LENGTH,
  rootCertificates?: Buffer | string,
  authenticationKeys?: [ec.KeyPair, ec.KeyPair] | null,
  adapterCls?: any,
): Promise<GrpcRequestResponseReturnType> {
  // If `rootCertificates` is a string, read the certificate file
  if (typeof rootCertificates === "string") {
    rootCertificates = await fs.promises.readFile(rootCertificates);
  }

  // Authentication interceptors
  let interceptors: RpcInterceptor[] | undefined = undefined;
  if (authenticationKeys) {
    interceptors = [AuthenticateClientInterceptor(authenticationKeys[0], authenticationKeys[1])];
  }

  const channel = createChannel(
    serverAddress,
    insecure,
    rootCertificates,
    maxMessageLength,
    // interceptors,
  );
  // channel.subscribe(onChannelStateChange);

  let stub = new FleetClient(channel);
  let metadata: Metadata | null = null;
  let node: Node | null = null;
  const pingStopEvent = createStopEvent();

  // Ping function
  async function ping(): Promise<void> {
    if (!node) {
      console.error("Node instance missing");
      return;
    }

    const req = {} as PingRequest;
    req.node = node;
    req.pingInterval = PING_DEFAULT_INTERVAL;

    // const res = (await retryInvoker.invoke(() =>
    //   stub.ping(req, { timeout: PING_CALL_TIMEOUT }),
    // )) as FinishedUnaryCall<PingRequest, PingResponse>;
    const res = await stub.ping(req, { timeout: PING_CALL_TIMEOUT });
    if (!res.response.success) {
      throw new Error("Ping failed unexpectedly.");
    }

    const randomFactor =
      Math.random() * (PING_RANDOM_RANGE[1] - PING_RANDOM_RANGE[0]) + PING_RANDOM_RANGE[0];
    const nextInterval =
      PING_DEFAULT_INTERVAL * (PING_BASE_MULTIPLIER + randomFactor) - PING_CALL_TIMEOUT;

    // setTimeout(() => {
    //   if (!pingStopEvent.is_set) {
    //     ping();
    //   }
    // }, nextInterval * 1000); // Convert seconds to milliseconds
  }

  // Create node
  async function createNode(): Promise<bigint | null> {
    const req = {} as CreateNodeRequest;
    req.pingInterval = PING_DEFAULT_INTERVAL;

    // const res = (await retryInvoker.invoke(() => stub.createNode(req))) as FinishedUnaryCall<
    //   CreateNodeRequest,
    //   CreateNodeResponse
    // >;
    const res = await stub.createNode(req);

    node = res.response.node!;
    // startPingLoop(ping, pingStopEvent);

    return node?.nodeId || null;
  }

  // Delete node
  async function deleteNode(): Promise<void> {
    if (!node) {
      console.error("Node instance missing");
      return;
    }

    pingStopEvent.set();

    const req = {} as DeleteNodeRequest;
    req.node = node;

    // await retryInvoker.invoke(() => stub.deleteNode(req));
    await stub.deleteNode(req);

    node = null;
  }

  // Receive message
  async function receive(): Promise<Message | null> {
    if (!node) {
      console.error("Node instance missing");
      return null;
    }

    const req = {} as PullTaskInsRequest;
    req.node = node;
    req.taskIds = [];

    // const res = (await retryInvoker.invoke(() => stub.pullTaskIns(req))) as FinishedUnaryCall<
    //   PullTaskInsRequest,
    //   PullTaskInsResponse
    // >;
    const res = await stub.pullTaskIns(req);

    let taskIns: TaskIns | null = getTaskIns(res.response);

    if (taskIns && !(taskIns.task?.consumer?.nodeId === node.nodeId && validateTaskIns(taskIns))) {
      taskIns = null;
    }

    const inMessage = taskIns ? messageFromTaskIns(taskIns) : null;
    metadata = inMessage?.metadata || null;
    return inMessage;
  }

  // Send message
  async function send(message: Message): Promise<void> {
    if (!node) {
      console.error("ERROR", "Node instance missing");
      return;
    }

    if (!metadata) {
      console.error("ERROR", "No current message");
      return;
    }

    if (!validateOutMessage(message, metadata)) {
      console.error("Invalid out message");
      return;
    }

    const taskRes = messageToTaskRes(message);
    let req = PushTaskResRequest.create();
    req.taskResList.push(taskRes);
    req.node = node;

    // await retryInvoker.invoke(() => stub.pushTaskRes(req));
    await stub.pushTaskRes(req);

    metadata = null;
  }

  // Get run
  async function getRun(runId: bigint): Promise<Run> {
    const req = {} as GetRunRequest;
    req.runId = runId;

    // const res = (await retryInvoker.invoke(() => stub.getRun(req))) as FinishedUnaryCall<
    //   GetRunRequest,
    //   GetRunResponse
    // >;
    const res = await stub.getRun(req);

    return {
      runId,
      fabId: res.response.run?.fabId,
      fabVersion: res.response.run?.fabVersion,
      fabHash: res.response.run?.fabHash,
      overrideConfig: res.response.run?.overrideConfig ? userConfigFromProto(res.response.run?.overrideConfig) : {},
    } as Run;
  }

  // Get fab
  async function getFab(fabHash: string): Promise<Fab> {
    const req = {} as GetFabRequest;
    req.hashStr = fabHash;

    // const res = (await retryInvoker.invoke(() => stub.getFab(req))) as FinishedUnaryCall<
    //   GetFabRequest,
    //   GetFabResponse
    // >;
    const res = await stub.getFab(req);

    return { hashStr: res.response.fab?.hashStr, content: res.response.fab?.content } as Fab;
  }

  return [receive, send, createNode, deleteNode, getRun, getFab];
}
