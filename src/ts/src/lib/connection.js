"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.grpcRequestResponse = grpcRequestResponse;
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
const fs = __importStar(require("fs"));
const grpc_1 = require("./grpc");
const fleet_1 = require("../protos/flwr/proto/fleet");
const heartbeat_1 = require("./heartbeat");
const constants_1 = require("./constants");
const message_handler_1 = require("./message_handler");
const task_handler_1 = require("./task_handler");
const serde_1 = require("./serde");
const client_interceptor_1 = require("./client_interceptor");
const fleet_client_1 = require("../protos/flwr/proto/fleet.client");
async function grpcRequestResponse(serverAddress, insecure, retryInvoker, maxMessageLength = grpc_1.GRPC_MAX_MESSAGE_LENGTH, rootCertificates, authenticationKeys, adapterCls) {
    // If `rootCertificates` is a string, read the certificate file
    if (typeof rootCertificates === "string") {
        rootCertificates = await fs.promises.readFile(rootCertificates);
    }
    // Authentication interceptors
    let interceptors = undefined;
    if (authenticationKeys) {
        interceptors = [(0, client_interceptor_1.AuthenticateClientInterceptor)(authenticationKeys[0], authenticationKeys[1])];
    }
    const channel = (0, grpc_1.createChannel)(serverAddress, insecure, rootCertificates, maxMessageLength);
    // channel.subscribe(onChannelStateChange);
    let stub = new fleet_client_1.FleetClient(channel);
    let metadata = null;
    let node = null;
    const pingStopEvent = (0, heartbeat_1.createStopEvent)();
    // Ping function
    async function ping() {
        if (!node) {
            console.error("Node instance missing");
            return;
        }
        const req = {};
        req.node = node;
        req.pingInterval = constants_1.PING_DEFAULT_INTERVAL;
        // const res = (await retryInvoker.invoke(() =>
        //   stub.ping(req, { timeout: PING_CALL_TIMEOUT }),
        // )) as FinishedUnaryCall<PingRequest, PingResponse>;
        const res = await stub.ping(req, { timeout: constants_1.PING_CALL_TIMEOUT });
        if (!res.response.success) {
            throw new Error("Ping failed unexpectedly.");
        }
        const randomFactor = Math.random() * (constants_1.PING_RANDOM_RANGE[1] - constants_1.PING_RANDOM_RANGE[0]) + constants_1.PING_RANDOM_RANGE[0];
        const nextInterval = constants_1.PING_DEFAULT_INTERVAL * (constants_1.PING_BASE_MULTIPLIER + randomFactor) - constants_1.PING_CALL_TIMEOUT;
        // setTimeout(() => {
        //   if (!pingStopEvent.is_set) {
        //     ping();
        //   }
        // }, nextInterval * 1000); // Convert seconds to milliseconds
    }
    // Create node
    async function createNode() {
        const req = {};
        req.pingInterval = constants_1.PING_DEFAULT_INTERVAL;
        // const res = (await retryInvoker.invoke(() => stub.createNode(req))) as FinishedUnaryCall<
        //   CreateNodeRequest,
        //   CreateNodeResponse
        // >;
        const res = await stub.createNode(req);
        node = res.response.node;
        // startPingLoop(ping, pingStopEvent);
        return node?.nodeId || null;
    }
    // Delete node
    async function deleteNode() {
        if (!node) {
            console.error("Node instance missing");
            return;
        }
        pingStopEvent.set();
        const req = {};
        req.node = node;
        // await retryInvoker.invoke(() => stub.deleteNode(req));
        await stub.deleteNode(req);
        node = null;
    }
    // Receive message
    async function receive() {
        if (!node) {
            console.error("Node instance missing");
            return null;
        }
        const req = {};
        req.node = node;
        req.taskIds = [];
        // const res = (await retryInvoker.invoke(() => stub.pullTaskIns(req))) as FinishedUnaryCall<
        //   PullTaskInsRequest,
        //   PullTaskInsResponse
        // >;
        const res = await stub.pullTaskIns(req);
        let taskIns = (0, task_handler_1.getTaskIns)(res.response);
        if (taskIns && !(taskIns.task?.consumer?.nodeId === node.nodeId && (0, task_handler_1.validateTaskIns)(taskIns))) {
            taskIns = null;
        }
        const inMessage = taskIns ? (0, serde_1.messageFromTaskIns)(taskIns) : null;
        metadata = inMessage?.metadata || null;
        return inMessage;
    }
    // Send message
    async function send(message) {
        if (!node) {
            console.error("ERROR", "Node instance missing");
            return;
        }
        if (!metadata) {
            console.error("ERROR", "No current message");
            return;
        }
        if (!(0, message_handler_1.validateOutMessage)(message, metadata)) {
            console.error("Invalid out message");
            return;
        }
        const taskRes = (0, serde_1.messageToTaskRes)(message);
        let req = fleet_1.PushTaskResRequest.create();
        req.taskResList.push(taskRes);
        req.node = node;
        // await retryInvoker.invoke(() => stub.pushTaskRes(req));
        await stub.pushTaskRes(req);
        metadata = null;
    }
    // Get run
    async function getRun(runId) {
        const req = {};
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
            overrideConfig: res.response.run?.overrideConfig ? (0, serde_1.userConfigFromProto)(res.response.run?.overrideConfig) : {},
        };
    }
    // Get fab
    async function getFab(fabHash) {
        const req = {};
        req.hashStr = fabHash;
        // const res = (await retryInvoker.invoke(() => stub.getFab(req))) as FinishedUnaryCall<
        //   GetFabRequest,
        //   GetFabResponse
        // >;
        const res = await stub.getFab(req);
        return { hashStr: res.response.fab?.hashStr, content: res.response.fab?.content };
    }
    return [receive, send, createNode, deleteNode, getRun, getFab];
}
