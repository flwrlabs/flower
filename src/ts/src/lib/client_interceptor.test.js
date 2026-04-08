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
const grpc = __importStar(require("@grpc/grpc-js"));
const crypto_helpers_1 = require("./crypto_helpers");
const crypto_1 = require("crypto");
const client_interceptor_1 = require("./client_interceptor");
const elliptic_1 = require("elliptic");
const ec = new elliptic_1.ec("p256");
// Mock Servicer for testing
class MockServicer {
    receivedMetadata = null;
    messageBytes = null;
    serverPrivateKey = null;
    serverPublicKey = null;
    constructor() {
        // Asynchronous key generation using elliptic curve
        (0, crypto_1.generateKeyPair)("ec", { namedCurve: "secp256k1" }, (err, publicKey, privateKey) => {
            if (err)
                throw err;
            this.serverPrivateKey = privateKey.export({ format: "pem", type: "pkcs8" }).toString();
            this.serverPublicKey = publicKey.export({ format: "pem", type: "spki" }).toString();
        });
    }
    handleUnaryCall(call, callback) {
        this.receivedMetadata = call.metadata;
        this.messageBytes = call.request.serializeBinary();
        const publicKeyBytes = (0, crypto_helpers_1.publicKeyToBytes)(ec.keyFromPublic(this.serverPublicKey)); // ECC key handling
        if ("pingInterval" in call.request) {
            const responseMetadata = new grpc.Metadata();
            responseMetadata.add(client_interceptor_1.PUBLIC_KEY_HEADER, Buffer.from(publicKeyBytes).toString("base64"));
            callback(null, { nodeId: 123 }, responseMetadata);
        }
        else if ("node" in call.request) {
            callback(null, {}, undefined);
        }
        else {
            callback({ code: grpc.status.INVALID_ARGUMENT, message: "Unknown request" });
        }
    }
    getReceivedMetadata() {
        return this.receivedMetadata;
    }
    getMessageBytes() {
        return this.messageBytes;
    }
    getServerPublicKey() {
        return this.serverPublicKey;
    }
    getServerPrivateKey() {
        return this.serverPrivateKey;
    }
}
// Setup and teardown for tests
let mockServer;
beforeAll(() => {
    mockServer = new MockServicer();
});
afterAll(() => {
    // Stop server if necessary
});
// Test: Authenticate Client with Create Node
// test('should authenticate client with create node', async () => {
//   const retryInvoker = {}; // Mock retry invoker
//   const { privateKey, publicKey } = generateKeyPairSync('rsa');
//   const interceptor = AuthenticateClientInterceptor(privateKey, publicKey);
//   const transport = new GrpcTransport({
//     host: "localhost:50051",
//     channelCredentials: grpc.credentials.createInsecure(),
//     interceptors: [interceptor],
//   });
//   const client = new FleetClient(transport);
//   // const client = new grpc.Client('localhost:50051', grpc.credentials.createInsecure(), {
//   //   interceptors: [interceptor],
//   // });
//   const request = CreateNodeRequest.create();
//   const response = await client.unaryUnary(request);
//   client.makeUnaryRequest(request)
//   const receivedMetadata = mockServer.getReceivedMetadata();
//   expect(receivedMetadata.get(PUBLIC_KEY_HEADER)).toBeTruthy();
//   const sharedSecret = generateSharedKey(mockServer.getServerPrivateKey(), publicKey);
//   const hmac = computeHMAC(sharedSecret, mockServer.getMessageBytes());
//   expect(receivedMetadata.get(AUTH_TOKEN_HEADER)).toEqual(hmac);
// });
// // Test: Authenticate Client with Delete Node
// test('should authenticate client with delete node', async () => {
//   const retryInvoker = {}; // Mock retry invoker
//   const { privateKey, publicKey } = generateKeyPairs();
//   const interceptor = AuthenticateClientInterceptor(privateKey, publicKey);
//   const client = new grpc.Client('localhost:50051', grpc.credentials.createInsecure(), {
//     interceptors: [interceptor],
//   });
//   const request = DeleteNodeRequest.create();
//   const response = await client.unaryUnary(request);
//   const receivedMetadata = mockServer.getReceivedMetadata();
//   expect(receivedMetadata!.get(PUBLIC_KEY_HEADER)).toBeTruthy();
//   const sharedSecret = generateSharedKey(mockServer.getServerPrivateKey(), publicKey);
//   const hmac = computeHMAC(sharedSecret, mockServer.getMessageBytes());
//   expect(receivedMetadata!.get(AUTH_TOKEN_HEADER)).toEqual(hmac);
// });
// // Test: Authenticate Client with Get Run
// test('should authenticate client with get run', async () => {
//   const retryInvoker = {}; // Mock retry invoker
//   const { privateKey, publicKey } = generateKeyPairs();
//   const interceptor = AuthenticateClientInterceptor(privateKey, publicKey);
//   const client = new grpc.Client('localhost:50051', grpc.credentials.createInsecure(), {
//     interceptors: [interceptor],
//   });
//   const request = GetRunRequest.create();
//   const response = await client.unaryUnary(request);
//   const receivedMetadata = mockServer.getReceivedMetadata();
//   expect(receivedMetadata.get(PUBLIC_KEY_HEADER)).toBeTruthy();
//   const sharedSecret = generateSharedKey(mockServer.getServerPrivateKey(), publicKey);
//   const hmac = computeHMAC(sharedSecret, mockServer.getMessageBytes());
//   expect(receivedMetadata.get(AUTH_TOKEN_HEADER)).toEqual(hmac);
// });
