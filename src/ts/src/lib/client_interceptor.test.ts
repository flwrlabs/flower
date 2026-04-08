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
import * as grpc from "@grpc/grpc-js";
import { computeHMAC, generateSharedKey, publicKeyToBytes } from "./crypto_helpers";
import { generateKeyPair, generateKeyPairSync } from "crypto";
import {
  AUTH_TOKEN_HEADER,
  PUBLIC_KEY_HEADER,
  AuthenticateClientInterceptor,
} from "./client_interceptor";
import { GrpcTransport } from "@protobuf-ts/grpc-transport";
import { CreateNodeRequest, DeleteNodeRequest } from "../protos/flwr/proto/fleet";
import { FleetClient } from "../protos/flwr/proto/fleet.client";
import { GetRunRequest } from "../protos/flwr/proto/run";
import { ec as EC } from "elliptic";

const ec = new EC("p256");

// Mock Servicer for testing
class MockServicer {
  private receivedMetadata: grpc.Metadata | null = null;
  private messageBytes: Buffer | null = null;
  private serverPrivateKey: string | null = null;
  private serverPublicKey: string | null = null;

  constructor() {
    // Asynchronous key generation using elliptic curve
    generateKeyPair("ec", { namedCurve: "secp256k1" }, (err, publicKey, privateKey) => {
      if (err) throw err;
      this.serverPrivateKey = privateKey.export({ format: "pem", type: "pkcs8" }).toString();
      this.serverPublicKey = publicKey.export({ format: "pem", type: "spki" }).toString();
    });
  }

  handleUnaryCall(call: grpc.ServerUnaryCall<any, any>, callback: grpc.sendUnaryData<any>) {
    this.receivedMetadata = call.metadata;
    this.messageBytes = call.request.serializeBinary();

    const publicKeyBytes = publicKeyToBytes(ec.keyFromPublic(this.serverPublicKey!)); // ECC key handling

    if ("pingInterval" in call.request) {
      const responseMetadata = new grpc.Metadata();
      responseMetadata.add(PUBLIC_KEY_HEADER, Buffer.from(publicKeyBytes).toString("base64"));
      callback(null, { nodeId: 123 }, responseMetadata);
    } else if ("node" in call.request) {
      callback(null, {}, undefined);
    } else {
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
let mockServer: MockServicer;

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
