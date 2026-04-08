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
exports.GRPC_MAX_MESSAGE_LENGTH = void 0;
exports.createChannel = createChannel;
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
const grpc_transport_1 = require("@protobuf-ts/grpc-transport");
exports.GRPC_MAX_MESSAGE_LENGTH = 536_870_912; // == 512 * 1024 * 1024
function createChannel(serverAddress, insecure, rootCertificates = null, maxMessageLength = exports.GRPC_MAX_MESSAGE_LENGTH, interceptors = null) {
    // Check for conflicting parameters
    if (insecure && rootCertificates !== null) {
        throw new Error("Invalid configuration: 'root_certificates' should not be provided " +
            "when 'insecure' is set to true. For an insecure connection, omit " +
            "'root_certificates', or set 'insecure' to false for a secure connection.");
    }
    let creds;
    if (insecure === true) {
        creds = grpc.credentials.createInsecure();
        console.debug("Opened insecure gRPC connection (no certificates were passed)");
    }
    else {
        creds = grpc.credentials.createSsl(rootCertificates);
        console.debug("Opened secure gRPC connection using certificates");
    }
    // gRPC channel options
    const clientOptions = {
        "grpc.max_send_message_length": maxMessageLength,
        "grpc.max_receive_message_length": maxMessageLength,
    };
    let rpcOptions = { host: serverAddress, channelCredentials: creds, clientOptions };
    if (interceptors !== null) {
        rpcOptions.interceptors = interceptors;
    }
    return new grpc_transport_1.GrpcTransport(rpcOptions);
}
