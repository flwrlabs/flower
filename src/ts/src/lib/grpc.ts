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
import { GrpcOptions, GrpcTransport } from "@protobuf-ts/grpc-transport";
import { RpcInterceptor, RpcOptions } from "@protobuf-ts/runtime-rpc";

export const GRPC_MAX_MESSAGE_LENGTH = 536_870_912; // == 512 * 1024 * 1024

export function createChannel(
  serverAddress: string,
  insecure: boolean,
  rootCertificates: Buffer | null = null,
  maxMessageLength: number = GRPC_MAX_MESSAGE_LENGTH,
  interceptors: RpcInterceptor[] | null = null,
): GrpcTransport {
  // Check for conflicting parameters
  if (insecure && rootCertificates !== null) {
    throw new Error(
      "Invalid configuration: 'root_certificates' should not be provided " +
      "when 'insecure' is set to true. For an insecure connection, omit " +
      "'root_certificates', or set 'insecure' to false for a secure connection.",
    );
  }

  let creds: grpc.ChannelCredentials;
  if (insecure === true) {
    creds = grpc.credentials.createInsecure();
    console.debug("Opened insecure gRPC connection (no certificates were passed)");
  } else {
    creds = grpc.credentials.createSsl(rootCertificates);
    console.debug("Opened secure gRPC connection using certificates");
  }

  // gRPC channel options
  const clientOptions: grpc.ClientOptions = {
    "grpc.max_send_message_length": maxMessageLength,
    "grpc.max_receive_message_length": maxMessageLength,
  };

  let rpcOptions: GrpcOptions = { host: serverAddress, channelCredentials: creds, clientOptions };

  if (interceptors !== null) {
    rpcOptions.interceptors = interceptors;
  }

  return new GrpcTransport(rpcOptions);
}
