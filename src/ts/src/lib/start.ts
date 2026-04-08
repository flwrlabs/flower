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
import { grpcRequestResponse } from "./connection";
import { Client } from "./client";
import { handleControlMessage } from "./message_handler";
import { RetryInvoker, RetryState, exponential, sleep } from "./retry_invoker";
import { GRPC_MAX_MESSAGE_LENGTH } from "./grpc";
import {
  Context,
  ClientFnExt,
  UserConfig,
  Run,
  Message,
  Error as LocalError,
  ErrorCode,
} from "./typing";
import { ClientApp, LoadClientAppError } from "./client_app";
import { parseAddress } from "./address";
import { NodeState } from "./node_state";
import { PathLike } from "fs";


class StopIteration extends Error { };

class AppStateTracker {
  public interrupt: boolean = false;
  public isConnected: boolean = false;

  constructor() {
    this.registerSignalHandler();
  }

  // Register handlers for exit signals (SIGINT and SIGTERM)
  registerSignalHandler(): void {
    const signalHandler = (signal: string): void => {
      console.log(`Received ${signal}. Exiting...`);
      this.interrupt = true;
      throw new StopIteration();
    };

    // Listen for SIGINT and SIGTERM signals
    process.on("SIGINT", () => signalHandler("SIGINT"));
    process.on("SIGTERM", () => signalHandler("SIGTERM"));
  }
}

export async function startClientInternal(
  serverAddress: string,
  nodeConfig: UserConfig,
  grpcMaxMessageLength: number = GRPC_MAX_MESSAGE_LENGTH,
  loadClientAppFn: ((_1: string, _2: string) => Promise<ClientApp>) | null = null,
  clientFn: ClientFnExt | null = null,
  insecure: boolean | null = null,
  maxRetries: number | null = null,
  maxWaitTime: number | null = null,
  client: Client | null = null,
  rootCertificates: string | null = null,
  flwrPath: PathLike | null = null,
): Promise<void> {

  if (insecure === null) {
    insecure = rootCertificates === null;
  }

  if (loadClientAppFn === null) {
    if (clientFn === null && client === null) {
      throw Error("Both \`client_fn\` and \`client\` are \`None\`, but one is required")
    }
    if (clientFn !== null && client !== null) {
      throw Error("Both \`client_fn\` and \`client\` are provided, but only one is allowed")
    }
  }

  if (clientFn === null) {
    function singleClientFactory(_context: Context): Client {
      if (client === null) {
        throw Error("Both \`client_fn\` and \`client\` are \`None\`, but one is required");
      }
      return client;
    }
    clientFn = singleClientFactory;
  }
  function _loadClientApp(_fabId: string, _fabVersion: string): Promise<ClientApp> {
    return new Promise((resolve, reject) => {
      try {
        const clientApp = new ClientApp(clientFn!);
        resolve(clientApp);
      } catch (error) {
        reject(error);
      }
    });
  }
  loadClientAppFn = _loadClientApp;

  let appStateTracker = new AppStateTracker();

  function onSuccess(retryState: RetryState): void {
    appStateTracker.isConnected = true;
    if (retryState.tries > 1) {
      console.log(`Connection successful after ${retryState.elapsedTime} seconds and ${retryState.tries} tries.`)
    }
  }

  function onBackoff(retryState: RetryState): void {
    appStateTracker.isConnected = false;
    if (retryState.tries === 1) {
      console.warn("Connection attempt failed, retrying...")
    } else {
      console.warn(`Connection attempt failed, retrying in ${retryState.actualWait} seconds`)
    }
  }

  function onGiveup(retryState: RetryState): void {
    if (retryState.tries > 1) {
      console.warn(`Giving up reconnection after ${retryState.elapsedTime} seconds and ${retryState.tries} tries.`)
    }
  }

  const retryInvoker = new RetryInvoker(
    exponential, Error, maxRetries ? maxRetries + 1 : null, maxWaitTime, {
    onSuccess,
    onBackoff,
    onGiveup,
  }
  );

  const parsedAdress = parseAddress(serverAddress)
  if (parsedAdress === null) {
    process.exit(`Server address ${serverAddress} cannot be parsed.`);
  }
  const address = parsedAdress.version ? `[${parsedAdress.host}]:${parsedAdress.port}` : `${parsedAdress.host}:${parsedAdress.port}`;

  let nodeState: NodeState | null = null;
  let runs: { [runId: number]: Run } = {};


  while (!appStateTracker.interrupt) {
    let sleepDuration = 0;
    const [receive, send, createNode, deleteNode, getRun, _getFab] = await grpcRequestResponse(
      address,
      rootCertificates === null,
      retryInvoker,
      grpcMaxMessageLength,
      rootCertificates ? rootCertificates : undefined,
      null,
    );
    if (nodeState === null) {
      const nodeId = await createNode();
      if (nodeId === null) {
        throw new Error("Node registration failed");
      }
      nodeState = new NodeState(nodeId, nodeConfig);
    }
    appStateTracker.registerSignalHandler();
    while (!appStateTracker.interrupt) {
      try {
        const message = await receive();
        if (message === null) {
          console.log("Pulling...")
          await sleep(3);
          continue;
        }

        console.log("")
        if (message.metadata.groupId.length > 0) {
          console.log(`[RUN ${message.metadata.runId}, ROUND ${message.metadata.groupId}]`)
        }
        console.log(`Received: ${message.metadata.messageType} message ${message.metadata.messageId}]`)

        let outMessage: Message | null;
        [outMessage, sleepDuration] = handleControlMessage(message);
        if (outMessage) {
          await send(outMessage);
          break;
        }

        const runId = message.metadata.runId;
        const nRunId = Number(runId);
        if (!(nRunId in runs)) {
          runs[nRunId] = await getRun(runId);
        }

        const run = runs[nRunId];
        const fab = null;

        nodeState.registerContext(nRunId, run, flwrPath, fab);
        let context = nodeState.retrieveContext(nRunId);
        let replyMessage = message.createErrorReply({ code: ErrorCode.UNKNOWN, reason: "Unknown" } as LocalError);
        try {
          const clientApp = await loadClientAppFn(run.fabId, run.fabVersion);
          replyMessage = clientApp.call(message, context);
        } catch (err) {
          let errorCode = ErrorCode.CLIENT_APP_RAISED_EXCEPTION;
          let reason = `${typeof err}:<'${err}'>`;
          let excEntity = "ClientApp";
          if (err instanceof LoadClientAppError) {
            reason = "An exception was raised when attempting to load `ClientApp`";
            err = ErrorCode.LOAD_CLIENT_APP_EXCEPTION;
            excEntity = "SuperNode";
          }
          if (!appStateTracker.interrupt) {
            // TODO Add excInfo=err
            console.error(`${excEntity} raised an exception`)
          }
          replyMessage = message.createErrorReply({ code: errorCode, reason });
        }
        context.runConfig = {};
        nodeState.updateContext(nRunId, context);
        await send(replyMessage);
        console.log("Sent reply");
      } catch (err) {
        if (err instanceof StopIteration) {
          sleepDuration = 0;
          break;
        } else {
          console.log(err);
          await sleep(3);
        }
      }
    }
    await deleteNode();

    if (sleepDuration === 0) {
      console.log("Disconnect and shut down")
      break;
    }

    console.log(`Disconnect, then re-establish connection after ${sleepDuration} second(s)`)
    await sleep(sleepDuration);
  }
}
