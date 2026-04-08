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
import { Context, Message, MessageType, ClientFnExt, Mod, ClientAppCallable } from "./typing";
import { handleLegacyMessageFromMsgType } from "./message_handler";
import { warnPreviewFeature } from "./logger";
import { existsSync } from 'fs';
import { join } from 'path';


export function makeFFN(ffn: ClientAppCallable, mods: Mod[]): ClientAppCallable {
  function wrapFFN(_ffn: ClientAppCallable, _mod: Mod): ClientAppCallable {
    return function newFFN(message: Message, context: Context): Message {
      return _mod(message, context, _ffn);
    };
  }

  // Apply each mod to ffn, in reversed order
  for (const mod of mods.reverse()) {
    ffn = wrapFFN(ffn, mod);
  }

  return ffn;
}

function alertErroneousClientFn(): void {
  throw new Error(
    "A `ClientApp` cannot make use of a `client_fn` that does not have a signature in the form: `function client_fn(context: Context)`. You can import the `Context` like this: `import { Context } from './common'`"
  );
}

function inspectMaybeAdaptClientFnSignature(clientFn: ClientFnExt): ClientFnExt {
  if (clientFn.length !== 1) {
    alertErroneousClientFn();
  }

  return clientFn;
}

export class ClientAppException extends Error {
  constructor(message: string) {
    const exName = "ClientAppException";
    super(`\nException ${exName} occurred. Message: ${message}`);
    this.name = exName;
  }
}

export class ClientApp {
  private _mods: Mod[];
  private _call: ClientAppCallable | null = null;
  private _train: ClientAppCallable | null = null;
  private _evaluate: ClientAppCallable | null = null;
  private _query: ClientAppCallable | null = null;

  constructor(clientFn?: ClientFnExt, mods?: Mod[]) {
    this._mods = mods || [];

    if (clientFn) {
      clientFn = inspectMaybeAdaptClientFnSignature(clientFn);

      const ffn: ClientAppCallable = (message, context) => {
        return handleLegacyMessageFromMsgType(clientFn!, message, context);
      };

      this._call = makeFFN(ffn, this._mods);
    }
  }

  call(message: Message, context: Context): Message {
    if (this._call) {
      return this._call(message, context);
    }

    switch (message.metadata.messageType) {
      case MessageType.TRAIN:
        if (this._train) return this._train(message, context);
        throw new Error("No `train` function registered");
      case MessageType.EVALUATE:
        if (this._evaluate) return this._evaluate(message, context);
        throw new Error("No `evaluate` function registered");
      case MessageType.QUERY:
        if (this._query) return this._query(message, context);
        throw new Error("No `query` function registered");
      default:
        throw new Error(`Unknown message_type: ${message.metadata.messageType}`);
    }
  }

  train(): (trainFn: ClientAppCallable) => ClientAppCallable {
    return (trainFn: ClientAppCallable) => {
      if (this._call) {
        throw registrationError("train");
      }

      warnPreviewFeature("ClientApp-register-train-function");
      this._train = makeFFN(trainFn, this._mods);
      return trainFn;
    };
  }

  evaluate(): (evaluateFn: ClientAppCallable) => ClientAppCallable {
    return (evaluateFn: ClientAppCallable) => {
      if (this._call) {
        throw registrationError("evaluate");
      }

      warnPreviewFeature("ClientApp-register-evaluate-function");
      this._evaluate = makeFFN(evaluateFn, this._mods);
      return evaluateFn;
    };
  }

  query(): (queryFn: ClientAppCallable) => ClientAppCallable {
    return (queryFn: ClientAppCallable) => {
      if (this._call) {
        throw registrationError("query");
      }

      warnPreviewFeature("ClientApp-register-query-function");
      this._query = makeFFN(queryFn, this._mods);
      return queryFn;
    };
  }
}

export class LoadClientAppError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LoadClientAppError";
  }
}

function registrationError(fnName: string): Error {
  return new Error(
    `Use either \`@app.${fnName}()\` or \`clientFn\`, but not both.\n\n` +
    `Use the \`ClientApp\` with an existing \`clientFn\`:\n\n` +
    `\`\`\`\nclass FlowerClient extends NumPyClient {}\n\n` +
    `function clientFn(context: Context) {\n` +
    `  return new FlowerClient().toClient();\n` +
    `}\n\n` +
    `const app = new ClientApp({ clientFn });\n\`\`\`\n\n` +
    `Use the \`ClientApp\` with a custom ${fnName} function:\n\n` +
    `\`\`\`\nconst app = new ClientApp();\n\n` +
    `app.${fnName}((message, context) => {\n` +
    `  console.log("ClientApp ${fnName} running");\n` +
    `  return message.createReply({ content: message.content });\n` +
    `});\n\`\`\`\n`
  );
}


async function loadApp(moduleAttributeStr: string): Promise<any> {
  const [modulePath, attributePath] = moduleAttributeStr.split(':');

  if (!modulePath || !attributePath) {
    throw new Error(`Invalid format. Expected '<module>:<attribute>', got '${moduleAttributeStr}'`);
  }

  // Dynamically import the module
  const moduleFullPath = join(process.cwd(), `${modulePath}.js`);
  if (!existsSync(moduleFullPath)) {
    throw new Error(`Module '${modulePath}' not found at '${moduleFullPath}'`);
  }

  const module = await import(moduleFullPath);

  // Access the attribute
  const attributes = attributePath.split('.');
  let attribute = module;
  for (const attr of attributes) {
    if (attribute[attr] === undefined) {
      throw new Error(`Attribute '${attr}' not found in module '${modulePath}'`);
    }
    attribute = attribute[attr];
  }

  return attribute;
}

export function getLoadClientAppFn(defaultAppRef: string, appPath: string | null): (fabId: string, fabVersion: string) => Promise<ClientApp> {
  console.debug(`Flower SuperNode will load and validate ClientApp \`${defaultAppRef}\``);

  return async function(fabId: string, fabVersion: string): Promise<ClientApp> {
    const clientApp = await loadApp(defaultAppRef) as unknown as ClientApp;
    return clientApp;
  }
}
