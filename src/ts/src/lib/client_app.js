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
exports.LoadClientAppError = exports.ClientApp = exports.ClientAppException = void 0;
exports.makeFFN = makeFFN;
exports.getLoadClientAppFn = getLoadClientAppFn;
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
const typing_1 = require("./typing");
const message_handler_1 = require("./message_handler");
const logger_1 = require("./logger");
const fs_1 = require("fs");
const path_1 = require("path");
function makeFFN(ffn, mods) {
    function wrapFFN(_ffn, _mod) {
        return function newFFN(message, context) {
            return _mod(message, context, _ffn);
        };
    }
    // Apply each mod to ffn, in reversed order
    for (const mod of mods.reverse()) {
        ffn = wrapFFN(ffn, mod);
    }
    return ffn;
}
function alertErroneousClientFn() {
    throw new Error("A `ClientApp` cannot make use of a `client_fn` that does not have a signature in the form: `function client_fn(context: Context)`. You can import the `Context` like this: `import { Context } from './common'`");
}
function inspectMaybeAdaptClientFnSignature(clientFn) {
    if (clientFn.length !== 1) {
        alertErroneousClientFn();
    }
    return clientFn;
}
class ClientAppException extends Error {
    constructor(message) {
        const exName = "ClientAppException";
        super(`\nException ${exName} occurred. Message: ${message}`);
        this.name = exName;
    }
}
exports.ClientAppException = ClientAppException;
class ClientApp {
    _mods;
    _call = null;
    _train = null;
    _evaluate = null;
    _query = null;
    constructor(clientFn, mods) {
        this._mods = mods || [];
        if (clientFn) {
            clientFn = inspectMaybeAdaptClientFnSignature(clientFn);
            const ffn = (message, context) => {
                return (0, message_handler_1.handleLegacyMessageFromMsgType)(clientFn, message, context);
            };
            this._call = makeFFN(ffn, this._mods);
        }
    }
    call(message, context) {
        if (this._call) {
            return this._call(message, context);
        }
        switch (message.metadata.messageType) {
            case typing_1.MessageType.TRAIN:
                if (this._train)
                    return this._train(message, context);
                throw new Error("No `train` function registered");
            case typing_1.MessageType.EVALUATE:
                if (this._evaluate)
                    return this._evaluate(message, context);
                throw new Error("No `evaluate` function registered");
            case typing_1.MessageType.QUERY:
                if (this._query)
                    return this._query(message, context);
                throw new Error("No `query` function registered");
            default:
                throw new Error(`Unknown message_type: ${message.metadata.messageType}`);
        }
    }
    train() {
        return (trainFn) => {
            if (this._call) {
                throw registrationError("train");
            }
            (0, logger_1.warnPreviewFeature)("ClientApp-register-train-function");
            this._train = makeFFN(trainFn, this._mods);
            return trainFn;
        };
    }
    evaluate() {
        return (evaluateFn) => {
            if (this._call) {
                throw registrationError("evaluate");
            }
            (0, logger_1.warnPreviewFeature)("ClientApp-register-evaluate-function");
            this._evaluate = makeFFN(evaluateFn, this._mods);
            return evaluateFn;
        };
    }
    query() {
        return (queryFn) => {
            if (this._call) {
                throw registrationError("query");
            }
            (0, logger_1.warnPreviewFeature)("ClientApp-register-query-function");
            this._query = makeFFN(queryFn, this._mods);
            return queryFn;
        };
    }
}
exports.ClientApp = ClientApp;
class LoadClientAppError extends Error {
    constructor(message) {
        super(message);
        this.name = "LoadClientAppError";
    }
}
exports.LoadClientAppError = LoadClientAppError;
function registrationError(fnName) {
    return new Error(`Use either \`@app.${fnName}()\` or \`clientFn\`, but not both.\n\n` +
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
        `});\n\`\`\`\n`);
}
async function loadApp(moduleAttributeStr) {
    const [modulePath, attributePath] = moduleAttributeStr.split(':');
    if (!modulePath || !attributePath) {
        throw new Error(`Invalid format. Expected '<module>:<attribute>', got '${moduleAttributeStr}'`);
    }
    // Dynamically import the module
    const moduleFullPath = (0, path_1.join)(process.cwd(), `${modulePath}.js`);
    if (!(0, fs_1.existsSync)(moduleFullPath)) {
        throw new Error(`Module '${modulePath}' not found at '${moduleFullPath}'`);
    }
    const module = await Promise.resolve(`${moduleFullPath}`).then(s => __importStar(require(s)));
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
function getLoadClientAppFn(defaultAppRef, appPath) {
    console.debug(`Flower SuperNode will load and validate ClientApp \`${defaultAppRef}\``);
    return async function (fabId, fabVersion) {
        const clientApp = await loadApp(defaultAppRef);
        return clientApp;
    };
}
