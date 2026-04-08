"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.Client = void 0;
exports.maybeCallGetProperties = maybeCallGetProperties;
exports.maybeCallGetParameters = maybeCallGetParameters;
exports.maybeCallFit = maybeCallFit;
exports.maybeCallEvaluate = maybeCallEvaluate;
const typing_1 = require("./typing");
class BaseClient {
    context;
    constructor(context) {
        this.context = context;
    }
    setContext(context) {
        this.context = context;
    }
    getContext() {
        return this.context;
    }
}
class Client extends BaseClient {
    getProperties(_ins) {
        return {
            status: {
                code: typing_1.Code.GET_PROPERTIES_NOT_IMPLEMENTED,
                message: "Client does not implement `get_properties`",
            },
            properties: {},
        };
    }
}
exports.Client = Client;
function hasGetProperties(client) {
    return client.getProperties !== undefined;
}
function hasGetParameters(client) {
    return client.getParameters !== undefined;
}
function hasFit(client) {
    return client.fit !== undefined;
}
function hasEvaluate(client) {
    return client.evaluate !== undefined;
}
function maybeCallGetProperties(client, getPropertiesIns) {
    if (!hasGetProperties(client)) {
        const status = {
            code: typing_1.Code.GET_PROPERTIES_NOT_IMPLEMENTED,
            message: "Client does not implement `get_properties`",
        };
        return { status, properties: {} };
    }
    return client.getProperties(getPropertiesIns);
}
function maybeCallGetParameters(client, getParametersIns) {
    if (!hasGetParameters(client)) {
        const status = {
            code: typing_1.Code.GET_PARAMETERS_NOT_IMPLEMENTED,
            message: "Client does not implement `get_parameters`",
        };
        return {
            status,
            parameters: { tensorType: "", tensors: [] },
        };
    }
    return client.getParameters(getParametersIns);
}
function maybeCallFit(client, fitIns) {
    if (!hasFit(client)) {
        const status = {
            code: typing_1.Code.FIT_NOT_IMPLEMENTED,
            message: "Client does not implement `fit`",
        };
        return {
            status,
            parameters: { tensorType: "", tensors: [] },
            numExamples: 0,
            metrics: {},
        };
    }
    return client.fit(fitIns);
}
function maybeCallEvaluate(client, evaluateIns) {
    if (!hasEvaluate(client)) {
        const status = {
            code: typing_1.Code.EVALUATE_NOT_IMPLEMENTED,
            message: "Client does not implement `evaluate`",
        };
        return {
            status,
            loss: 0.0,
            numExamples: 0,
            metrics: {},
        };
    }
    return client.evaluate(evaluateIns);
}
