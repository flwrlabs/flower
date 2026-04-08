"use strict";
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
const client_1 = require("./client");
const typing_1 = require("./typing");
// Mock classes for testing
class OverridingClient extends client_1.Client {
    getParameters(_ins) {
        return {
            status: { code: typing_1.Code.OK, message: "Success" },
            parameters: { tensors: [], tensorType: "" },
        };
    }
    fit(_ins) {
        return {
            status: { code: typing_1.Code.OK, message: "Success" },
            parameters: { tensors: [], tensorType: "" },
            numExamples: 1,
            metrics: {},
        };
    }
    evaluate(_ins) {
        return {
            status: { code: typing_1.Code.OK, message: "Success" },
            loss: 1.0,
            numExamples: 1,
            metrics: {},
        };
    }
    getProperties(_ins) {
        return {
            status: { code: typing_1.Code.OK, message: "Success" },
            properties: {},
        };
    }
}
class NotOverridingClient extends client_1.Client {
    getParameters(_ins) {
        return {
            status: { code: typing_1.Code.GET_PARAMETERS_NOT_IMPLEMENTED, message: "Not Implemented" },
            parameters: { tensors: [], tensorType: "" },
        };
    }
    fit(_ins) {
        return {
            status: { code: typing_1.Code.FIT_NOT_IMPLEMENTED, message: "Not Implemented" },
            parameters: { tensors: [], tensorType: "" },
            numExamples: 0,
            metrics: {},
        };
    }
    evaluate(_ins) {
        return {
            status: { code: typing_1.Code.EVALUATE_NOT_IMPLEMENTED, message: "Not Implemented" },
            loss: 0.0,
            numExamples: 0,
            metrics: {},
        };
    }
}
// Test Suite for maybeCallGetProperties
describe("maybeCallGetProperties", () => {
    it("should return OK when client implements getProperties", () => {
        const client = new OverridingClient({
            nodeId: BigInt(1),
            nodeConfig: {},
            state: {},
            runConfig: {},
        });
        const result = (0, client_1.maybeCallGetProperties)(client, {});
        expect(result.status.code).toBe(typing_1.Code.OK);
    });
    it("should return GET_PROPERTIES_NOT_IMPLEMENTED when client does not implement getProperties", () => {
        const client = new NotOverridingClient({
            nodeId: BigInt(1),
            nodeConfig: {},
            state: {},
            runConfig: {},
        });
        const result = (0, client_1.maybeCallGetProperties)(client, {});
        expect(result.status.code).toBe(typing_1.Code.GET_PROPERTIES_NOT_IMPLEMENTED);
    });
});
// Test Suite for maybeCallGetParameters
describe("maybeCallGetParameters", () => {
    it("should return OK when client implements getParameters", () => {
        const client = new OverridingClient({
            nodeId: BigInt(1),
            nodeConfig: {},
            state: {},
            runConfig: {},
        });
        const result = (0, client_1.maybeCallGetParameters)(client, {});
        expect(result.status.code).toBe(typing_1.Code.OK);
    });
    it("should return GET_PARAMETERS_NOT_IMPLEMENTED when client does not implement getParameters", () => {
        const client = new NotOverridingClient({
            nodeId: BigInt(1),
            nodeConfig: {},
            state: {},
            runConfig: {},
        });
        const result = (0, client_1.maybeCallGetParameters)(client, {});
        expect(result.status.code).toBe(typing_1.Code.GET_PARAMETERS_NOT_IMPLEMENTED);
    });
});
// Test Suite for maybeCallFit
describe("maybeCallFit", () => {
    it("should return OK when client implements fit", () => {
        const client = new OverridingClient({
            nodeId: BigInt(1),
            nodeConfig: {},
            state: {},
            runConfig: {},
        });
        const result = (0, client_1.maybeCallFit)(client, {});
        expect(result.status.code).toBe(typing_1.Code.OK);
    });
    it("should return FIT_NOT_IMPLEMENTED when client does not implement fit", () => {
        const client = new NotOverridingClient({
            nodeId: BigInt(1),
            nodeConfig: {},
            state: {},
            runConfig: {},
        });
        const result = (0, client_1.maybeCallFit)(client, {});
        expect(result.status.code).toBe(typing_1.Code.FIT_NOT_IMPLEMENTED);
    });
});
// Test Suite for maybeCallEvaluate
describe("maybeCallEvaluate", () => {
    it("should return OK when client implements evaluate", () => {
        const client = new OverridingClient({
            nodeId: BigInt(1),
            nodeConfig: {},
            state: {},
            runConfig: {},
        });
        const result = (0, client_1.maybeCallEvaluate)(client, {});
        expect(result.status.code).toBe(typing_1.Code.OK);
    });
    it("should return EVALUATE_NOT_IMPLEMENTED when client does not implement evaluate", () => {
        const client = new NotOverridingClient({
            nodeId: BigInt(1),
            nodeConfig: {},
            state: {},
            runConfig: {},
        });
        const result = (0, client_1.maybeCallEvaluate)(client, {});
        expect(result.status.code).toBe(typing_1.Code.EVALUATE_NOT_IMPLEMENTED);
    });
});
