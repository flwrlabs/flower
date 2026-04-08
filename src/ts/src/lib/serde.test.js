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
const serde_1 = require("./serde");
const typing_1 = require("./typing");
const recordset_1 = require("./recordset");
let bytes = new Uint8Array(8);
bytes[0] = 256;
// Mock Protobuf messages and local types for testing
const mockProtoParams = {
    tensors: [bytes, bytes],
    tensorType: "float32",
};
const mockLocalParams = {
    tensors: [bytes, bytes],
    tensorType: "float32",
};
const mockProtoScalar = {
    scalar: { oneofKind: "double", double: 1.23 },
};
const mockLocalScalar = 1.23;
const mockStatus = {
    code: typing_1.Code.OK,
    message: "OK",
};
// Tests for parametersToProto
describe("parametersToProto", () => {
    it("should convert local parameters to proto format", () => {
        const protoParams = (0, serde_1.parametersToProto)(mockLocalParams);
        expect(protoParams).toEqual(mockProtoParams);
    });
});
// Tests for parametersFromProto
describe("parametersFromProto", () => {
    it("should convert proto parameters to local format", () => {
        const localParams = (0, serde_1.parametersFromProto)(mockProtoParams);
        expect(localParams).toEqual(mockLocalParams);
    });
});
// Tests for scalarToProto
describe("scalarToProto", () => {
    it("should convert local scalar to proto format", () => {
        const protoScalar = (0, serde_1.scalarToProto)(mockLocalScalar);
        expect(protoScalar).toEqual(mockProtoScalar);
    });
});
// Tests for scalarFromProto
describe("scalarFromProto", () => {
    it("should convert proto scalar to local format", () => {
        const localScalar = (0, serde_1.scalarFromProto)(mockProtoScalar);
        expect(localScalar).toEqual(mockLocalScalar);
    });
});
// Tests for metricsToProto
describe("metricsToProto", () => {
    it("should convert metrics to proto format", () => {
        const localMetrics = { accuracy: 0.95 };
        const expectedProtoMetrics = { accuracy: { scalar: { oneofKind: "double", double: 0.95 } } };
        const protoMetrics = (0, serde_1.metricsToProto)(localMetrics);
        expect(protoMetrics).toEqual(expectedProtoMetrics);
    });
});
// Tests for metricsFromProto
describe("metricsFromProto", () => {
    it("should convert proto metrics to local format", () => {
        const protoMetrics = {
            accuracy: { scalar: { oneofKind: "double", double: 0.95 } },
        };
        const expectedLocalMetrics = { accuracy: 0.95 };
        const localMetrics = (0, serde_1.metricsFromProto)(protoMetrics);
        expect(localMetrics).toEqual(expectedLocalMetrics);
    });
});
// Tests for parameterResToProto
describe("parameterResToProto", () => {
    it("should convert GetParametersRes to proto format", () => {
        const res = { parameters: mockLocalParams, status: mockStatus };
        const protoRes = (0, serde_1.parameterResToProto)(res);
        expect(protoRes.parameters).toEqual((0, serde_1.parametersToProto)(res.parameters));
        expect(protoRes.status).toEqual((0, serde_1.statusToProto)(res.status));
    });
});
// Tests for fitInsFromProto
describe("fitInsFromProto", () => {
    it("should convert proto FitIns to local format", () => {
        const protoFitIns = {
            parameters: mockProtoParams,
            config: { accuracy: { scalar: { oneofKind: "double", double: 0.95 } } },
        };
        const localFitIns = (0, serde_1.fitInsFromProto)(protoFitIns);
        expect(localFitIns.parameters).toEqual(mockLocalParams);
        expect(localFitIns.config).toEqual({ accuracy: 0.95 });
    });
});
// Tests for fitResToProto
describe("fitResToProto", () => {
    it("should convert FitRes to proto format", () => {
        const localFitRes = {
            parameters: mockLocalParams,
            numExamples: 100,
            metrics: { accuracy: 0.95 },
            status: mockStatus,
        };
        const protoFitRes = (0, serde_1.fitResToProto)(localFitRes);
        expect(protoFitRes.parameters).toEqual((0, serde_1.parametersToProto)(localFitRes.parameters));
        expect(protoFitRes.metrics).toEqual((0, serde_1.metricsToProto)(localFitRes.metrics));
        expect(protoFitRes.status).toEqual((0, serde_1.statusToProto)(localFitRes.status));
    });
});
// Tests for evaluateInsFromProto
describe("evaluateInsFromProto", () => {
    it("should convert proto EvaluateIns to local format", () => {
        const protoEvaluateIns = {
            parameters: mockProtoParams,
            config: { accuracy: { scalar: { oneofKind: "double", double: 0.95 } } },
        };
        const localEvaluateIns = (0, serde_1.evaluateInsFromProto)(protoEvaluateIns);
        expect(localEvaluateIns.parameters).toEqual(mockLocalParams);
        expect(localEvaluateIns.config).toEqual({ accuracy: 0.95 });
    });
});
// Tests for evaluateResToProto
describe("evaluateResToProto", () => {
    it("should convert EvaluateRes to proto format", () => {
        const localEvaluateRes = {
            loss: 0.05,
            numExamples: 100,
            metrics: { accuracy: 0.95 },
            status: mockStatus,
        };
        const protoEvaluateRes = (0, serde_1.evaluateResToProto)(localEvaluateRes);
        expect(protoEvaluateRes.loss).toEqual(localEvaluateRes.loss);
        expect(protoEvaluateRes.metrics).toEqual((0, serde_1.metricsToProto)(localEvaluateRes.metrics));
        expect(protoEvaluateRes.status).toEqual((0, serde_1.statusToProto)(localEvaluateRes.status));
    });
});
// Tests for statusToProto
describe("statusToProto", () => {
    it("should convert local status to proto format", () => {
        const protoStatus = (0, serde_1.statusToProto)(mockStatus);
        expect(protoStatus).toEqual({ code: typing_1.Code.OK, message: "OK" });
    });
});
// Tests for getPropertiesInsFromProto
describe("getPropertiesInsFromProto", () => {
    it("should convert proto GetPropertiesIns to local format", () => {
        const protoGetPropertiesIns = {
            config: { accuracy: { scalar: { oneofKind: "double", double: 0.95 } } },
        };
        const localGetPropertiesIns = (0, serde_1.getPropertiesInsFromProto)(protoGetPropertiesIns);
        expect(localGetPropertiesIns.config).toEqual({ accuracy: 0.95 });
    });
});
// Tests for getPropertiesResToProto
describe("getPropertiesResToProto", () => {
    it("should convert GetPropertiesRes to proto format", () => {
        const localGetPropertiesRes = { properties: { accuracy: 0.95 }, status: mockStatus };
        const protoGetPropertiesRes = (0, serde_1.getPropertiesResToProto)(localGetPropertiesRes);
        expect(protoGetPropertiesRes.properties).toEqual((0, serde_1.propertiesToProto)(localGetPropertiesRes.properties));
        expect(protoGetPropertiesRes.status).toEqual((0, serde_1.statusToProto)(localGetPropertiesRes.status));
    });
});
// Tests for recordSetToProto and recordSetFromProto
describe("recordSetToProto and recordSetFromProto", () => {
    it("should convert local record set to proto and back", () => {
        const localRecordSet = new recordset_1.RecordSet({
            parametersRecord1: new recordset_1.ParametersRecord({
                tensor1: new recordset_1.ArrayData("float32", [1, 2], "NDArray", new Uint8Array([1, 2])),
            }),
        }, {}, {});
        const protoRecordSet = (0, serde_1.recordSetToProto)(localRecordSet);
        const recoveredRecordSet = (0, serde_1.recordSetFromProto)(protoRecordSet);
        expect(recoveredRecordSet).toEqual(localRecordSet);
    });
});
// Tests for messageFromTaskIns and messageToTaskRes
describe("messageFromTaskIns and messageToTaskRes", () => {
    it("should convert taskIns to message and back to taskRes", () => {
        const mockTaskIns = {
            runId: BigInt(1),
            taskId: "task1",
            groupId: "group1",
            task: {
                consumer: { nodeId: BigInt(1), anonymous: false },
                producer: { nodeId: BigInt(2), anonymous: false },
                taskType: "train",
                ttl: 10,
            },
        };
        const message = (0, serde_1.messageFromTaskIns)(mockTaskIns);
        const taskRes = (0, serde_1.messageToTaskRes)(message);
        expect(taskRes.task?.taskType).toEqual(mockTaskIns.task?.taskType);
    });
});
// Tests for userConfigFromProto and userConfigValueToProto
describe("userConfigFromProto and userConfigValueToProto", () => {
    it("should convert user config from proto and back", () => {
        const protoConfig = { key1: { scalar: { oneofKind: "double", double: 0.95 } } };
        const localConfig = { key1: 0.95 };
        const recoveredConfig = (0, serde_1.userConfigFromProto)(protoConfig);
        expect(recoveredConfig).toEqual(localConfig);
    });
});
