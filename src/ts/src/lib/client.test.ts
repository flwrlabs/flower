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
import {
  Client,
  maybeCallGetProperties,
  maybeCallGetParameters,
  maybeCallFit,
  maybeCallEvaluate,
} from "./client";
import {
  GetParametersIns,
  GetPropertiesIns,
  FitIns,
  EvaluateIns,
  Code,
  GetParametersRes,
  GetPropertiesRes,
  FitRes,
  EvaluateRes,
} from "./typing";
import { RecordSet } from "./recordset";

// Mock classes for testing
class OverridingClient extends Client {
  getParameters(_ins: GetParametersIns): GetParametersRes {
    return {
      status: { code: Code.OK, message: "Success" },
      parameters: { tensors: [], tensorType: "" },
    };
  }

  fit(_ins: FitIns): FitRes {
    return {
      status: { code: Code.OK, message: "Success" },
      parameters: { tensors: [], tensorType: "" },
      numExamples: 1,
      metrics: {},
    };
  }

  evaluate(_ins: EvaluateIns): EvaluateRes {
    return {
      status: { code: Code.OK, message: "Success" },
      loss: 1.0,
      numExamples: 1,
      metrics: {},
    };
  }

  getProperties(_ins: GetPropertiesIns): GetPropertiesRes {
    return {
      status: { code: Code.OK, message: "Success" },
      properties: {},
    };
  }
}

class NotOverridingClient extends Client {
  getParameters(_ins: GetParametersIns): GetParametersRes {
    return {
      status: { code: Code.GET_PARAMETERS_NOT_IMPLEMENTED, message: "Not Implemented" },
      parameters: { tensors: [], tensorType: "" },
    };
  }

  fit(_ins: FitIns): FitRes {
    return {
      status: { code: Code.FIT_NOT_IMPLEMENTED, message: "Not Implemented" },
      parameters: { tensors: [], tensorType: "" },
      numExamples: 0,
      metrics: {},
    };
  }

  evaluate(_ins: EvaluateIns): EvaluateRes {
    return {
      status: { code: Code.EVALUATE_NOT_IMPLEMENTED, message: "Not Implemented" },
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
      state: {} as RecordSet,
      runConfig: {},
    });
    const result = maybeCallGetProperties(client, {} as GetPropertiesIns);
    expect(result.status.code).toBe(Code.OK);
  });

  it("should return GET_PROPERTIES_NOT_IMPLEMENTED when client does not implement getProperties", () => {
    const client = new NotOverridingClient({
      nodeId: BigInt(1),
      nodeConfig: {},
      state: {} as RecordSet,
      runConfig: {},
    });
    const result = maybeCallGetProperties(client, {} as GetPropertiesIns);
    expect(result.status.code).toBe(Code.GET_PROPERTIES_NOT_IMPLEMENTED);
  });
});

// Test Suite for maybeCallGetParameters
describe("maybeCallGetParameters", () => {
  it("should return OK when client implements getParameters", () => {
    const client = new OverridingClient({
      nodeId: BigInt(1),
      nodeConfig: {},
      state: {} as RecordSet,
      runConfig: {},
    });
    const result = maybeCallGetParameters(client, {} as GetParametersIns);
    expect(result.status.code).toBe(Code.OK);
  });

  it("should return GET_PARAMETERS_NOT_IMPLEMENTED when client does not implement getParameters", () => {
    const client = new NotOverridingClient({
      nodeId: BigInt(1),
      nodeConfig: {},
      state: {} as RecordSet,
      runConfig: {},
    });
    const result = maybeCallGetParameters(client, {} as GetParametersIns);
    expect(result.status.code).toBe(Code.GET_PARAMETERS_NOT_IMPLEMENTED);
  });
});

// Test Suite for maybeCallFit
describe("maybeCallFit", () => {
  it("should return OK when client implements fit", () => {
    const client = new OverridingClient({
      nodeId: BigInt(1),
      nodeConfig: {},
      state: {} as RecordSet,
      runConfig: {},
    });
    const result = maybeCallFit(client, {} as FitIns);
    expect(result.status.code).toBe(Code.OK);
  });

  it("should return FIT_NOT_IMPLEMENTED when client does not implement fit", () => {
    const client = new NotOverridingClient({
      nodeId: BigInt(1),
      nodeConfig: {},
      state: {} as RecordSet,
      runConfig: {},
    });
    const result = maybeCallFit(client, {} as FitIns);
    expect(result.status.code).toBe(Code.FIT_NOT_IMPLEMENTED);
  });
});

// Test Suite for maybeCallEvaluate
describe("maybeCallEvaluate", () => {
  it("should return OK when client implements evaluate", () => {
    const client = new OverridingClient({
      nodeId: BigInt(1),
      nodeConfig: {},
      state: {} as RecordSet,
      runConfig: {},
    });
    const result = maybeCallEvaluate(client, {} as EvaluateIns);
    expect(result.status.code).toBe(Code.OK);
  });

  it("should return EVALUATE_NOT_IMPLEMENTED when client does not implement evaluate", () => {
    const client = new NotOverridingClient({
      nodeId: BigInt(1),
      nodeConfig: {},
      state: {} as RecordSet,
      runConfig: {},
    });
    const result = maybeCallEvaluate(client, {} as EvaluateIns);
    expect(result.status.code).toBe(Code.EVALUATE_NOT_IMPLEMENTED);
  });
});
