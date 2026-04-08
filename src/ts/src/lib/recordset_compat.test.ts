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
  Parameters,
  Scalar,
  FitIns,
  FitRes,
  GetParametersRes,
  GetPropertiesRes,
  EvaluateIns,
  EvaluateRes,
} from "./typing";
import {
  recordSetToFitIns,
  fitInsToRecordSet,
  recordSetToFitRes,
  fitResToRecordSet,
  recordSetToEvaluateIns,
  evaluateInsToRecordSet,
  recordSetToEvaluateRes,
  evaluateResToRecordSet,
  recordSetToGetParametersIns,
  getParametersInsToRecordSet,
  recordSetToGetPropertiesIns,
  getPropertiesInsToRecordSet,
  getParametersResToRecordSet,
  getPropertiesResToRecordSet,
  recordSetToGetParametersRes,
  recordSetToGetPropertiesRes,
} from "./recordset_compat";

// Mock data
const mockScalar: Scalar = "test_scalar";
const mockParameters: Parameters = {
  tensors: [new Uint8Array([1, 2, 3])],
  tensorType: "float32",
};
const mockFitIns: FitIns = { parameters: mockParameters, config: { key1: mockScalar } };
const mockFitRes: FitRes = {
  parameters: mockParameters,
  numExamples: 100,
  metrics: { key2: mockScalar },
  status: { code: 0, message: "OK" },
};
const mockEvaluateIns: EvaluateIns = { parameters: mockParameters, config: { key3: mockScalar } };
const mockEvaluateRes: EvaluateRes = {
  loss: 1.5,
  numExamples: 50,
  metrics: { key4: mockScalar },
  status: { code: 0, message: "OK" },
};
const mockGetParametersRes: GetParametersRes = {
  parameters: mockParameters,
  status: { code: 0, message: "OK" },
};
const mockGetPropertiesRes: GetPropertiesRes = {
  properties: { key5: mockScalar },
  status: { code: 0, message: "OK" },
};

describe("RecordSet Compatibility Functions", () => {
  it("should convert recordset to FitIns", () => {
    const recordset = fitInsToRecordSet(mockFitIns, true);
    const fitIns = recordSetToFitIns(recordset, true);
    expect(fitIns).toEqual(mockFitIns);
  });

  it("should convert recordset to FitRes", () => {
    const recordset = fitResToRecordSet(mockFitRes, true);
    const fitRes = recordSetToFitRes(recordset, true);
    expect(fitRes).toEqual(mockFitRes);
  });

  it("should convert recordset to EvaluateIns", () => {
    const recordset = evaluateInsToRecordSet(mockEvaluateIns, true);
    const evaluateIns = recordSetToEvaluateIns(recordset, true);
    expect(evaluateIns).toEqual(mockEvaluateIns);
  });

  it("should convert recordset to EvaluateRes", () => {
    const recordset = evaluateResToRecordSet(mockEvaluateRes);
    const evaluateRes = recordSetToEvaluateRes(recordset);
    expect(evaluateRes).toEqual(mockEvaluateRes);
  });

  it("should convert recordset to GetParametersIns", () => {
    const recordset = getParametersInsToRecordSet({ config: { key6: mockScalar } });
    const getParametersIns = recordSetToGetParametersIns(recordset);
    expect(getParametersIns).toEqual({ config: { key6: mockScalar } });
  });

  it("should convert recordset to GetPropertiesIns", () => {
    const recordset = getPropertiesInsToRecordSet({ config: { key7: mockScalar } });
    const getPropertiesIns = recordSetToGetPropertiesIns(recordset);
    expect(getPropertiesIns).toEqual({ config: { key7: mockScalar } });
  });

  it("should convert GetParametersRes to RecordSet and back", () => {
    const recordset = getParametersResToRecordSet(mockGetParametersRes, true);
    const getParametersRes = recordSetToGetParametersRes(recordset, true);
    expect(getParametersRes).toEqual(mockGetParametersRes);
  });

  it("should convert GetPropertiesRes to RecordSet and back", () => {
    const recordset = getPropertiesResToRecordSet(mockGetPropertiesRes);
    const getPropertiesRes = recordSetToGetPropertiesRes(recordset);
    expect(getPropertiesRes).toEqual(mockGetPropertiesRes);
  });
});
