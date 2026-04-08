"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const recordset_compat_1 = require("./recordset_compat");
// Mock data
const mockScalar = "test_scalar";
const mockParameters = {
    tensors: [new Uint8Array([1, 2, 3])],
    tensorType: "float32",
};
const mockFitIns = { parameters: mockParameters, config: { key1: mockScalar } };
const mockFitRes = {
    parameters: mockParameters,
    numExamples: 100,
    metrics: { key2: mockScalar },
    status: { code: 0, message: "OK" },
};
const mockEvaluateIns = { parameters: mockParameters, config: { key3: mockScalar } };
const mockEvaluateRes = {
    loss: 1.5,
    numExamples: 50,
    metrics: { key4: mockScalar },
    status: { code: 0, message: "OK" },
};
const mockGetParametersRes = {
    parameters: mockParameters,
    status: { code: 0, message: "OK" },
};
const mockGetPropertiesRes = {
    properties: { key5: mockScalar },
    status: { code: 0, message: "OK" },
};
describe("RecordSet Compatibility Functions", () => {
    it("should convert recordset to FitIns", () => {
        const recordset = (0, recordset_compat_1.fitInsToRecordSet)(mockFitIns, true);
        const fitIns = (0, recordset_compat_1.recordSetToFitIns)(recordset, true);
        expect(fitIns).toEqual(mockFitIns);
    });
    it("should convert recordset to FitRes", () => {
        const recordset = (0, recordset_compat_1.fitResToRecordSet)(mockFitRes, true);
        const fitRes = (0, recordset_compat_1.recordSetToFitRes)(recordset, true);
        expect(fitRes).toEqual(mockFitRes);
    });
    it("should convert recordset to EvaluateIns", () => {
        const recordset = (0, recordset_compat_1.evaluateInsToRecordSet)(mockEvaluateIns, true);
        const evaluateIns = (0, recordset_compat_1.recordSetToEvaluateIns)(recordset, true);
        expect(evaluateIns).toEqual(mockEvaluateIns);
    });
    it("should convert recordset to EvaluateRes", () => {
        const recordset = (0, recordset_compat_1.evaluateResToRecordSet)(mockEvaluateRes);
        const evaluateRes = (0, recordset_compat_1.recordSetToEvaluateRes)(recordset);
        expect(evaluateRes).toEqual(mockEvaluateRes);
    });
    it("should convert recordset to GetParametersIns", () => {
        const recordset = (0, recordset_compat_1.getParametersInsToRecordSet)({ config: { key6: mockScalar } });
        const getParametersIns = (0, recordset_compat_1.recordSetToGetParametersIns)(recordset);
        expect(getParametersIns).toEqual({ config: { key6: mockScalar } });
    });
    it("should convert recordset to GetPropertiesIns", () => {
        const recordset = (0, recordset_compat_1.getPropertiesInsToRecordSet)({ config: { key7: mockScalar } });
        const getPropertiesIns = (0, recordset_compat_1.recordSetToGetPropertiesIns)(recordset);
        expect(getPropertiesIns).toEqual({ config: { key7: mockScalar } });
    });
    it("should convert GetParametersRes to RecordSet and back", () => {
        const recordset = (0, recordset_compat_1.getParametersResToRecordSet)(mockGetParametersRes, true);
        const getParametersRes = (0, recordset_compat_1.recordSetToGetParametersRes)(recordset, true);
        expect(getParametersRes).toEqual(mockGetParametersRes);
    });
    it("should convert GetPropertiesRes to RecordSet and back", () => {
        const recordset = (0, recordset_compat_1.getPropertiesResToRecordSet)(mockGetPropertiesRes);
        const getPropertiesRes = (0, recordset_compat_1.recordSetToGetPropertiesRes)(recordset);
        expect(getPropertiesRes).toEqual(mockGetPropertiesRes);
    });
});
