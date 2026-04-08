"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.recordSetToFitIns = recordSetToFitIns;
exports.fitInsToRecordSet = fitInsToRecordSet;
exports.recordSetToFitRes = recordSetToFitRes;
exports.fitResToRecordSet = fitResToRecordSet;
exports.recordSetToEvaluateIns = recordSetToEvaluateIns;
exports.evaluateInsToRecordSet = evaluateInsToRecordSet;
exports.recordSetToEvaluateRes = recordSetToEvaluateRes;
exports.evaluateResToRecordSet = evaluateResToRecordSet;
exports.recordSetToGetParametersIns = recordSetToGetParametersIns;
exports.recordSetToGetPropertiesIns = recordSetToGetPropertiesIns;
exports.getParametersInsToRecordSet = getParametersInsToRecordSet;
exports.getPropertiesInsToRecordSet = getPropertiesInsToRecordSet;
exports.getParametersResToRecordSet = getParametersResToRecordSet;
exports.getPropertiesResToRecordSet = getPropertiesResToRecordSet;
exports.recordSetToGetParametersRes = recordSetToGetParametersRes;
exports.recordSetToGetPropertiesRes = recordSetToGetPropertiesRes;
const recordset_1 = require("./recordset");
function parametersrecordToParameters(record, keepInput) {
    const parameters = { tensors: [], tensorType: "" };
    Object.keys(record).forEach((key) => {
        const arrayData = record[key];
        if (key !== "EMPTY_TENSOR_KEY") {
            parameters.tensors.push(arrayData.data);
        }
        if (!parameters.tensorType) {
            parameters.tensorType = arrayData.stype;
        }
        if (!keepInput) {
            delete record[key];
        }
    });
    return parameters;
}
function parametersToParametersRecord(parameters, keepInput) {
    const tensorType = parameters.tensorType;
    const orderedDict = new recordset_1.ParametersRecord({});
    parameters.tensors.forEach((tensor, idx) => {
        const array = new recordset_1.ArrayData("", [], tensorType, tensor);
        orderedDict[String(idx)] = array;
        if (!keepInput) {
            parameters.tensors.shift();
        }
    });
    if (parameters.tensors.length === 0) {
        orderedDict["EMPTY_TENSOR_KEY"] = new recordset_1.ArrayData("", [], tensorType, new Uint8Array());
    }
    return orderedDict;
}
function checkMappingFromRecordScalarTypeToScalar(recordData) {
    if (!recordData) {
        throw new TypeError("Invalid input: recordData is undefined or null");
    }
    Object.values(recordData).forEach((value) => {
        if (typeof value !== "number" &&
            typeof value !== "string" &&
            typeof value !== "boolean" &&
            !(value instanceof Uint8Array)) {
            throw new TypeError(`Invalid scalar type found: ${typeof value}`);
        }
    });
    return recordData;
}
function recordSetToFitOrEvaluateInsComponents(recordset, insStr, keepInput) {
    const parametersRecord = recordset.parametersRecords[`${insStr}.parameters`];
    const parameters = parametersrecordToParameters(parametersRecord, keepInput);
    const configRecord = recordset.configsRecords[`${insStr}.config`];
    const configDict = checkMappingFromRecordScalarTypeToScalar(configRecord);
    return { parameters, config: configDict };
}
function fitOrEvaluateInsToRecordSet(ins, keepInput, insStr) {
    const recordset = new recordset_1.RecordSet();
    const parametersRecord = parametersToParametersRecord(ins.parameters, keepInput);
    recordset.parametersRecords[`${insStr}.parameters`] = parametersRecord;
    recordset.configsRecords[`${insStr}.config`] = new recordset_1.ConfigsRecord(ins.config);
    return recordset;
}
function embedStatusIntoRecordSet(resStr, status, recordset) {
    const statusDict = {
        code: status.code,
        message: status.message,
    };
    recordset.configsRecords[`${resStr}.status`] = new recordset_1.ConfigsRecord(statusDict);
    return recordset;
}
function extractStatusFromRecordSet(resStr, recordset) {
    const status = recordset.configsRecords[`${resStr}.status`];
    const code = status["code"];
    return { code, message: status["message"] };
}
function recordSetToFitIns(recordset, keepInput) {
    const { parameters, config } = recordSetToFitOrEvaluateInsComponents(recordset, "fitins", keepInput);
    return { parameters, config };
}
function fitInsToRecordSet(fitins, keepInput) {
    return fitOrEvaluateInsToRecordSet(fitins, keepInput, "fitins");
}
function recordSetToFitRes(recordset, keepInput) {
    const insStr = "fitres";
    const parameters = parametersrecordToParameters(recordset.parametersRecords[`${insStr}.parameters`], keepInput);
    const numExamples = recordset.metricsRecords[`${insStr}.num_examples`]["num_examples"];
    const configRecord = recordset.configsRecords[`${insStr}.metrics`];
    const metrics = checkMappingFromRecordScalarTypeToScalar(configRecord);
    const status = extractStatusFromRecordSet(insStr, recordset);
    return { status, parameters, numExamples, metrics };
}
function fitResToRecordSet(fitres, keepInput) {
    const recordset = new recordset_1.RecordSet();
    const resStr = "fitres";
    recordset.configsRecords[`${resStr}.metrics`] = new recordset_1.ConfigsRecord(fitres.metrics);
    recordset.metricsRecords[`${resStr}.num_examples`] = new recordset_1.MetricsRecord({
        num_examples: fitres.numExamples,
    });
    recordset.parametersRecords[`${resStr}.parameters`] = parametersToParametersRecord(fitres.parameters, keepInput);
    return embedStatusIntoRecordSet(resStr, fitres.status, recordset);
}
function recordSetToEvaluateIns(recordset, keepInput) {
    const { parameters, config } = recordSetToFitOrEvaluateInsComponents(recordset, "evaluateins", keepInput);
    return { parameters, config };
}
function evaluateInsToRecordSet(evaluateIns, keepInput) {
    return fitOrEvaluateInsToRecordSet(evaluateIns, keepInput, "evaluateins");
}
function recordSetToEvaluateRes(recordset) {
    const insStr = "evaluateres";
    const loss = recordset.metricsRecords[`${insStr}.loss`]["loss"];
    const numExamples = recordset.metricsRecords[`${insStr}.num_examples`]["numExamples"];
    const configsRecord = recordset.configsRecords[`${insStr}.metrics`];
    const metrics = Object.fromEntries(Object.entries(configsRecord).map(([key, value]) => [key, value]));
    const status = extractStatusFromRecordSet(insStr, recordset);
    return { status, loss, numExamples, metrics };
}
function evaluateResToRecordSet(evaluateRes) {
    const recordset = new recordset_1.RecordSet();
    const resStr = "evaluateres";
    recordset.metricsRecords[`${resStr}.loss`] = new recordset_1.MetricsRecord({ loss: evaluateRes.loss });
    recordset.metricsRecords[`${resStr}.num_examples`] = new recordset_1.MetricsRecord({
        numExamples: evaluateRes.numExamples,
    });
    recordset.configsRecords[`${resStr}.metrics`] = new recordset_1.ConfigsRecord(evaluateRes.metrics);
    return embedStatusIntoRecordSet(resStr, evaluateRes.status, recordset);
}
function recordSetToGetParametersIns(recordset) {
    const configRecord = recordset.configsRecords["getparametersins.config"];
    const configDict = checkMappingFromRecordScalarTypeToScalar(configRecord);
    return { config: configDict };
}
function recordSetToGetPropertiesIns(recordset) {
    const configRecord = recordset.configsRecords["getpropertiesins.config"];
    const configDict = checkMappingFromRecordScalarTypeToScalar(configRecord);
    return { config: configDict };
}
function getParametersInsToRecordSet(getParametersIns) {
    const recordset = new recordset_1.RecordSet();
    recordset.configsRecords["getparametersins.config"] = new recordset_1.ConfigsRecord(getParametersIns.config);
    return recordset;
}
function getPropertiesInsToRecordSet(getPropertiesIns) {
    try {
        const recordset = new recordset_1.RecordSet();
        let config;
        if (getPropertiesIns && "config" in getPropertiesIns)
            config = getPropertiesIns.config || {};
        else
            config = {};
        recordset.configsRecords["getpropertiesins.config"] = new recordset_1.ConfigsRecord(config);
        return recordset;
    }
    catch (error) {
        console.error("Error in getPropertiesInsToRecordSet:", error);
        throw error; // You can throw or return a default value based on your requirement
    }
}
function getParametersResToRecordSet(getParametersRes, keepInput) {
    const recordset = new recordset_1.RecordSet();
    const parametersRecord = parametersToParametersRecord(getParametersRes.parameters, keepInput);
    recordset.parametersRecords["getparametersres.parameters"] = parametersRecord;
    return embedStatusIntoRecordSet("getparametersres", getParametersRes.status, recordset);
}
function getPropertiesResToRecordSet(getPropertiesRes) {
    const recordset = new recordset_1.RecordSet();
    recordset.configsRecords["getpropertiesres.properties"] = new recordset_1.ConfigsRecord(getPropertiesRes.properties);
    return embedStatusIntoRecordSet("getpropertiesres", getPropertiesRes.status, recordset);
}
function recordSetToGetParametersRes(recordset, keepInput) {
    const resStr = "getparametersres";
    const parameters = parametersrecordToParameters(recordset.parametersRecords[`${resStr}.parameters`], keepInput);
    const status = extractStatusFromRecordSet(resStr, recordset);
    return { status, parameters };
}
function recordSetToGetPropertiesRes(recordset) {
    const resStr = "getpropertiesres";
    const properties = checkMappingFromRecordScalarTypeToScalar(recordset.configsRecords[`${resStr}.properties`]);
    const status = extractStatusFromRecordSet(resStr, recordset);
    return { status, properties };
}
