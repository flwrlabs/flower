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
  GetParametersIns,
  GetParametersRes,
  GetPropertiesIns,
  GetPropertiesRes,
  EvaluateIns,
  EvaluateRes,
  Metrics,
} from "./typing";
import {
  ParametersRecord,
  RecordSet,
  ArrayData,
  ConfigsRecordValue,
  ConfigsRecord,
  MetricsRecord,
  MetricsRecordValue,
} from "./recordset";

function parametersrecordToParameters(record: ParametersRecord, keepInput: boolean): Parameters {
  const parameters: Parameters = { tensors: [], tensorType: "" };

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

function parametersToParametersRecord(
  parameters: Parameters,
  keepInput: boolean,
): ParametersRecord {
  const tensorType = parameters.tensorType;
  const orderedDict: ParametersRecord = new ParametersRecord({});

  parameters.tensors.forEach((tensor, idx) => {
    const array = new ArrayData("", [], tensorType, tensor);
    orderedDict[String(idx)] = array;

    if (!keepInput) {
      parameters.tensors.shift();
    }
  });

  if (parameters.tensors.length === 0) {
    orderedDict["EMPTY_TENSOR_KEY"] = new ArrayData("", [], tensorType, new Uint8Array());
  }

  return orderedDict;
}

function checkMappingFromRecordScalarTypeToScalar(
  recordData: Record<string, ConfigsRecordValue>,
): Record<string, Scalar> {
  if (!recordData) {
    throw new TypeError("Invalid input: recordData is undefined or null");
  }
  Object.values(recordData).forEach((value) => {
    if (
      typeof value !== "number" &&
      typeof value !== "string" &&
      typeof value !== "boolean" &&
      !(value instanceof Uint8Array)
    ) {
      throw new TypeError(`Invalid scalar type found: ${typeof value}`);
    }
  });

  return recordData as Record<string, Scalar>;
}

function recordSetToFitOrEvaluateInsComponents(
  recordset: RecordSet,
  insStr: string,
  keepInput: boolean,
): { parameters: Parameters; config: Record<string, Scalar> } {
  const parametersRecord = recordset.parametersRecords[`${insStr}.parameters`];
  const parameters = parametersrecordToParameters(parametersRecord, keepInput);

  const configRecord = recordset.configsRecords[`${insStr}.config`];
  const configDict = checkMappingFromRecordScalarTypeToScalar(configRecord);

  return { parameters, config: configDict };
}

function fitOrEvaluateInsToRecordSet(
  ins: { parameters: Parameters; config: Record<string, Scalar> },
  keepInput: boolean,
  insStr: string,
): RecordSet {
  const recordset = new RecordSet();

  const parametersRecord = parametersToParametersRecord(ins.parameters, keepInput);
  recordset.parametersRecords[`${insStr}.parameters`] = parametersRecord;

  recordset.configsRecords[`${insStr}.config`] = new ConfigsRecord(
    ins.config as Record<string, ConfigsRecordValue>,
  );

  return recordset;
}

function embedStatusIntoRecordSet(
  resStr: string,
  status: { code: number; message: string },
  recordset: RecordSet,
): RecordSet {
  const statusDict: Record<string, Scalar> = {
    code: status.code,
    message: status.message,
  };

  recordset.configsRecords[`${resStr}.status`] = new ConfigsRecord(
    statusDict as Record<string, ConfigsRecordValue>,
  );

  return recordset;
}

function extractStatusFromRecordSet(
  resStr: string,
  recordset: RecordSet,
): { code: number; message: string } {
  const status = recordset.configsRecords[`${resStr}.status`];
  const code = status["code"] as number;
  return { code, message: status["message"] as string };
}

export function recordSetToFitIns(recordset: RecordSet, keepInput: boolean): FitIns {
  const { parameters, config } = recordSetToFitOrEvaluateInsComponents(
    recordset,
    "fitins",
    keepInput,
  );
  return { parameters, config };
}

export function fitInsToRecordSet(fitins: FitIns, keepInput: boolean): RecordSet {
  return fitOrEvaluateInsToRecordSet(fitins, keepInput, "fitins");
}

export function recordSetToFitRes(recordset: RecordSet, keepInput: boolean): FitRes {
  const insStr = "fitres";
  const parameters = parametersrecordToParameters(
    recordset.parametersRecords[`${insStr}.parameters`],
    keepInput,
  );

  const numExamples = recordset.metricsRecords[`${insStr}.num_examples`]["num_examples"] as number;

  const configRecord = recordset.configsRecords[`${insStr}.metrics`];
  const metrics = checkMappingFromRecordScalarTypeToScalar(configRecord);
  const status = extractStatusFromRecordSet(insStr, recordset);

  return { status, parameters, numExamples, metrics };
}

export function fitResToRecordSet(fitres: FitRes, keepInput: boolean): RecordSet {
  const recordset = new RecordSet();
  const resStr = "fitres";

  recordset.configsRecords[`${resStr}.metrics`] = new ConfigsRecord(
    fitres.metrics as Record<string, ConfigsRecordValue>,
  );
  recordset.metricsRecords[`${resStr}.num_examples`] = new MetricsRecord({
    num_examples: fitres.numExamples as MetricsRecordValue,
  });

  recordset.parametersRecords[`${resStr}.parameters`] = parametersToParametersRecord(
    fitres.parameters,
    keepInput,
  );

  return embedStatusIntoRecordSet(resStr, fitres.status, recordset);
}

export function recordSetToEvaluateIns(recordset: RecordSet, keepInput: boolean): EvaluateIns {
  const { parameters, config } = recordSetToFitOrEvaluateInsComponents(
    recordset,
    "evaluateins",
    keepInput,
  );
  return { parameters, config };
}

export function evaluateInsToRecordSet(evaluateIns: EvaluateIns, keepInput: boolean): RecordSet {
  return fitOrEvaluateInsToRecordSet(evaluateIns, keepInput, "evaluateins");
}

export function recordSetToEvaluateRes(recordset: RecordSet): EvaluateRes {
  const insStr = "evaluateres";

  const loss = recordset.metricsRecords[`${insStr}.loss`]["loss"] as number;
  const numExamples = recordset.metricsRecords[`${insStr}.num_examples`]["numExamples"] as number;
  const configsRecord = recordset.configsRecords[`${insStr}.metrics`];
  const metrics = Object.fromEntries(
    Object.entries(configsRecord).map(([key, value]) => [key, value]),
  ) as Metrics;
  const status = extractStatusFromRecordSet(insStr, recordset);

  return { status, loss, numExamples, metrics };
}

export function evaluateResToRecordSet(evaluateRes: EvaluateRes): RecordSet {
  const recordset = new RecordSet();
  const resStr = "evaluateres";

  recordset.metricsRecords[`${resStr}.loss`] = new MetricsRecord({ loss: evaluateRes.loss });
  recordset.metricsRecords[`${resStr}.num_examples`] = new MetricsRecord({
    numExamples: evaluateRes.numExamples,
  });
  recordset.configsRecords[`${resStr}.metrics`] = new ConfigsRecord(
    evaluateRes.metrics as Record<string, ConfigsRecordValue>,
  );

  return embedStatusIntoRecordSet(resStr, evaluateRes.status, recordset);
}

export function recordSetToGetParametersIns(recordset: RecordSet): GetParametersIns {
  const configRecord = recordset.configsRecords["getparametersins.config"];
  const configDict = checkMappingFromRecordScalarTypeToScalar(configRecord);
  return { config: configDict };
}

export function recordSetToGetPropertiesIns(recordset: RecordSet): GetPropertiesIns {
  const configRecord = recordset.configsRecords["getpropertiesins.config"];
  const configDict = checkMappingFromRecordScalarTypeToScalar(configRecord);
  return { config: configDict };
}

export function getParametersInsToRecordSet(getParametersIns: GetParametersIns): RecordSet {
  const recordset = new RecordSet();
  recordset.configsRecords["getparametersins.config"] = new ConfigsRecord(
    getParametersIns.config as Record<string, ConfigsRecordValue>,
  );
  return recordset;
}

export function getPropertiesInsToRecordSet(getPropertiesIns: GetPropertiesIns | null): RecordSet {
  try {
    const recordset = new RecordSet();

    let config: Record<string, ConfigsRecordValue>;
    if (getPropertiesIns && "config" in getPropertiesIns)
      config = (getPropertiesIns.config as Record<string, ConfigsRecordValue>) || {};
    else config = {};

    recordset.configsRecords["getpropertiesins.config"] = new ConfigsRecord(config);
    return recordset;
  } catch (error) {
    console.error("Error in getPropertiesInsToRecordSet:", error);
    throw error; // You can throw or return a default value based on your requirement
  }
}

export function getParametersResToRecordSet(
  getParametersRes: GetParametersRes,
  keepInput: boolean,
): RecordSet {
  const recordset = new RecordSet();
  const parametersRecord = parametersToParametersRecord(getParametersRes.parameters, keepInput);
  recordset.parametersRecords["getparametersres.parameters"] = parametersRecord;

  return embedStatusIntoRecordSet("getparametersres", getParametersRes.status, recordset);
}

export function getPropertiesResToRecordSet(getPropertiesRes: GetPropertiesRes): RecordSet {
  const recordset = new RecordSet();
  recordset.configsRecords["getpropertiesres.properties"] = new ConfigsRecord(
    getPropertiesRes.properties as Record<string, ConfigsRecordValue>,
  );

  return embedStatusIntoRecordSet("getpropertiesres", getPropertiesRes.status, recordset);
}

export function recordSetToGetParametersRes(
  recordset: RecordSet,
  keepInput: boolean,
): GetParametersRes {
  const resStr = "getparametersres";
  const parameters = parametersrecordToParameters(
    recordset.parametersRecords[`${resStr}.parameters`],
    keepInput,
  );

  const status = extractStatusFromRecordSet(resStr, recordset);
  return { status, parameters };
}

export function recordSetToGetPropertiesRes(recordset: RecordSet): GetPropertiesRes {
  const resStr = "getpropertiesres";
  const properties = checkMappingFromRecordScalarTypeToScalar(
    recordset.configsRecords[`${resStr}.properties`],
  );

  const status = extractStatusFromRecordSet(resStr, recordset);
  return { status, properties };
}
