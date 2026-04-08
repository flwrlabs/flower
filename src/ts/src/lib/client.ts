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
  GetPropertiesIns,
  GetPropertiesRes,
  GetParametersRes,
  GetParametersIns,
  FitIns,
  FitRes,
  EvaluateRes,
  EvaluateIns,
  Context,
  Code,
  Status,
} from "./typing";

abstract class BaseClient {
  protected context: Context;

  constructor(context: Context) {
    this.context = context;
  }

  setContext(context: Context) {
    this.context = context;
  }

  getContext(): Context {
    return this.context;
  }
}

export abstract class Client extends BaseClient {
  abstract getParameters(ins: GetParametersIns): GetParametersRes;
  abstract fit(ins: FitIns): FitRes;
  abstract evaluate(ins: EvaluateIns): EvaluateRes;
  getProperties(_ins: GetPropertiesIns): GetPropertiesRes {
    return {
      status: {
        code: Code.GET_PROPERTIES_NOT_IMPLEMENTED,
        message: "Client does not implement `get_properties`",
      },
      properties: {},
    };
  }
}

function hasGetProperties(client: Client): boolean {
  return client.getProperties !== undefined;
}

function hasGetParameters(client: Client): boolean {
  return client.getParameters !== undefined;
}

function hasFit(client: Client): boolean {
  return client.fit !== undefined;
}

function hasEvaluate(client: Client): boolean {
  return client.evaluate !== undefined;
}

export function maybeCallGetProperties(
  client: Client,
  getPropertiesIns: GetPropertiesIns,
): GetPropertiesRes {
  if (!hasGetProperties(client)) {
    const status: Status = {
      code: Code.GET_PROPERTIES_NOT_IMPLEMENTED,
      message: "Client does not implement `get_properties`",
    };
    return { status, properties: {} };
  }
  return client.getProperties!(getPropertiesIns);
}

export function maybeCallGetParameters(
  client: Client,
  getParametersIns: GetParametersIns,
): GetParametersRes {
  if (!hasGetParameters(client)) {
    const status: Status = {
      code: Code.GET_PARAMETERS_NOT_IMPLEMENTED,
      message: "Client does not implement `get_parameters`",
    };
    return {
      status,
      parameters: { tensorType: "", tensors: [] },
    };
  }
  return client.getParameters!(getParametersIns);
}

export function maybeCallFit(client: Client, fitIns: FitIns): FitRes {
  if (!hasFit(client)) {
    const status: Status = {
      code: Code.FIT_NOT_IMPLEMENTED,
      message: "Client does not implement `fit`",
    };
    return {
      status,
      parameters: { tensorType: "", tensors: [] },
      numExamples: 0,
      metrics: {},
    };
  }
  return client.fit!(fitIns);
}

export function maybeCallEvaluate(client: Client, evaluateIns: EvaluateIns): EvaluateRes {
  if (!hasEvaluate(client)) {
    const status: Status = {
      code: Code.EVALUATE_NOT_IMPLEMENTED,
      message: "Client does not implement `evaluate`",
    };
    return {
      status,
      loss: 0.0,
      numExamples: 0,
      metrics: {},
    };
  }
  return client.evaluate!(evaluateIns);
}
