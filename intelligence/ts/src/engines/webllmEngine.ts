// Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
// =============================================================================

import {
  type ChatCompletionMessageParam,
  CreateMLCEngine,
  type InitProgressReport,
  type MLCEngineInterface,
} from '@mlc-ai/web-llm';

import { getAvailableRAM } from '../env';
import {
  ChatResponseResult,
  FailureCode,
  Message,
  Progress,
  ResponseFormat,
  Result,
  StreamEvent,
  Tool,
  ToolChoice,
} from '../typing';
import { getEngineModelConfig } from './common/model';
import { BaseEngine } from './engine';

async function runQuery(
  engine: MLCEngineInterface,
  messages: Message[],
  stream?: boolean,
  onStreamEvent?: (event: StreamEvent) => void,
  temperature?: number,
  topP?: number,
  maxTokens?: number,
  responseFormat?: ResponseFormat,
  signal?: AbortSignal
) {
  if (signal) {
    signal.addEventListener('abort', () => {
      engine.interruptGenerate();
    });
  }
  if (stream && onStreamEvent) {
    const reply = await engine.chat.completions.create({
      stream: true,
      messages: messages as ChatCompletionMessageParam[],
      temperature,
      top_p: topP,
      max_tokens: maxTokens,
      response_format: {
        type: 'json_object',
        schema: JSON.stringify(responseFormat?.json_schema),
      },
    });
    for await (const chunk of reply) {
      if (signal?.aborted) break;
      onStreamEvent({ chunk: chunk.choices[0]?.delta?.content ?? '' });
    }
    return await engine.getMessage();
  } else {
    const reply = await engine.chat.completions.create({
      messages: messages as ChatCompletionMessageParam[],
      temperature,
      top_p: topP,
      max_tokens: maxTokens,
      response_format: {
        type: 'json_object',
        schema: JSON.stringify(responseFormat?.json_schema),
      },
    });
    return reply.choices[0].message.content ?? '';
  }
}

export class WebllmEngine extends BaseEngine {
  #loadedEngines: Record<string, MLCEngineInterface> = {};

  async chat(
    messages: Message[],
    model: string,
    temperature?: number,
    topP?: number,
    maxCompletionTokens?: number,
    responseFormat?: ResponseFormat,
    stream?: boolean,
    onStreamEvent?: (event: StreamEvent) => void,
    _tools?: Tool[],
    _toolChoice?: ToolChoice,
    _encrypt?: boolean,
    signal?: AbortSignal
  ): Promise<ChatResponseResult> {
    const modelConfigRes = await getEngineModelConfig(model, 'webllm');
    if (!modelConfigRes.ok) {
      return {
        ok: false,
        failure: {
          code: FailureCode.UnsupportedModelError,
          description: `The model ${model} is not supported on the WebLLM engine.`,
        },
      };
    }
    try {
      if (!(model in this.#loadedEngines)) {
        this.#loadedEngines.model = await CreateMLCEngine(modelConfigRes.value.name);
      }
      const result = await runQuery(
        this.#loadedEngines.model,
        messages,
        stream,
        onStreamEvent,
        temperature,
        topP,
        maxCompletionTokens,
        responseFormat,
        signal
      );
      return {
        ok: true,
        message: {
          role: 'assistant',
          content: result,
        },
      };
    } catch (error) {
      return {
        ok: false,
        failure: {
          code: FailureCode.LocalEngineChatError,
          description: `WebLLM engine failed with: ${String(error)}`,
        },
      };
    }
  }

  async fetchModel(model: string, callback: (progress: Progress) => void): Promise<Result<void>> {
    const modelConfigRes = await getEngineModelConfig(model, 'webllm');
    if (!modelConfigRes.ok) {
      return {
        ok: false,
        failure: {
          code: FailureCode.UnsupportedModelError,
          description: `The model ${model} is not supported on the WebLLM engine.`,
        },
      };
    }
    try {
      if (!(model in this.#loadedEngines)) {
        this.#loadedEngines.model = await CreateMLCEngine(modelConfigRes.value.name, {
          initProgressCallback: (report: InitProgressReport) => {
            callback({ percentage: report.progress, description: report.text });
          },
        });
      }
      return { ok: true, value: undefined };
    } catch (error) {
      return {
        ok: false,
        failure: { code: FailureCode.LocalEngineFetchError, description: String(error) },
      };
    }
  }

  async isSupported(model: string): Promise<Result<void>> {
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      const modelConfigRes = await getEngineModelConfig(model, 'webllm');
      if (modelConfigRes.ok) {
        if (modelConfigRes.value.vram) {
          const availableRamRes = await getAvailableRAM();
          if (availableRamRes.ok) {
            if (modelConfigRes.value.vram < availableRamRes.value) {
              return {
                ok: true,
                value: undefined,
              };
            } else {
              return {
                ok: false,
                failure: {
                  code: FailureCode.InsufficientRAMError,
                  description: `Model ${model} requires at least ${String(modelConfigRes.value.vram)} MB to be loaded, but on ${String(availableRamRes.value)} MB are currently available.`,
                },
              };
            }
          }
        }
        return {
          ok: true,
          value: undefined,
        };
      }
      return {
        ok: false,
        failure: {
          code: FailureCode.UnsupportedModelError,
          description: `Model ${model} is unavailable for local inference.`,
        },
      };
    }
    return {
      ok: false,
      failure: {
        code: FailureCode.EngineSpecificError,
        description:
          'A WebGPU compatible browser is required to run inference. More info on https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API#browser_compatibility',
      },
    };
  }
}
