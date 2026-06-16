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

import type { ProgressInfo, TextGenerationConfig } from '@huggingface/transformers';
import {
  InterruptableStoppingCriteria,
  pipeline,
  StoppingCriteriaList,
  Tensor,
  TextGenerationPipeline,
  TextStreamer,
} from '@huggingface/transformers';

import { getAvailableRAM } from '../env';
import {
  ChatResponseResult,
  FailureCode,
  Message,
  Progress,
  ResponseFormat,
  Result,
} from '../typing';
import { getEngineModelConfig } from './common/model';
import { BaseEngine } from './engine';

const stoppingCriteria = new InterruptableStoppingCriteria();
const choice = 0;
const textGenerationPipeline = pipeline as (
  task: 'text-generation',
  model: string,
  options?: { dtype?: DTYPE }
) => Promise<TextGenerationPipeline>;

export class TransformersEngine extends BaseEngine {
  private generationPipelines: Record<string, TextGenerationPipeline> = {};

  async chat(
    messages: Message[],
    model: string,
    temperature?: number,
    topP?: number,
    maxCompletionTokens?: number,
    _responseFormat?: ResponseFormat,
    stream?: boolean,
    onStreamEvent?: (event: { chunk: string }) => void
  ): Promise<ChatResponseResult> {
    const modelConfigRes = await getEngineModelConfig(model, 'onnx');
    if (!modelConfigRes.ok) {
      return {
        ok: false,
        failure: {
          code: FailureCode.UnsupportedModelError,
          description: `The model ${model} is not supported on the Transformers.js engine.`,
        },
      };
    }
    try {
      if (!(model in this.generationPipelines)) {
        const modelElems = modelConfigRes.value.name.split('|');
        const modelId = modelElems[0];

        const pipelineOptions: { dtype?: DTYPE } = {};
        if (modelElems.length > 1) {
          pipelineOptions.dtype = modelElems[1] as DTYPE;
        }
        this.generationPipelines.model = await textGenerationPipeline(
          'text-generation',
          modelId,
          pipelineOptions
        );
      }
      const tokenizer = this.generationPipelines.model.tokenizer;
      const modelInstance = this.generationPipelines.model.model;

      const inputs = tokenizer.apply_chat_template(messages, {
        add_generation_prompt: true,
        return_dict: true,
      }) as {
        input_ids: Tensor | number[] | number[][];
        attention_mask: Tensor | number[] | number[][];
        token_type_ids?: Tensor | number[] | number[][] | undefined;
      };

      let streamer = undefined;
      if (stream && onStreamEvent) {
        streamer = new TextStreamer(tokenizer, {
          skip_prompt: true,
          callback_function: (output: string) => {
            let formattedOutput = output;
            for (const str of tokenizer.special_tokens as string[]) {
              formattedOutput = formattedOutput.replace(str, '');
            }
            onStreamEvent({ chunk: formattedOutput });
          },
        });
      }

      stoppingCriteria.reset();
      const stoppingCriteriaList = new StoppingCriteriaList();
      stoppingCriteriaList.push(stoppingCriteria);
      const { past_key_values: _, sequences } = (await modelInstance.generate({
        ...inputs,
        generation_config: {
          do_sample: false,
          max_new_tokens: maxCompletionTokens ?? 1024,
          temperature: temperature ?? 1,
          return_dict_in_generate: true,
          top_p: topP ?? 1,
        } as TextGenerationConfig,
        stopping_criteria: stoppingCriteriaList,
        ...(streamer && { streamer }),
      })) as { past_key_values: object; sequences: Tensor };

      const decoded = tokenizer.batch_decode(sequences, {
        skip_special_tokens: true,
      });

      let promptLengths: number[] | undefined;
      const inputIds = inputs.input_ids as Tensor;
      const inputDim = inputIds.dims[inputIds.dims.length - 1];
      if (typeof inputDim === 'number' && inputDim > 0) {
        promptLengths = tokenizer
          .batch_decode(inputIds, { skip_special_tokens: true })
          .map((x) => x.length);
      }

      if (promptLengths) {
        for (let i = 0; i < decoded.length; ++i) {
          decoded[i] = decoded[i].slice(promptLengths[i]);
        }
      }

      return {
        ok: true,
        message: {
          role: 'assistant',
          content: decoded[choice],
        },
      };
    } catch (error) {
      return {
        ok: false,
        failure: {
          code: FailureCode.LocalEngineChatError,
          description: `Transformers.js engine failed with: ${String(error)}`,
        },
      };
    }
  }

  async fetchModel(model: string, callback: (progress: Progress) => void): Promise<Result<void>> {
    const modelConfigRes = await getEngineModelConfig(model, 'onnx');
    if (!modelConfigRes.ok) {
      return {
        ok: false,
        failure: {
          code: FailureCode.UnsupportedModelError,
          description: `The model ${model} is not supported on the Transformers.js engine.`,
        },
      };
    }
    try {
      if (!(model in this.generationPipelines)) {
        const modelElems = modelConfigRes.value.name.split('|');
        const modelId = modelElems[0];

        const pipelineOptions: {
          dtype?: DTYPE;
          progress_callback: (progressInfo: ProgressInfo) => void;
        } = {
          progress_callback: (progressInfo: ProgressInfo) => {
            let percentage = 0;
            let total = 0;
            let loaded = 0;
            let description = progressInfo.status as string;
            if (progressInfo.status == 'progress') {
              percentage = progressInfo.progress;
              total = progressInfo.total;
              loaded = progressInfo.loaded;
              description = progressInfo.file;
            } else if (progressInfo.status === 'done') {
              percentage = 100;
              description = progressInfo.status;
            }
            callback({
              totalBytes: total,
              loadedBytes: loaded,
              percentage,
              description,
            });
          },
        };
        if (modelElems.length > 1) {
          pipelineOptions.dtype = modelElems[1] as DTYPE;
        }
        this.generationPipelines.model = await textGenerationPipeline(
          'text-generation',
          modelId,
          pipelineOptions
        );
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
    const modelConfigRes = await getEngineModelConfig(model, 'onnx');
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
}

type DTYPE =
  | 'auto'
  | 'fp32'
  | 'fp16'
  | 'q8'
  | 'int8'
  | 'uint8'
  | 'q4'
  | 'bnb4'
  | 'q4f16'
  | Record<string, 'auto' | 'fp32' | 'fp16' | 'q8' | 'int8' | 'uint8' | 'q4' | 'bnb4' | 'q4f16'>;
