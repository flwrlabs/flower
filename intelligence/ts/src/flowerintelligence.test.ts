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

import { vi } from 'vitest';

vi.mock('./constants', () => ({
  DEFAULT_MODEL: 'meta/llama3.2-1b/instruct-fp16',
  REMOTE_URL: process.env.FI_DEV_REMOTE_URL ?? 'https://example.test',
  VERSION: '0.2.6',
  SDK: 'TS',
  ALLOWED_ROLES: ['system', 'assistant', 'user'],
}));

vi.mock('./engines/common/model', () => ({
  getEngineModelConfig: async () => ({
    ok: true,
    value: { name: 'hf-internal-testing/tiny-random-gpt2', vram: 0 },
  }),
}));

import { describe, expect, it, beforeEach, assert } from 'vitest';
import { FlowerIntelligence } from './flowerintelligence';
import { RemoteEngine } from './engines/remoteEngine';
import { TransformersEngine } from './engines/transformersEngine';
import { FailureCode, ResponseFormat } from './typing';

describe('FlowerIntelligence', () => {
  let fi: FlowerIntelligence;

  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(RemoteEngine.prototype, 'chat').mockImplementation(
      async (_messages, _model, _temperature, _topP, maxCompletionTokens, responseFormat) => {
        const content: string = responseFormat
          ? JSON.stringify({ brand: 'Toyota', model: 'Supra', car_type: 'Coupe' })
          : maxCompletionTokens
            ? 'Short reply'
            : 'Hello from the remote engine.';

        return {
          ok: true,
          message: {
            role: 'assistant',
            content,
          },
        };
      }
    );

    fi = new FlowerIntelligence();
    fi.remoteHandoff = true;
    fi.apiKey = process.env.FI_API_KEY ?? 'test-api-key';
  });

  describe('getEngine', () => {
    it('should return a remote engine when forceRemote is true', async () => {
      const getEngineRes = await fi['getEngine']('meta/llama3.2-1b/instruct-fp16', true, false);
      expect(getEngineRes.ok).toBe(true);
      if (getEngineRes.ok) {
        expect(getEngineRes.value).toBeInstanceOf(RemoteEngine);
      }
    });

    it('should return a local engine when the model can run locally', async () => {
      const getEngineRes = await fi['getEngine'](
        'huggingface/smollm2-360m/instruct-q4',
        false,
        false
      );
      expect(getEngineRes.ok).toBe(true);
      if (getEngineRes.ok) {
        expect(getEngineRes.value).toBeInstanceOf(TransformersEngine);
      }
    });

    it('should throw an error if the model cannot run locally or remotely', async () => {
      fi.remoteHandoff = false;
      const getEngineRes = await fi['getEngine']('meta/llama3.2-1b/instruct-fp16', true, false);
      expect(getEngineRes.ok).toBe(false);
      if (!getEngineRes.ok) {
        expect(Math.floor(getEngineRes.failure.code / 100) * 100).toBe(FailureCode.ConfigError);
      }
    });

    it('should throw an error if forceLocal and forceRemote are both true', async () => {
      const getEngineRes = await fi['getEngine']('meta/llama3.2-1b/instruct-fp16', true, true);
      expect(getEngineRes.ok).toBe(false);
      if (!getEngineRes.ok) {
        expect(getEngineRes.failure.description).toBe(
          'The `forceLocal` and `forceRemote` options cannot be true at the same time.'
        );
      }
    });

    // it('should throw an error if the model ID is invalid', async () => {
    //   const getEngineRes = await fi['getEngine']('invalid-model', false, false);
    //   expect(getEngineRes.ok).toBe(false);
    //   if (!getEngineRes.ok) {
    //     expect(getEngineRes.failure.code).toBe(FailureCode.UnknownModelError);
    //   }
    // });
  });

  describe('chooseLocalEngine', () => {
    it('should return a local engine for a valid provider', async () => {
      const chooseEngineRes = await fi['chooseLocalEngine']('huggingface/smollm2-360m/instruct-q4');
      expect(chooseEngineRes.ok).toBe(true);
      if (chooseEngineRes.ok) {
        expect(chooseEngineRes.value).toBeInstanceOf(TransformersEngine);
      }
    });
  });

  describe('Chat', () => {
    it('generates some text', { timeout: 10_000 }, async () => {
      const data = await fi.chat({
        messages: [
          { role: 'system', content: 'You are a helpful assistant' },
          {
            role: 'user',
            content:
              'Can you write a single sentence email to announce a ground breaking research project?',
          },
        ],
        forceRemote: true,
      });
      expect(data).not.toBeNull();
    });
    it('generates some text from encrypted request', { timeout: 10_000 }, async () => {
      const data = await fi.chat({
        messages: [
          { role: 'system', content: 'You are a helpful assistant' },
          {
            role: 'user',
            content:
              'Can you write a single sentence email to announce a ground breaking research project?',
          },
        ],
        forceRemote: true,
        encrypt: true,
      });
      expect(data).not.toBeNull();
    });
    it('generates some reduced text', { timeout: 10_000 }, async () => {
      const data = await fi.chat({
        messages: [
          { role: 'system', content: 'You are a helpful assistant' },
          {
            role: 'user',
            content: 'Can you write an email to announce a ground breaking research project?',
          },
        ],
        forceRemote: true,
        maxCompletionTokens: 5,
      });
      if (!data.ok) {
        assert.fail(data.failure.description);
      } else {
        expect(data.message.content).not.toBeNull();
        if (data.message.content) {
          expect(data.message.content.trim().split(/\s+/)).length.lessThanOrEqual(5);
        }
      }
    });
    it('generates some text with a specific format', { timeout: 10_000 }, async () => {
      const myFormat: ResponseFormat = {
        type: 'json_schema',
        json_schema: {
          name: 'car-description',
          schema: {
            $defs: {
              CarType: {
                enum: ['sedan', 'SUV', 'Truck', 'Coupe'],
                title: 'CarType',
                type: 'string',
              },
            },
            properties: {
              brand: { title: 'Brand', type: 'string' },
              model: { title: 'Model', type: 'string' },
              car_type: { $ref: '#/$defs/CarType', title: 'Car Type', type: 'string' },
            },
            required: ['brand', 'model', 'car_type'],
            title: 'CarDescription',
            type: 'object',
          },
        },
      };

      const data = await fi.chat({
        model: 'meta/llama3.2-1b/instruct-fp16',
        messages: [
          {
            role: 'user',
            content:
              'Generate a JSON with the brand, model, car_type of the most iconic car from the 90s.',
          },
        ],
        responseFormat: myFormat,
        forceRemote: true,
      });

      if (!data.ok) {
        assert.fail(data.failure.description);
      } else {
        expect(data.message.content).not.toBeNull();
        if (data.message.content) {
          expect(JSON.parse(data.message.content).brand).not.toBeNull();
        }
      }
    });
  });
});
