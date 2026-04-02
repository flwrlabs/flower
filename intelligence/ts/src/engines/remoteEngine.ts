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

import { REMOTE_URL } from '../constants';
import {
  ChatResponseResult,
  Embedding,
  EmbeddingInput,
  FailureCode,
  Message,
  Progress,
  ResponseFormat,
  Result,
  StreamEvent,
  Tool,
  ToolChoice,
} from '../typing';
import { BaseEngine } from './engine';
import { chatStream, extractChatOutput } from './remoteEngine/chat';
import { CryptographyHandler } from './remoteEngine/cryptoHandler';
import { extractEmbedOutput } from './remoteEngine/embed';
import {
  createChatRequestData,
  createEmbedRequestData,
  getHeaders,
  sendRequest,
} from './remoteEngine/remoteUtils';
import { ChatCompletionsResponse, EmbedResponse } from './remoteEngine/typing';

export class RemoteEngine extends BaseEngine {
  private baseUrl: string;
  private apiKey: string;
  private cryptoHandler: CryptographyHandler;

  constructor(apiKey: string) {
    super();
    this.baseUrl = REMOTE_URL;
    this.apiKey = apiKey;
    this.cryptoHandler = new CryptographyHandler(this.baseUrl, this.apiKey);
  }

  async chat(
    messages: Message[],
    model: string,
    temperature?: number,
    topP?: number,
    maxCompletionTokens?: number,
    responseFormat?: ResponseFormat,
    stream?: boolean,
    onStreamEvent?: (event: StreamEvent) => void,
    tools?: Tool[],
    toolChoice?: ToolChoice,
    encrypt = false,
    signal?: AbortSignal
  ): Promise<ChatResponseResult> {
    if (encrypt) {
      const keyRes = await this.cryptoHandler.initializeKeysAndExchange();
      if (!keyRes.ok) {
        return keyRes;
      }
      const encryptRes = await this.cryptoHandler.encryptMessages(messages);
      if (!encryptRes.ok) {
        return encryptRes;
      }
    }
    if (stream) {
      const response = await chatStream(
        this.baseUrl,
        this.cryptoHandler,
        this.apiKey,
        messages,
        model,
        encrypt,
        temperature,
        topP,
        maxCompletionTokens,
        responseFormat,
        tools,
        toolChoice,
        onStreamEvent,
        signal
      );
      return response;
    } else {
      const requestData = createChatRequestData(
        messages,
        model,
        temperature,
        topP,
        maxCompletionTokens,
        responseFormat,
        false,
        tools,
        toolChoice,
        encrypt,
        this.cryptoHandler.encryptionId
      );
      const response = await sendRequest(
        requestData,
        '/v1/chat/completions',
        this.baseUrl,
        getHeaders(this.apiKey),
        signal
      );
      if (!response.ok) {
        return response;
      }
      const chatResponse = (await response.value.json()) as ChatCompletionsResponse;
      return await extractChatOutput(chatResponse, encrypt, this.cryptoHandler);
    }
  }

  async embed(model: string, input: EmbeddingInput): Promise<Result<Embedding[]>> {
    const requestData = createEmbedRequestData(input, model);
    const response = await sendRequest(
      requestData,
      '/v1/embeddings',
      this.baseUrl,
      getHeaders(this.apiKey)
    );
    if (!response.ok) {
      return response;
    }
    const embedResponse = (await response.value.json()) as EmbedResponse;
    return extractEmbedOutput(embedResponse);
  }

  async fetchModel(_model: string, _callback: (progress: Progress) => void): Promise<Result<void>> {
    await Promise.resolve();
    return {
      ok: false,
      failure: {
        code: FailureCode.EngineSpecificError,
        description: 'Cannot fetch model with remote inference engine.',
      },
    };
  }

  async isSupported(_model: string): Promise<Result<void>> {
    await Promise.resolve();
    return {
      ok: true,
      value: undefined,
    };
  }
}
