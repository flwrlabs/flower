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

import { ALLOWED_ROLES, DEFAULT_MODEL } from './constants';
import { Engine } from './engines/engine';
import { RemoteEngine } from './engines/remoteEngine';
import { isNode } from './env';
import {
  ChatOptions,
  ChatResponseResult,
  Embedding,
  EmbeddingInput,
  Failure,
  FailureCode,
  Message,
  Progress,
  Result,
} from './typing';

/**
 * Class representing the core intelligence service for Flower Labs.
 * It facilitates chat, generation, and summarization tasks, with the option of using a
 * local or remote engine based on configurations and availability.
 */
export class FlowerIntelligence {
  static #instance: FlowerIntelligence | null = null;
  static #remoteHandoff = false;
  static #apiKey?: string;

  #remoteEngine?: RemoteEngine;
  #localEngineLoaders: (() => Promise<Engine>)[] = isNode
    ? [
        async () => {
          const { TransformersEngine } = await import('./engines/transformersEngine');
          return new TransformersEngine();
        },
      ]
    : [
        async () => {
          const { WebllmEngine } = await import('./engines/webllmEngine');
          return new WebllmEngine();
        },
      ];

  /**
   * Get the initialized FlowerIntelligence instance.
   * Initializes the instance if it doesn't exist.
   * @returns The initialized FlowerIntelligence instance.
   */
  public static get instance(): FlowerIntelligence {
    if (!this.#instance) {
      this.#instance = new FlowerIntelligence();
    }
    return this.#instance;
  }
  /**
   * Sets the remote handoff boolean.
   * @param remoteHandoffValue - If true, the processing might be done on a secure
   remote server instead of locally (if resources are lacking).
   */
  public set remoteHandoff(remoteHandoffValue: boolean) {
    FlowerIntelligence.#remoteHandoff = remoteHandoffValue;
  }

  /**
   * Gets the current remote handoff status.
   * @returns boolean - the value of the remote handoff variable
   */
  public get remoteHandoff() {
    return FlowerIntelligence.#remoteHandoff;
  }

  /**
   * Set apiKey for FlowerIntelligence.
   */
  public set apiKey(apiKey: string) {
    FlowerIntelligence.#apiKey = apiKey;
  }

  /**
   * Downloads and loads a model into memory.
   * @param model Model name of the model to download.
   * @param callback A callback function taking a {@link Progress} object to handle the loading event.
   * @returns A {@link Result} containing either a {@link Failure} (containing `code: number` and `description: string`) if `ok` is false or a value of `void`, if `ok` is true (meaning the loading was successful).
   */
  async fetchModel(model: string, callback: (progress: Progress) => void): Promise<Result<void>> {
    const engineResult = await this.getEngine(model, false, false);
    if (!engineResult.ok) {
      return engineResult;
    } else {
      return await engineResult.value.fetchModel(model, callback);
    }
  }

  /**
   * Creates an embedding vector representing the input text.
   * @param model Model name to use for the chat.
   * @param input, text to embed, encoded as a string or array of tokens. To embed multiple inputs in a single request, pass an array of strings or array of token arrays. The input must not exceed the max input tokens for the model (8192 tokens for all embedding models), cannot be an empty string, and any array must be 2048 dimensions or less.
   * @returns A {@link Result} containing either a {@link Failure} (containing `code: number` and `description: string`) if `ok` is false or, if `ok` is true (meaning the loading was successful), a value which is a list of embedding vectors, which are lists of floats. The length of vector depends on the model.
   */
  async embed(options: { model: string; input: EmbeddingInput }): Promise<Result<Embedding[]>> {
    const engineResult = this.getOrCreateRemoteEngine();
    if (!engineResult.ok) {
      return engineResult;
    } else {
      return await engineResult.value.embed(options.model, options.input);
    }
  }

  // Overload for string input with an optional options object
  async chat(input: string, options?: ChatOptions): Promise<ChatResponseResult>;

  // Overload for a single object that includes messages along with other options
  async chat(options: ChatOptions & { messages: Message[] }): Promise<ChatResponseResult>;

  /**
   * Conducts a chat interaction using the specified model and options.
   *
   * This method can be invoked in one of two ways:
   *
   * 1. With a string input (plus an optional options object). In this case the string
   *    is automatically wrapped as a single message with role 'user'.
   *
   *    Example:
   *    ```ts
   *    fi.chat("Why is the sky blue?", { temperature: 0.7 });
   *    ```
   *
   * 2. With a single object that includes a {@link Message} array along with additional options.
   *
   *    Example:
   *    ```ts
   *    fi.chat({
   *      messages: [{ role: 'user', content: "Why is the sky blue?" }],
   *      model: "meta/llama3.2-1b"
   *    });
   *    ```
   *
   * @param inputOrOptions - Either a string input or a {@link ChatOptions} object that must include a `messages` array.
   * @param maybeOptions - An optional {@link ChatOptions} object (used only when the first parameter is a string).
   * @returns A Promise that resolves to a {@link ChatResponseResult}. On success, the result contains the
   *          message reply and optionally any tool call details; on failure, it includes an error code and description.
   */
  async chat(
    inputOrOptions: string | (ChatOptions & { messages: Message[] }),
    maybeOptions?: ChatOptions
  ): Promise<ChatResponseResult> {
    const chatResult = await this.internalChat(inputOrOptions, maybeOptions);
    if (!chatResult.ok) {
      if (
        chatResult.failure.code === FailureCode.LocalEngineChatError &&
        this.remoteHandoff &&
        this.apiKey
      ) {
        return await this.internalChat(inputOrOptions, { ...maybeOptions, forceRemote: true });
      }
    }
    return chatResult;
  }

  private async internalChat(
    inputOrOptions: string | (ChatOptions & { messages: Message[] }),
    maybeOptions?: ChatOptions
  ): Promise<ChatResponseResult> {
    let options: ChatOptions;
    let messages: Message[];

    if (typeof inputOrOptions === 'string') {
      options = maybeOptions ?? {};
      messages = [{ role: 'user', content: inputOrOptions }];
    } else {
      ({ messages, ...options } = inputOrOptions);

      if (messages.some((msg) => !ALLOWED_ROLES.includes(msg.role))) {
        return {
          ok: false,
          failure: {
            code: FailureCode.InvalidArgumentsError,
            description: `Invalid message role${messages.length > 1 ? 's' : ''}: ${messages
              .filter((msg) => !ALLOWED_ROLES.includes(msg.role))
              .map((msg) => msg.role)
              .join(', ')}`,
          },
        };
      }
    }

    const model = options.model ?? DEFAULT_MODEL;
    const engineResult = await this.getEngine(
      model,
      options.forceRemote ?? false,
      options.forceLocal ?? false
    );

    if (!engineResult.ok) {
      return engineResult;
    }

    return await engineResult.value.chat(
      messages,
      model,
      options.temperature,
      options.topP,
      options.maxCompletionTokens,
      options.responseFormat,
      options.stream,
      options.onStreamEvent,
      options.tools,
      options.toolChoice,
      options.encrypt,
      options.signal
    );
  }

  private async getEngine(
    modelId: string,
    forceRemote: boolean,
    forceLocal: boolean
  ): Promise<Result<Engine>> {
    const argsResult = this.validateArgs(forceRemote, forceLocal);
    if (!argsResult.ok) {
      return argsResult;
    }

    if (forceRemote) {
      return this.getOrCreateRemoteEngine();
    }

    const localEngineResult = await this.chooseLocalEngine(modelId);
    if (localEngineResult.ok) {
      return localEngineResult;
    }

    return this.getOrCreateRemoteEngine(localEngineResult);
  }

  private async chooseLocalEngine(modelId: string): Promise<Result<Engine>> {
    const results = await Promise.all(
      this.#localEngineLoaders.map(async (load) => {
        const engine = await load();
        const supportResult = await engine.isSupported(modelId);
        return { engine, supportResult };
      })
    );
    const compatibleEngines = results
      .filter(({ supportResult }) => supportResult.ok)
      .map(({ engine }) => engine);

    if (compatibleEngines.length > 0) {
      // Currently we just select the first compatible localEngine without further check
      return { ok: true, value: compatibleEngines[0] };
    } else {
      // We extract the failures from the results that didn't return a true `ok`
      const failures = results
        .filter(
          (result): result is { engine: Engine; supportResult: { ok: false; failure: Failure } } =>
            !result.supportResult.ok
        )
        .map(({ supportResult }) => supportResult.failure);

      // We then compute which failure has the highest FailureCode
      // usually corresponding to the most specific error
      const highestFailure = failures.reduce(
        (max, failure) => (failure.code > max.code ? failure : max),
        failures[0]
      );
      return {
        ok: false,
        failure: highestFailure,
      };
    }
  }

  private getOrCreateRemoteEngine(localFailure?: Result<Engine>): Result<Engine> {
    if (
      localFailure &&
      !localFailure.ok &&
      (!FlowerIntelligence.#remoteHandoff || !FlowerIntelligence.#apiKey)
    ) {
      let description = localFailure.failure.description;
      description += FlowerIntelligence.#remoteHandoff
        ? '\nAdditionally, a valid API key for Remote Handoff was not provided.'
        : '\nAdditionally, Remote Handoff was not enabled.';
      return {
        ok: false,
        failure: {
          code: localFailure.failure.code,
          description,
        },
      };
    }
    if (!FlowerIntelligence.#remoteHandoff) {
      return {
        ok: false,
        failure: {
          description: 'To use remote inference, remote handoff must be allowed.',
          code: FailureCode.InvalidRemoteConfigError,
        },
      };
    }
    if (!FlowerIntelligence.#apiKey) {
      return {
        ok: false,
        failure: {
          description: 'To use remote inference, a valid API key must be set.',
          code: FailureCode.InvalidRemoteConfigError,
        },
      };
    }
    this.#remoteEngine = this.#remoteEngine ?? new RemoteEngine(FlowerIntelligence.#apiKey);
    return { ok: true, value: this.#remoteEngine };
  }

  private validateArgs(forceRemote: boolean, forceLocal: boolean): Result<void> {
    if (forceLocal && forceRemote) {
      return {
        ok: false,
        failure: {
          description:
            'The `forceLocal` and `forceRemote` options cannot be true at the same time.',
          code: FailureCode.InvalidArgumentsError,
        },
      };
    }
    return { ok: true, value: undefined };
  }
}
