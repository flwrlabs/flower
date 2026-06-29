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
  ChatResponseResult,
  Failure,
  FailureCode,
  Message,
  ResponseFormat,
  StreamEvent,
  Tool,
  ToolCall,
  ToolChoice,
  Usage,
} from '../../typing';
import { CryptographyHandler } from './cryptoHandler';
import { createChatRequestData, getHeaders, sendRequest } from './remoteUtils';
import {
  ChatCompletionsResponse,
  getServerSentEventData,
  isFinalChunk,
  isGenericError,
  isHTTPError,
  isPlatformHttpError,
  isStreamChunk,
} from './typing';

export async function chatStream(
  baseUrl: string,
  cryptoHandler: CryptographyHandler,
  apiKey: string,
  messages: Message[],
  model: string,
  encrypt: boolean,
  temperature?: number,
  topP?: number,
  maxCompletionTokens?: number,
  responseFormat?: ResponseFormat,
  tools?: Tool[],
  toolChoice?: ToolChoice,
  onStreamEvent?: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<ChatResponseResult> {
  const requestData = createChatRequestData(
    messages,
    model,
    temperature,
    topP,
    maxCompletionTokens,
    responseFormat,
    true,
    tools,
    toolChoice,
    encrypt,
    cryptoHandler.encryptionId
  );
  const response = await sendRequest(
    requestData,
    '/v1/chat/completions',
    baseUrl,
    getHeaders(apiKey),
    signal
  );

  if (!response.ok) return response;
  const body = response.value.body;
  if (!body) {
    return {
      ok: false,
      failure: {
        code: FailureCode.ConnectionError,
        description: 'No response body received.',
      },
    };
  }
  try {
    return await processStream(body, cryptoHandler, encrypt, onStreamEvent, signal);
  } catch (error) {
    return handleStreamError(error);
  }
}

async function processStream(
  body: ReadableStream<Uint8Array>,
  cryptoHandler: CryptographyHandler,
  encrypt: boolean,
  onStreamEvent?: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<ChatResponseResult> {
  const decoder = new TextDecoder('utf-8');
  const reader = body.getReader();
  let accumulated = '';
  let finalTools: ToolCall[] | null = null;
  let usage: Usage | undefined;
  const pendingToolCalls: Record<string, { name: string; buffer: string }> = {};
  let done = false;
  const abortListener = () => void reader.cancel();

  signal?.addEventListener('abort', abortListener);

  try {
    while (!done && !signal?.aborted) {
      const { done: streamDone, value } = await reader.read();
      done = streamDone;

      const text = decoder.decode(value, { stream: true });
      for (const part of splitJsonChunks(text)) {
        const chunkResult = await processChunk(
          part,
          finalTools,
          usage,
          pendingToolCalls,
          cryptoHandler,
          encrypt,
          onStreamEvent
        );
        if (!chunkResult.ok) {
          return chunkResult;
        }
        if (chunkResult.done) {
          done = true;
          usage = chunkResult.usage;
          break;
        }
        if (chunkResult.toolsUpdated && chunkResult.message.toolCalls) {
          finalTools = chunkResult.message.toolCalls;
          usage = chunkResult.usage;
        }
        accumulated += chunkResult.message.content;
      }
    }

    if (finalTools) {
      return {
        ok: true,
        message: {
          role: 'assistant',
          content: '',
          toolCalls: finalTools,
        },
        usage,
      };
    }

    return {
      ok: true,
      message: {
        role: 'assistant',
        content: accumulated,
      },
      usage,
    };
  } finally {
    signal?.removeEventListener('abort', abortListener);
  }
}

function splitJsonChunks(text: string): string[] {
  return text.trim().split(/\n\n+/).filter(Boolean);
}

async function processChunk(
  chunk: string,
  finalTools: ToolCall[] | null,
  usage: Usage | undefined,
  pendingToolCalls: Record<string, { name: string; buffer: string }>,
  cryptoHandler: CryptographyHandler,
  encrypt: boolean,
  onStreamEvent?: (event: StreamEvent) => void
): Promise<ChatResponseResult & { toolsUpdated?: boolean; done?: boolean }> {
  const data = getServerSentEventData(chunk);
  if (data === '[DONE]') {
    return { ok: true, message: { role: 'assistant', content: '' }, usage, done: true };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(data);
  } catch {
    return {
      ok: false,
      failure: { code: FailureCode.RemoteError, description: 'Invalid JSON chunk received.' },
    };
  }

  if (isFinalChunk(parsed)) {
    return {
      ok: true,
      message: {
        role: 'assistant',
        content: '',
      },
      usage: {
        promptTokens: parsed.usage.prompt_tokens,
        completionTokens: parsed.usage.completion_tokens,
        totalTokens: parsed.usage.total_tokens,
      },
      done: true,
    };
  }

  if (isStreamChunk(parsed)) {
    let text = '';
    let toolsUpdated = false;

    for (const choice of parsed.choices) {
      const delta = choice.delta;
      if (delta.tool_calls) {
        finalTools ??= [];
        for (const t of delta.tool_calls) {
          const callId = String(t.index);
          const fn = t.function;
          const name = fn.name ?? '';

          // start the buffer if first fragment
          if (!(callId in pendingToolCalls)) {
            pendingToolCalls[callId] = {
              name,
              buffer: '',
            };
          }
          const fragment = fn.arguments ?? '';
          let buf = pendingToolCalls[callId].buffer;

          // if the model starts a new JSON blob (e.g. it emits "{"…),
          // discard any old buffer and start fresh
          if (fragment.trim().startsWith('{')) {
            buf = fragment;
          } else {
            buf += fragment;
          }
          pendingToolCalls[callId].buffer = buf;

          try {
            const args = JSON.parse(pendingToolCalls[callId].buffer) as Record<string, string>;
            onStreamEvent?.({
              toolCall: {
                index: callId,
                name: pendingToolCalls[callId].name,
                arguments: args,
                complete: true,
              },
            });
            finalTools.push({
              function: {
                name: pendingToolCalls[callId].name,
                arguments: args,
              },
            });
            toolsUpdated = true;
            continue;
          } catch {
            // not complete yet, wait for more chunks
            onStreamEvent?.({
              toolCall: { index: callId, name, arguments: buf, complete: false },
            });
          }
        }
      }
      if (!delta.content) continue;

      let content = delta.content;
      if (encrypt) {
        const decrypted = await cryptoHandler.decryptMessage(delta.content);
        if (!decrypted.ok) {
          throw new StreamProcessingError(decrypted.failure);
        }
        content = decrypted.value;
      }

      onStreamEvent?.({ chunk: content });
      text += content;
    }

    return {
      ok: true,
      message: {
        role: 'assistant',
        content: text,
        ...(finalTools && { toolCalls: finalTools }),
      },
      usage,
      toolsUpdated,
    };
  }

  if (isPlatformHttpError(parsed)) {
    throw new StreamProcessingError({
      code: FailureCode.RemoteError,
      description: parsed.detail.message,
    });
  }

  if (isHTTPError(parsed)) {
    throw new StreamProcessingError({
      code: FailureCode.ConnectionError,
      description: parsed.detail,
    });
  }

  if (isGenericError(parsed)) {
    throw new StreamProcessingError({
      code: FailureCode.RemoteError,
      description: parsed.error,
    });
  }

  return {
    ok: false,
    failure: { code: FailureCode.RemoteError, description: 'Unknown chunk type received.' },
  };
}

function handleStreamError(error: unknown): ChatResponseResult {
  if (error instanceof StreamProcessingError) {
    return { ok: false, failure: error.failure };
  }

  if (
    (error instanceof Error && error.name === 'AbortError') ||
    (error instanceof DOMException && error.name === 'AbortError')
  ) {
    return {
      ok: false,
      failure: {
        code: FailureCode.RequestAborted,
        description: 'Request was aborted by the user.',
      },
    };
  }

  return { ok: false, failure: { code: FailureCode.RemoteError, description: String(error) } };
}

class StreamProcessingError extends Error {
  constructor(public failure: Failure) {
    super();
  }
}

export async function extractChatOutput(
  response: ChatCompletionsResponse,
  encrypt: boolean,
  cryptoHandler: CryptographyHandler
): Promise<ChatResponseResult> {
  const message = response.choices[0].message;
  let content: string;
  if (encrypt) {
    const decryptedResult = await cryptoHandler.decryptMessage(message.content ?? '');
    if (!decryptedResult.ok) {
      return decryptedResult;
    }
    content = decryptedResult.value;
  } else {
    content = message.content ?? '';
  }
  const toolCalls = message.tool_calls;

  return {
    ok: true,
    message: {
      role: message.role as Message['role'],
      content: content,
      ...(toolCalls && { toolCalls: toolCalls }),
    },
    usage: response.usage,
  };
}
