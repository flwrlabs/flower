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
export const sleep = (s: number) => new Promise((r) => setTimeout(r, s * 1000));

export interface RetryState {
  target: (...args: any[]) => any;
  args: any[];
  kwargs: Record<string, any>;
  tries: number;
  elapsedTime: number;
  exception?: Error;
  actualWait?: number;
}

// Generator function for exponential backoff strategy
export function* exponential(
  baseDelay: number = 1,
  multiplier: number = 2,
  maxDelay?: number,
): Generator<number, void, unknown> {
  let delay = maxDelay === undefined ? baseDelay : Math.min(baseDelay, maxDelay);
  while (true) {
    yield delay;
    delay *= multiplier;
    if (maxDelay !== undefined) {
      delay = Math.min(delay, maxDelay);
    }
  }
}

// Generator function for constant wait times
export function* constant(
  interval: number | Iterable<number> = 1,
): Generator<number, void, unknown> {
  if (typeof interval === "number") {
    while (true) {
      yield interval;
    }
  } else {
    yield* interval;
  }
}

// Full jitter algorithm
export function fullJitter(maxValue: number): number {
  return Math.random() * maxValue;
}

export class RetryInvoker {
  private waitGenFactory: () => Generator<number, void, unknown>;
  private recoverableExceptions: any;
  private maxTries: number | null;
  private maxTime: number | null;
  private onSuccess?: (state: RetryState) => void;
  private onBackoff?: (state: RetryState) => void;
  private onGiveup?: (state: RetryState) => void;
  private jitter?: (waitTime: number) => number;
  private shouldGiveup?: (err: Error) => boolean;
  private waitFunction: (waitTime: number) => Promise<unknown>;

  constructor(
    waitGenFactory: () => Generator<number, void, unknown>,
    recoverableExceptions: any,
    maxTries: number | null,
    maxTime: number | null,
    options: {
      onSuccess?: (state: RetryState) => void;
      onBackoff?: (state: RetryState) => void;
      onGiveup?: (state: RetryState) => void;
      jitter?: (waitTime: number) => number;
      shouldGiveup?: (err: Error) => boolean;
      waitFunction?: (waitTime: number) => Promise<unknown>;
    } = {},
  ) {
    this.waitGenFactory = waitGenFactory;
    this.recoverableExceptions = recoverableExceptions;
    this.maxTries = maxTries;
    this.maxTime = maxTime;
    this.onSuccess = options.onSuccess;
    this.onBackoff = options.onBackoff;
    this.onGiveup = options.onGiveup;
    this.jitter = options.jitter ?? fullJitter;
    this.shouldGiveup = options.shouldGiveup;
    this.waitFunction = options.waitFunction ?? sleep;
  }

  public async invoke(target: (...args: any[]) => any, ...args: any[]): Promise<any> {
    const startTime = Date.now();
    let tryCount = 0;
    const waitGenerator = this.waitGenFactory();

    while (true) {
      tryCount++;
      const elapsedTime = (Date.now() - startTime) / 1000;
      const state: RetryState = {
        target,
        args,
        kwargs: {},
        tries: tryCount,
        elapsedTime,
      };

      try {
        // Attempt the target function call
        const result = await target(...args);

        // On success, call onSuccess handler if defined
        if (this.onSuccess) {
          this.onSuccess(state);
        }

        return result;
      } catch (err) {
        if (!(err instanceof this.recoverableExceptions)) {
          throw err; // Not a recoverable exception, rethrow it
        }

        state.exception = err as Error;

        const giveup = this.shouldGiveup && this.shouldGiveup(err as Error);
        const maxTriesExceeded = this.maxTries !== null && tryCount >= this.maxTries;
        const maxTimeExceeded = this.maxTime !== null && elapsedTime >= this.maxTime;

        // Check if we should give up
        if (giveup || maxTriesExceeded || maxTimeExceeded) {
          if (this.onGiveup) {
            this.onGiveup(state);
          }
          throw err; // Give up and rethrow the error
        }

        let waitTime = waitGenerator.next().value as number;
        if (this.jitter) {
          waitTime = this.jitter(waitTime);
        }
        if (this.maxTime !== null) {
          waitTime = Math.min(waitTime, this.maxTime - elapsedTime);
        }
        state.actualWait = waitTime;

        // Call onBackoff handler if defined
        if (this.onBackoff) {
          this.onBackoff(state);
        }

        // Wait for the specified time
        await this.waitFunction(waitTime * 1000);
      }
    }
  }
}
