"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RetryInvoker = exports.sleep = void 0;
exports.exponential = exponential;
exports.constant = constant;
exports.fullJitter = fullJitter;
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
const sleep = (s) => new Promise((r) => setTimeout(r, s * 1000));
exports.sleep = sleep;
// Generator function for exponential backoff strategy
function* exponential(baseDelay = 1, multiplier = 2, maxDelay) {
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
function* constant(interval = 1) {
    if (typeof interval === "number") {
        while (true) {
            yield interval;
        }
    }
    else {
        yield* interval;
    }
}
// Full jitter algorithm
function fullJitter(maxValue) {
    return Math.random() * maxValue;
}
class RetryInvoker {
    waitGenFactory;
    recoverableExceptions;
    maxTries;
    maxTime;
    onSuccess;
    onBackoff;
    onGiveup;
    jitter;
    shouldGiveup;
    waitFunction;
    constructor(waitGenFactory, recoverableExceptions, maxTries, maxTime, options = {}) {
        this.waitGenFactory = waitGenFactory;
        this.recoverableExceptions = recoverableExceptions;
        this.maxTries = maxTries;
        this.maxTime = maxTime;
        this.onSuccess = options.onSuccess;
        this.onBackoff = options.onBackoff;
        this.onGiveup = options.onGiveup;
        this.jitter = options.jitter ?? fullJitter;
        this.shouldGiveup = options.shouldGiveup;
        this.waitFunction = options.waitFunction ?? exports.sleep;
    }
    async invoke(target, ...args) {
        const startTime = Date.now();
        let tryCount = 0;
        const waitGenerator = this.waitGenFactory();
        while (true) {
            tryCount++;
            const elapsedTime = (Date.now() - startTime) / 1000;
            const state = {
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
            }
            catch (err) {
                if (!(err instanceof this.recoverableExceptions)) {
                    throw err; // Not a recoverable exception, rethrow it
                }
                state.exception = err;
                const giveup = this.shouldGiveup && this.shouldGiveup(err);
                const maxTriesExceeded = this.maxTries !== null && tryCount >= this.maxTries;
                const maxTimeExceeded = this.maxTime !== null && elapsedTime >= this.maxTime;
                // Check if we should give up
                if (giveup || maxTriesExceeded || maxTimeExceeded) {
                    if (this.onGiveup) {
                        this.onGiveup(state);
                    }
                    throw err; // Give up and rethrow the error
                }
                let waitTime = waitGenerator.next().value;
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
exports.RetryInvoker = RetryInvoker;
