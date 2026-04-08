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
import { EventEmitter } from "events";
import { ServiceError, status } from "@grpc/grpc-js";
import { RetryInvoker, RetryState, exponential } from "./retry_invoker";
import { PING_CALL_TIMEOUT } from "./constants";


class StopEvent extends EventEmitter {
  public is_set: boolean;

  constructor() {
    super();
    this.is_set = false;

    this.on("set", () => {
      this.is_set = true;
    });
  }

  set(): void {
    this.emit("set");
  }
}

export function createStopEvent(): StopEvent {
  return new StopEvent();
}

function pingLoop(pingFn: () => void, stopEvent: StopEvent): Promise<void> {
  const waitFn = (waitTime: number): Promise<void> =>
    new Promise((resolve) => {
      if (!stopEvent.is_set) {
        setTimeout(resolve, waitTime * 1000);
      }
    });

  const onBackoff = (state: RetryState): void => {
    const err = state.exception as ServiceError;
    if (!err) return;

    const statusCode = err.code;
    if (statusCode === status.DEADLINE_EXCEEDED) {
      if (state.actualWait !== undefined) {
        state.actualWait = Math.max(state.actualWait - PING_CALL_TIMEOUT, 0);
      }
    }
  };

  const wrappedPing = (): void => {
    if (!stopEvent.is_set) {
      pingFn();
    }
  };

  const retrier = new RetryInvoker(exponential, Error, null, null, {
    onBackoff,
    waitFunction: waitFn,
  });

  return new Promise(async (resolve) => {
    while (!stopEvent.is_set) {
      await retrier.invoke(wrappedPing);
    }
    resolve(); // Resolve when stopEvent is triggered
  });
}

// TypeScript version of startPingLoop
export function startPingLoop(pingFn: () => void, stopEvent: StopEvent): NodeJS.Timeout {
  // Start the loop, but do not block
  pingLoop(pingFn, stopEvent).then(() => {
    console.log("Ping loop terminated.");
  });

  const intervalId = setInterval(() => {
    if (stopEvent.is_set) {
      clearInterval(intervalId); // Clear the interval when stopEvent is set
    }
  }, 1000); // Interval to keep the loop alive

  return intervalId; // Return the interval ID}
}
