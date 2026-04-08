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
import { RetryInvoker, constant } from "./retry_invoker";
describe("RetryInvoker", () => {
  // Mocking utilities (similar to pytest's fixture)
  let mockTime: jest.SpyInstance;
  let mockSleep: jest.SpyInstance;

  beforeEach(() => {
    mockTime = jest.spyOn(Date, "now").mockImplementation(() => 0);

    mockSleep = jest
      .spyOn(global, "setTimeout")
      .mockImplementation((fn: () => void, ms?: number) => {
        fn(); // Immediately call the function to avoid waiting
        return setTimeout(fn, ms) as unknown as NodeJS.Timeout; // Return a proper Timeout object
      });
  });

  afterEach(() => {
    mockTime.mockRestore();
    mockSleep.mockRestore();
  });

  // Test successful invocation
  it("should succeed when the function does not throw", async () => {
    // Prepare
    const successHandler = jest.fn();
    const backoffHandler = jest.fn();
    const giveupHandler = jest.fn();
    const invoker = new RetryInvoker(() => constant(0.1), Error, null, null, {
      onSuccess: successHandler,
      onBackoff: backoffHandler,
      onGiveup: giveupHandler,
    });
    const successfulFunction = () => "success";

    // Assert that the invoker returns the correct result
    const result = await invoker.invoke(successfulFunction);
    expect(result).toBe("success");
  });

  it("should retry and fail on the failing function", async () => {
    const invoker = new RetryInvoker(
      () => constant(0.1), // Retry every 0.1 seconds
      Error, // Retry on Error
      2, // Maximum 2 retries
      2.5, // Maximum time to retry (in seconds)
    );

    const failingFunction = () => {
      throw new Error("failed");
    };

    // Assert that the invoker throws an error
    await expect(async () => {
      await invoker.invoke(failingFunction);
    }).rejects.toThrow("failed");
  });

  it("should call onSuccess handler when successful", async () => {
    const successHandler = jest.fn();
    const invoker = new RetryInvoker(() => constant(0.1), Error, 2, 2.5, {
      onSuccess: successHandler,
    });

    const successfulFunction = () => "success";

    // Call the function and assert success
    await invoker.invoke(successfulFunction);

    // Ensure the onSuccess handler was called
    expect(successHandler).toHaveBeenCalled();
  });

  it("should retry on failure and call onBackoff", async () => {
    const backoffHandler = jest.fn();
    const invoker = new RetryInvoker(() => constant(0.1), Error, 2, 2.5, {
      onBackoff: backoffHandler,
    });

    const failingFunction = () => {
      throw new Error("failed");
    };

    // Assert the invoker throws an error and triggers retries
    await expect(invoker.invoke(failingFunction)).rejects.toThrow("failed");

    // Ensure the backoff handler was called
    expect(backoffHandler).toHaveBeenCalled();
  });

  it("should stop after max retries", async () => {
    const invoker = new RetryInvoker(
      () => constant(0.1),
      Error,
      2, // Max retries is 2
      2.5,
    );

    const failingFunction = () => {
      throw new Error("failed");
    };

    // Assert the invoker gives up after max retries
    await expect(invoker.invoke(failingFunction)).rejects.toThrow("failed");
  });
});
