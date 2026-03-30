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

import os from 'os';
import path from 'path';
import { beforeEach, describe, expect, it, vi } from 'vitest';

// --- In-memory file system ---
// This object will simulate our file system.
let fileSystem: Record<string, string> = {};

// --- Mock fs/promises ---
// We intercept dynamic imports of 'fs/promises' by providing mocked implementations.
vi.mock('fs/promises', () => ({
  readFile: vi.fn(async (filePath: string, _encoding: string) => {
    await Promise.resolve();
    if (filePath in fileSystem) {
      return fileSystem[filePath];
    }
    throw new Error('File not found');
  }),
  writeFile: vi.fn(async (filePath: string, data: string, _encoding: string) => {
    await Promise.resolve();
    fileSystem[filePath] = data;
  }),
  mkdir: vi.fn(async (_folder: string, _options: { recursive: boolean }) => {
    // In our mock, we don't need to do anything.
    await Promise.resolve();
    return;
  }),
}));

import { VERSION } from '../../constants';
import { CachedMapping, NodeCacheStorage } from './storage';

// --- Tests ---
describe('NodeCacheStorage', () => {
  let cacheStorage: NodeCacheStorage;
  let expectedCacheFilePath: string;

  beforeEach(() => {
    // Reset the in-memory file system before each test.
    fileSystem = {};
    cacheStorage = new NodeCacheStorage();

    // Compute the expected cache file path as created in getCacheFilePath():
    const homeDir = os.homedir();
    const cacheFolder = path.join(homeDir, '.flwr', 'cache');
    expectedCacheFilePath = path.join(cacheFolder, `intelligence-${VERSION}-model-names.json`);
  });

  it('should return null when no cache file exists', async () => {
    const item = await cacheStorage.getItem('nonexistent');
    expect(item).toBeNull();
  });

  it('should set and then get an item correctly', async () => {
    const key = 'key1';
    const value = 'value1';

    // Set the item.
    await cacheStorage.setItem(key, value);
    // Get the item.
    const item = await cacheStorage.getItem(key);

    expect(item).not.toBeNull();
    expect(item?.value).toBe(value);

    // Also, verify that our in-memory file system has been updated.
    const data = fileSystem[expectedCacheFilePath];
    expect(data).toBeDefined();
    const parsedData = JSON.parse(data) as CachedMapping;
    expect(parsedData.mapping[key].value).toBe(value);
  });

  it('should remove an item when setItem is called with null', async () => {
    const key = 'key2';
    const value = 'value2';

    // First, set the item.
    await cacheStorage.setItem(key, value);
    let item = await cacheStorage.getItem(key);
    expect(item).not.toBeNull();
    expect(item?.value).toBe(value);

    // Now, remove the item by setting it to null.
    await cacheStorage.setItem(key, null);
    item = await cacheStorage.getItem(key);
    expect(item).toBeNull();

    // Verify the in-memory cache file no longer contains the key.
    const data = fileSystem[expectedCacheFilePath];
    expect(data).toBeDefined();
    const parsedData = JSON.parse(data) as CachedMapping;
    expect(parsedData.mapping[key]).toBeUndefined();
  });
});
