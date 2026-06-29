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

import { VERSION } from '../../constants';

interface CachedEntry {
  value: string;
  lastUpdate: number;
}

export abstract class CacheStorage {
  abstract getItem(key: string): Promise<CachedEntry | null>;
  abstract setItem(key: string, value: string | null): Promise<void>;
}

export class WebCacheStorage extends CacheStorage {
  private readonly CACHE_KEY_PREFIX = `flwr-cache-${VERSION}-`;

  async getItem(key: string): Promise<CachedEntry | null> {
    await Promise.resolve();
    const data = localStorage.getItem(`${this.CACHE_KEY_PREFIX}${key}`);
    if (data) {
      try {
        return JSON.parse(data) as CachedEntry;
      } catch {
        return null;
      }
    }
    return null;
  }
  async setItem(key: string, value: string | null): Promise<void> {
    await Promise.resolve();
    if (value) {
      localStorage.setItem(
        `${this.CACHE_KEY_PREFIX}${key}`,
        JSON.stringify({ value, lastUpdate: Date.now() })
      );
    } else {
      localStorage.removeItem(`${this.CACHE_KEY_PREFIX}${key}`);
    }
  }
}

export interface CachedMapping {
  mapping: Record<string, CachedEntry>;
}

export class NodeCacheStorage extends CacheStorage {
  private filePath: string | undefined;

  private async getCacheFilePath(): Promise<string> {
    if (!this.filePath) {
      const os = await import('os');
      const path = await import('path');
      const { mkdir } = await import('fs/promises');
      const homeDir = os.homedir();
      const cacheFolder = path.join(homeDir, '.flwr', 'cache');
      await mkdir(cacheFolder, { recursive: true });
      this.filePath = path.join(cacheFolder, `intelligence-${VERSION}-model-names.json`);
    }
    return this.filePath;
  }

  private async load(): Promise<CachedMapping | null> {
    try {
      const { readFile } = await import('fs/promises');
      const filePath = await this.getCacheFilePath();
      const data = await readFile(filePath, 'utf-8');
      return JSON.parse(data) as CachedMapping;
    } catch (_) {
      return null;
    }
  }

  private async save(cache: CachedMapping): Promise<void> {
    try {
      const { writeFile } = await import('fs/promises');
      const filePath = await this.getCacheFilePath();
      await writeFile(filePath, JSON.stringify(cache), 'utf-8');
    } catch (_) {
      return undefined;
    }
  }

  async getItem(key: string): Promise<CachedEntry | null> {
    let cache = await this.load();
    cache ??= { mapping: {} };

    if (key in cache.mapping) {
      return cache.mapping[key];
    }
    return null;
  }

  async setItem(key: string, value: string | null): Promise<void> {
    const now = Date.now();
    let cache = await this.load();
    cache ??= { mapping: {} };
    if (value) {
      cache.mapping[key] = { value: value, lastUpdate: now };
      await this.save(cache);
    } else {
      if (key in cache.mapping) {
        const { [key]: removed, ...rest } = cache.mapping;
        cache.mapping = rest;
        await this.save(cache);
      }
    }
  }
}
