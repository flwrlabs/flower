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

import { assert, beforeEach, describe, expect, it, vi } from 'vitest';

import { CryptographyHandler } from './remoteEngine/cryptoHandler';
import { getTimestamp } from './remoteEngine/cryptoUtils';
import { KeyManager } from './remoteEngine/keyManager';
import { NetworkService } from './remoteEngine/networkService';

const API_KEY = process.env.FI_API_KEY ?? '';
const REMOTE_URL = process.env.FI_DEV_REMOTE_URL ?? '';
const STRING_TIMESTAMP = '2025-03-06T13:19:47.353034';
const INT_TIMESTAMP = 1741267187353;

vi.mock('./constants', () => ({
  DEFAULT_MODEL: 'meta/llama3.2-1b/instruct-fp16',
  REMOTE_URL: REMOTE_URL,
  VERSION: '0.1.7',
  SDK: 'TS',
  ALLOWED_ROLES: ['system', 'assistant', 'user'],
}));

describe('CryptographyHandler', () => {
  let cryptographyHandler: CryptographyHandler;

  beforeEach(async () => {
    cryptographyHandler = new CryptographyHandler(REMOTE_URL, API_KEY);
    await cryptographyHandler.initializeKeysAndExchange();
  });

  it('should initialize keys and perform key exchange', () => {
    if (!cryptographyHandler.encryptionId)
      assert.fail('encryptionId is undefined, submitting client  public key probably failed.');
    expect(cryptographyHandler.encryptionId.length).toBeGreaterThan(0);
  });

  it('should correctly encrypt and decrypt a basic message', async () => {
    const message = 'Hello, world!';

    const encryptedRes = await cryptographyHandler.encryptMessage(message);
    if (!encryptedRes.ok) {
      assert.fail(encryptedRes.failure.description);
    }
    // Ciphertext should be different from the original message
    expect(encryptedRes.value).not.toEqual(message);

    const decryptedRes = await cryptographyHandler.decryptMessage(encryptedRes.value);
    if (!decryptedRes.ok) {
      assert.fail(decryptedRes.failure.description);
    }
    expect(decryptedRes.value).toEqual(message);
  });

  it('should return different ciphertexts for the same message', async () => {
    const message = 'Hello, world!';

    const encryptedRes1 = await cryptographyHandler.encryptMessage(message);
    if (!encryptedRes1.ok) {
      assert.fail(encryptedRes1.failure.description);
    }

    const encryptedRes2 = await cryptographyHandler.encryptMessage(message);
    if (!encryptedRes2.ok) {
      assert.fail(encryptedRes2.failure.description);
    }

    // Each encryption should produce a different ciphertext (e.g., different IVs)
    expect(encryptedRes1.value).not.toEqual(encryptedRes2.value);
  });

  it('should not decrypt a message with a different key', async () => {
    const message = 'Secret Data';

    const encryptedRes = await cryptographyHandler.encryptMessage(message);
    if (!encryptedRes.ok) {
      assert.fail(encryptedRes.failure.description);
    }
    expect(encryptedRes.ok).toBe(true);

    // Create a new instance with a different key by performing a separate key exchange
    const newHandler = new CryptographyHandler(REMOTE_URL, API_KEY);
    await newHandler.initializeKeysAndExchange();

    const decryptRes = await newHandler.decryptMessage(encryptedRes.value);
    expect(decryptRes.ok).toBe(false);
    if (!decryptRes.ok) {
      expect(decryptRes.failure.description.length).toBeGreaterThan(0);
    }
  });

  it('should handle empty messages correctly', async () => {
    const message = '';

    const encryptedRes = await cryptographyHandler.encryptMessage(message);
    if (!encryptedRes.ok) {
      assert.fail(encryptedRes.failure.description);
    }
    // Even an empty string should produce a non-empty ciphertext
    expect(encryptedRes.value).not.toEqual('');

    const decryptedRes = await cryptographyHandler.decryptMessage(encryptedRes.value);
    if (!decryptedRes.ok) {
      assert.fail(decryptedRes.failure.description);
    }
    expect(decryptedRes.value).toEqual(message);
  });

  it('should encrypt and decrypt messages with special characters', async () => {
    const message = '¡Hola! 😊 🚀';

    const encryptedRes = await cryptographyHandler.encryptMessage(message);
    if (!encryptedRes.ok) {
      assert.fail(encryptedRes.failure.description);
    }
    expect(encryptedRes.value).not.toEqual(message);

    const decryptedRes = await cryptographyHandler.decryptMessage(encryptedRes.value);
    if (!decryptedRes.ok) {
      assert.fail(decryptedRes.failure.description);
    }
    expect(decryptedRes.value).toEqual(message);
  });

  it('should encrypt and decrypt long messages', async () => {
    const message = 'A'.repeat(10_000); // 10,000 character message

    const encryptedRes = await cryptographyHandler.encryptMessage(message);
    if (!encryptedRes.ok) {
      assert.fail(encryptedRes.failure.description);
    }
    expect(encryptedRes.value.length).toBeGreaterThan(0);

    const decryptedRes = await cryptographyHandler.decryptMessage(encryptedRes.value);
    if (!decryptedRes.ok) {
      assert.fail(decryptedRes.failure.description);
    }
    expect(decryptedRes.value).toEqual(message);
  });

  it('should fail to decrypt if ciphertext is modified', async () => {
    const message = 'Tamper test';
    const encryptedRes = await cryptographyHandler.encryptMessage(message);
    if (!encryptedRes.ok) {
      assert.fail(encryptedRes.failure.description);
    }

    // Modify the ciphertext by changing its last character
    const tamperedMessage = encryptedRes.value.slice(0, -1) + 'A';

    const decryptedRes = await cryptographyHandler.decryptMessage(tamperedMessage);
    expect(decryptedRes.ok).toBe(false);
    if (!decryptedRes.ok) {
      expect(decryptedRes.failure.description.length).toBeGreaterThan(0);
    }
  });
});

describe('KeyManager', () => {
  let keyManager: KeyManager;

  beforeEach(async () => {
    keyManager = new KeyManager();
    await keyManager.generateKeyPair();
  });

  it('should generate a key pair', async () => {
    expect(await keyManager.exportPublicKey()).toBeDefined();
  });

  it('should export a valid Base64-encoded public key', async () => {
    const publicKey = await keyManager.exportPublicKey();
    expect(publicKey).toBeDefined();
  });

  it('should derive a shared secret correctly', async () => {
    const publicKey = await keyManager.exportPublicKey();
    if (!publicKey.ok) {
      assert.fail(publicKey.failure.description);
    }
    const sharedSecret = await keyManager.deriveSharedSecret(publicKey.value);
    if (!sharedSecret.ok) {
      assert.fail(sharedSecret.failure.description);
    }
    expect(sharedSecret.value.byteLength).toBe(32);
  });
});

describe('NetworkService', () => {
  let networkService: NetworkService;

  beforeEach(() => {
    networkService = new NetworkService(REMOTE_URL, API_KEY);
  });

  it('should fetch and return the server public key', async () => {
    const publicKey = await networkService.getServerPublicKey();
    expect(publicKey).toBeDefined();
  });
});

describe('getTimestamp', () => {
  it('should convert ISO 8601 date string to timestamp', () => {
    const timestamp = getTimestamp(STRING_TIMESTAMP);
    expect(timestamp).toBe(INT_TIMESTAMP);
  });
});
