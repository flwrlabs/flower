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

import getRandomValues from 'get-random-values';

import { FailureCode, Message, Result } from '../../typing';
import { KeyManager } from './keyManager';
import { NetworkService } from './networkService';

const webCrypto = (globalThis as { crypto?: Crypto }).crypto;

const requireWebCrypto = () => {
  if (!webCrypto) {
    throw new Error('Web Crypto API is not available.');
  }
  return webCrypto;
};

const GCM_IV_LENGTH = 12;
const BIT_TAG_LENGTH = 8 * 16; // 128 bits or 16 bytes
const CRYPTO_ALG = 'AES-GCM';

/** Orchestrates key management, key exchange, and message encryption/decryption */
export class CryptographyHandler {
  private keyManager: KeyManager;
  private networkService: NetworkService;
  private sharedSecretKey: ArrayBuffer | null = null;
  #encryptionId: string | null = null;

  constructor(serverUrl: string, apiKey: string) {
    this.keyManager = new KeyManager();
    this.networkService = new NetworkService(serverUrl, apiKey);
  }

  get encryptionId() {
    return this.#encryptionId;
  }

  async initializeKeysAndExchange(): Promise<Result<void>> {
    if (this.networkService.isClientKeyExpired()) {
      await this.keyManager.generateKeyPair();
      const clientPublicKey = await this.keyManager.exportPublicKey();
      if (!clientPublicKey.ok) {
        return clientPublicKey;
      }
      const encryptionId = await this.networkService.submitClientPublicKey(clientPublicKey.value);
      if (!encryptionId.ok) {
        return encryptionId;
      }
      this.#encryptionId = encryptionId.value;
    }
    const serverPublicKey = await this.networkService.getServerPublicKey();
    if (!serverPublicKey.ok) {
      return serverPublicKey;
    }
    const sharedSecretKey = await this.keyManager.deriveSharedSecret(serverPublicKey.value);
    if (!sharedSecretKey.ok) {
      return sharedSecretKey;
    }
    this.sharedSecretKey = sharedSecretKey.value;
    return { ok: true, value: undefined };
  }

  async encryptMessage(message: string): Promise<Result<string>> {
    if (!this.sharedSecretKey) {
      return {
        ok: false,
        failure: {
          code: FailureCode.EncryptionError,
          description: 'Shared secret is not derived.',
        },
      };
    }

    try {
      const iv = getRandomValues(new Uint8Array(GCM_IV_LENGTH));
      const aesKey = await requireWebCrypto().subtle.importKey(
        'raw',
        this.sharedSecretKey,
        { name: CRYPTO_ALG },
        false,
        ['encrypt']
      );

      const encodedMessage = new TextEncoder().encode(message);
      const encryptedData = await requireWebCrypto().subtle.encrypt(
        { name: CRYPTO_ALG, iv, tagLength: BIT_TAG_LENGTH },
        aesKey,
        encodedMessage
      );

      const encryptedBytes = new Uint8Array(encryptedData);
      const combined = new Uint8Array(iv.length + encryptedBytes.length);
      combined.set(iv, 0);
      combined.set(encryptedBytes, iv.length);

      return { ok: true, value: btoa(String.fromCharCode(...combined)) };
    } catch (error) {
      return {
        ok: false,
        failure: { code: FailureCode.EncryptionError, description: String(error) },
      };
    }
  }

  async encryptMessages(messages: Message[]): Promise<Result<void>> {
    for (const message of messages) {
      const encryptedContent = await this.encryptMessage(message.content);
      if (!encryptedContent.ok) {
        return encryptedContent;
      }
      message.content = encryptedContent.value;
    }
    return { ok: true, value: undefined };
  }

  async decryptMessage(encryptedMessage: string): Promise<Result<string>> {
    if (!this.sharedSecretKey) {
      return {
        ok: false,
        failure: {
          code: FailureCode.EncryptionError,
          description: 'Shared secret is not derived.',
        },
      };
    }

    try {
      const data = Uint8Array.from(atob(encryptedMessage), (char) => char.charCodeAt(0));
      const iv = data.slice(0, GCM_IV_LENGTH);
      const ciphertext = data.slice(GCM_IV_LENGTH);

      const aesKey = await requireWebCrypto().subtle.importKey(
        'raw',
        this.sharedSecretKey,
        { name: CRYPTO_ALG },
        false,
        ['decrypt']
      );

      const plaintext = await requireWebCrypto().subtle.decrypt(
        { name: CRYPTO_ALG, iv },
        aesKey,
        ciphertext
      );
      return { ok: true, value: new TextDecoder().decode(plaintext) };
    } catch (error: unknown) {
      return {
        ok: false,
        failure: { code: FailureCode.EncryptionError, description: String(error) },
      };
    }
  }
}
