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

import { FailureCode, Result } from '../../typing';
import { hkdf } from './cryptoUtils';

const webCrypto = (globalThis as { crypto?: Crypto }).crypto;

const requireWebCrypto = () => {
  if (!webCrypto) {
    throw new Error('Web Crypto API is not available.');
  }
  return webCrypto;
};

const KEY_TYPE = 'ECDH';
const CURVE = 'P-384'; // secp384r1 in Web Crypto
const KEY_FORMAT = 'spki';

const AES_KEY_LENGTH = 32;
const HASH_ALG = 'SHA-256';
const HKDF_INFO = new TextEncoder().encode('ecdh key exchange');
/** Handles key generation and ECDH shared secret derivation */
export class KeyManager {
  private privateKey: CryptoKey | null = null;
  private publicKey: CryptoKey | null = null;
  private sharedSecretKey: ArrayBuffer | null = null;

  /**
   * Generate a new ECDH key pair.
   */
  async generateKeyPair() {
    const keyParams: EcKeyGenParams = {
      name: KEY_TYPE,
      namedCurve: CURVE,
    };
    const keyPair = await requireWebCrypto().subtle.generateKey(keyParams, true, [
      'deriveKey',
      'deriveBits',
    ]);

    this.privateKey = keyPair.privateKey;
    this.publicKey = keyPair.publicKey;
  }

  /**
   * Export the public key as a Base64 string.
   */
  async exportPublicKey(): Promise<Result<string>> {
    if (!this.publicKey) {
      return {
        ok: false,
        failure: { code: FailureCode.EncryptionError, description: 'Public key not generated.' },
      };
    }
    const exportedKey = await requireWebCrypto().subtle.exportKey(KEY_FORMAT, this.publicKey);
    return { ok: true, value: btoa(String.fromCharCode(...new Uint8Array(exportedKey))) };
  }

  /**
   * Derive a shared secret using the server's public key.
   */
  async deriveSharedSecret(serverPublicKeyBase64: string): Promise<Result<ArrayBuffer>> {
    if (!this.privateKey) {
      return {
        ok: false,
        failure: {
          code: FailureCode.EncryptionError,
          description: 'Private key is not initialized.',
        },
      };
    }

    const serverPublicKeyBuffer = Uint8Array.from(atob(serverPublicKeyBase64), (char) =>
      char.charCodeAt(0)
    );

    // Import server's public key
    const importParams: EcKeyImportParams = {
      name: KEY_TYPE,
      namedCurve: CURVE,
    };
    const serverPublicKey = await requireWebCrypto().subtle.importKey(
      KEY_FORMAT,
      serverPublicKeyBuffer,
      importParams,
      false,
      []
    );

    // Compute shared secret
    const sharedSecret = await requireWebCrypto().subtle.deriveBits(
      { name: KEY_TYPE, public: serverPublicKey },
      this.privateKey,
      384
    );

    // Apply HKDF to derive final encryption key
    this.sharedSecretKey = await hkdf(HASH_ALG, sharedSecret, HKDF_INFO, AES_KEY_LENGTH);
    return { ok: true, value: this.sharedSecretKey };
  }

  getSharedSecretKey(): ArrayBuffer | null {
    return this.sharedSecretKey;
  }
}
