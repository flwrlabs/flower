"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AUTH_TOKEN_HEADER = exports.PUBLIC_KEY_HEADER = void 0;
exports.AuthenticateClientInterceptor = AuthenticateClientInterceptor;
const crypto_helpers_1 = require("./crypto_helpers");
exports.PUBLIC_KEY_HEADER = "public-key";
exports.AUTH_TOKEN_HEADER = "auth-token";
// Helper function to extract values from metadata
function getValueFromMetadata(key, metadata) {
    const values = metadata[key];
    return values.length > 0 && typeof values[0] === "string" ? values[0] : "";
}
function base64UrlEncode(buffer) {
    return buffer
        .toString("base64") // Standard Base64 encoding
        .replace(/\+/g, "-") // Replace + with -
        .replace(/\//g, "_") // Replace / with _
        .replace(/=+$/, ""); // Remove padding (trailing = characters)
}
function AuthenticateClientInterceptor(privateKey, publicKey) {
    let sharedSecret = null;
    let serverPublicKey = null;
    // Convert the public key to bytes and encode it
    const encodedPublicKey = base64UrlEncode((0, crypto_helpers_1.publicKeyToBytes)(publicKey));
    return {
        interceptUnary(next, method, input, options) {
            // Manipulate metadata before sending the request
            const metadata = options.meta || {};
            // Always add the public key to the metadata
            metadata[exports.PUBLIC_KEY_HEADER] = encodedPublicKey;
            const postprocess = "pingInterval" in input;
            // Add HMAC to metadata if a shared secret exists
            if (sharedSecret !== null) {
                const serializedMessage = method.I.toBinary(input);
                const hmac = (0, crypto_helpers_1.computeHMAC)(sharedSecret, Buffer.from(serializedMessage));
                metadata[exports.AUTH_TOKEN_HEADER] = base64UrlEncode(hmac);
            }
            const continuation = next(method, input, { ...options, meta: metadata });
            if (postprocess) {
                handlePostprocess(metadata);
            }
            return continuation;
        },
    };
    function handlePostprocess(metadata) {
        const serverPublicKeyBytes = getValueFromMetadata(exports.PUBLIC_KEY_HEADER, metadata);
        if (serverPublicKeyBytes.length > 0) {
            serverPublicKey = (0, crypto_helpers_1.bytesToPublicKey)(Buffer.from(serverPublicKeyBytes));
        }
        else {
            console.warn("Couldn't get server public key, server may be offline");
        }
        if (serverPublicKey) {
            sharedSecret = (0, crypto_helpers_1.generateSharedKey)(privateKey, serverPublicKey);
        }
    }
}
