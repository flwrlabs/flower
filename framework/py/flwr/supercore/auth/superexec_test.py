# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for SuperExec auth primitives."""


from unittest import TestCase

from flwr.proto.appio_pb2 import RequestTokenRequest  # pylint: disable=E0611

from .superexec import (
    canonicalize_superexec_auth_input,
    compute_request_body_sha256,
    compute_superexec_signature,
    derive_auth_secret,
    verify_superexec_signature,
)


class TestSuperExecAuthPrimitives(TestCase):
    """Unit tests for SuperExec auth helpers."""

    def test_canonicalize_superexec_auth_input(self) -> None:
        """Canonicalization should produce deterministic UTF-8 bytes."""
        canonical = canonicalize_superexec_auth_input(
            method="/flwr.proto.ServerAppIo/RequestToken",
            timestamp=123,
            nonce="nonce-1",
            body_sha256="abc",
        )

        self.assertEqual(
            canonical,
            (
                b"method=/flwr.proto.ServerAppIo/RequestToken\n"
                b"ts=123\n"
                b"nonce=nonce-1\n"
                b"body_sha256=abc"
            ),
        )

    def test_compute_request_body_sha256_is_deterministic(self) -> None:
        """Body SHA256 should be deterministic for equivalent request payloads."""
        req_a = RequestTokenRequest(run_id=11)
        req_b = RequestTokenRequest(run_id=11)
        req_c = RequestTokenRequest(run_id=12)

        hash_a = compute_request_body_sha256(req_a)
        hash_b = compute_request_body_sha256(req_b)
        hash_c = compute_request_body_sha256(req_c)

        self.assertEqual(hash_a, hash_b)
        self.assertNotEqual(hash_a, hash_c)
        self.assertEqual(len(hash_a), 64)

    def test_derive_auth_secret_is_deterministic(self) -> None:
        """Derived auth secret should be deterministic for one master secret."""
        master_secret = b"master-secret"

        first = derive_auth_secret(master_secret)
        second = derive_auth_secret(master_secret)
        other = derive_auth_secret(b"other-master-secret")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertTrue(first)

    def test_verify_superexec_signature(self) -> None:
        """Signature verification should return True only for matching signatures."""
        auth_secret = derive_auth_secret(b"master-secret")
        good_signature = compute_superexec_signature(
            auth_secret=auth_secret,
            method="/flwr.proto.ServerAppIo/RequestToken",
            timestamp=456,
            nonce="nonce-2",
            body_sha256="f" * 64,
        )
        bad_signature = "0" * 64

        self.assertTrue(verify_superexec_signature(good_signature, good_signature))
        self.assertFalse(verify_superexec_signature(good_signature, bad_signature))
