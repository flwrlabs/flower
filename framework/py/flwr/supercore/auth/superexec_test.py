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


from datetime import UTC, datetime
from unittest import TestCase
from unittest.mock import Mock, patch

from flwr.proto.runtime_pb2 import ClaimTaskRequest  # pylint: disable=E0611
from flwr.supercore.constant import (
    MAX_TIMESTAMP_DIFF_SECONDS,
    MIN_TIMESTAMP_DIFF_SECONDS,
)

from .superexec import (
    canonicalize_superexec_auth_input,
    compute_request_body_sha256,
    compute_superexec_signature,
    derive_auth_secret,
    verify_superexec_request,
    verify_superexec_signature,
)

_METHOD = "/flwr.proto.Runtime/ClaimTask"
_TIMESTAMP = 1_000


class TestSuperExecAuthPrimitives(TestCase):
    """Unit tests for SuperExec auth helpers."""

    def test_canonicalize_superexec_auth_input(self) -> None:
        """Canonicalization should produce deterministic UTF-8 bytes."""
        canonical = canonicalize_superexec_auth_input(
            method="/flwr.proto.Runtime/RequestToken",
            timestamp=123,
            nonce="nonce-1",
            body_sha256="abc",
        )

        self.assertEqual(
            canonical,
            (
                b"method=/flwr.proto.Runtime/RequestToken\n"
                b"ts=123\n"
                b"nonce=nonce-1\n"
                b"body_sha256=abc"
            ),
        )

    def test_compute_request_body_sha256_is_deterministic(self) -> None:
        """Body SHA256 should be deterministic for equivalent request payloads."""
        req_a = ClaimTaskRequest(task_id=11)
        req_b = ClaimTaskRequest(task_id=11)
        req_c = ClaimTaskRequest(task_id=12)

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
            method="/flwr.proto.Runtime/ClaimTask",
            timestamp=456,
            nonce="nonce-2",
            body_sha256="f" * 64,
        )
        bad_signature = "0" * 64

        self.assertTrue(verify_superexec_signature(good_signature, good_signature))
        self.assertFalse(verify_superexec_signature(good_signature, bad_signature))


class TestVerifySuperExecRequest(TestCase):
    """Unit tests for complete SuperExec request verification."""

    def setUp(self) -> None:
        """Create valid authentication fields."""
        self.request = ClaimTaskRequest(task_id=11)
        self.auth_secret = derive_auth_secret(b"master-secret")
        self.body_sha256 = compute_request_body_sha256(self.request)
        self.signature = compute_superexec_signature(
            auth_secret=self.auth_secret,
            method=_METHOD,
            timestamp=_TIMESTAMP,
            nonce="nonce",
            body_sha256=self.body_sha256,
        )
        self.nonce_store = Mock()
        self.nonce_store.reserve_nonce.return_value = True

    def _verify(self, **overrides: object) -> bool:
        values = {
            "request": self.request,
            "method": _METHOD,
            "auth_secret": self.auth_secret,
            "timestamp_raw": str(_TIMESTAMP),
            "nonce": "nonce",
            "body_sha256_header": self.body_sha256,
            "signature": self.signature,
            "nonce_store": self.nonce_store,
            **overrides,
        }
        return verify_superexec_request(**values)  # type: ignore[arg-type]

    @patch(
        "flwr.supercore.auth.superexec.now",
        return_value=datetime.fromtimestamp(_TIMESTAMP + 1, UTC),
    )
    def test_accepts_valid_request(self, _now: Mock) -> None:
        """Accept a valid signature and reserve its nonce."""
        self.assertTrue(self._verify())
        self.nonce_store.reserve_nonce.assert_called_once_with(
            namespace=f"superexec:{_METHOD}",
            nonce="nonce",
            expires_at=float(_TIMESTAMP + MAX_TIMESTAMP_DIFF_SECONDS),
        )

    @patch(
        "flwr.supercore.auth.superexec.now",
        return_value=datetime.fromtimestamp(_TIMESTAMP + 1, UTC),
    )
    def test_rejects_invalid_authentication_fields(self, _now: Mock) -> None:
        """Reject incomplete, expired, or mismatched authentication fields."""
        invalid_overrides = [
            {"timestamp_raw": None},
            {"timestamp_raw": "invalid"},
            {"timestamp_raw": str(_TIMESTAMP - MAX_TIMESTAMP_DIFF_SECONDS)},
            {"timestamp_raw": str(_TIMESTAMP + 1 - MIN_TIMESTAMP_DIFF_SECONDS)},
            {"body_sha256_header": "invalid"},
            {"signature": "invalid"},
        ]
        for overrides in invalid_overrides:
            with self.subTest(overrides=overrides):
                self.assertFalse(self._verify(**overrides))

        self.nonce_store.reserve_nonce.assert_not_called()

    @patch(
        "flwr.supercore.auth.superexec.now",
        return_value=datetime.fromtimestamp(_TIMESTAMP + 1, UTC),
    )
    def test_rejects_replayed_nonce(self, _now: Mock) -> None:
        """Reject a valid request when its nonce was already reserved."""
        self.nonce_store.reserve_nonce.return_value = False

        self.assertFalse(self._verify())
