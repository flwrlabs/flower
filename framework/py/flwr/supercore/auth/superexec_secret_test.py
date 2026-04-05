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
"""Tests for SuperExec secret loading utilities."""


import argparse
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from .superexec_secret import (
    add_superexec_auth_secret_args,
    generate_superexec_auth_secret,
    load_superexec_auth_secret,
)


class TestSuperExecSecret(TestCase):
    """Unit tests for SuperExec shared-secret loading helpers."""

    def test_add_superexec_auth_secret_args_are_mutually_exclusive(self) -> None:
        """CLI args should reject specifying both file and stdin."""
        parser = argparse.ArgumentParser()
        add_superexec_auth_secret_args(parser)

        with self.assertRaises(SystemExit):
            _ = parser.parse_args(
                [
                    "--superexec-auth-secret-file",
                    "/tmp/secret",
                    "--superexec-auth-secret-stdin",
                ]
            )

    def test_load_secret_returns_none_when_unset(self) -> None:
        """No secret source should return None."""
        loaded = load_superexec_auth_secret(secret_file=None, secret_stdin=False)
        self.assertIsNone(loaded)

    def test_load_secret_from_file(self) -> None:
        """File source should be read and normalized."""
        with TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "secret.txt"
            secret_path.write_bytes(b"  abc123  \n")

            loaded = load_superexec_auth_secret(
                secret_file=str(secret_path), secret_stdin=False
            )

        self.assertEqual(loaded, b"abc123")

    def test_load_secret_from_stdin(self) -> None:
        """Stdin source should be read and normalized."""
        fake_stdin = SimpleNamespace(buffer=BytesIO(b"  stdin-secret \n"))
        with patch("sys.stdin", fake_stdin):
            loaded = load_superexec_auth_secret(secret_file=None, secret_stdin=True)

        self.assertEqual(loaded, b"stdin-secret")

    def test_load_secret_rejects_empty_after_strip(self) -> None:
        """Empty (or whitespace-only) secrets should be rejected."""
        with TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "empty-secret.txt"
            secret_path.write_bytes(b"  \n\t ")
            with self.assertRaises(ValueError):
                _ = load_superexec_auth_secret(
                    secret_file=str(secret_path), secret_stdin=False
                )

        fake_stdin = SimpleNamespace(buffer=BytesIO(b"   "))
        with patch("sys.stdin", fake_stdin):
            with self.assertRaises(ValueError):
                _ = load_superexec_auth_secret(secret_file=None, secret_stdin=True)

    def test_generate_superexec_auth_secret(self) -> None:
        """Secret generation should return bytes of requested length."""
        secret = generate_superexec_auth_secret(48)
        self.assertIsInstance(secret, bytes)
        self.assertEqual(len(secret), 48)
