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
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from .superexec_secret import add_superexec_auth_secret_args, load_superexec_auth_secret


class TestSuperExecSecret(TestCase):
    """Unit tests for SuperExec shared-secret loading helpers."""

    def test_add_superexec_auth_secret_args(self) -> None:
        """CLI should accept a secret file argument."""
        parser = argparse.ArgumentParser()
        add_superexec_auth_secret_args(parser)

        args = parser.parse_args(["--superexec-auth-secret-file", "/tmp/secret"])
        self.assertEqual(args.superexec_auth_secret_file, "/tmp/secret")

    def test_load_secret_returns_none_when_unset(self) -> None:
        """No secret source should return None."""
        loaded = load_superexec_auth_secret(secret_file=None)
        self.assertIsNone(loaded)

    def test_load_secret_from_file(self) -> None:
        """File source should be read as raw bytes."""
        with TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "secret.txt"
            secret_path.write_bytes(b"abc123")

            loaded = load_superexec_auth_secret(secret_file=str(secret_path))

        self.assertEqual(loaded, b"abc123")

    def test_load_secret_rejects_empty_file(self) -> None:
        """Empty secret files should be rejected."""
        with TemporaryDirectory() as temp_dir:
            secret_path = Path(temp_dir) / "empty-secret.txt"
            secret_path.write_bytes(b"")
            with self.assertRaises(ValueError):
                _ = load_superexec_auth_secret(secret_file=str(secret_path))

    def test_load_secret_rejects_missing_file(self) -> None:
        """A missing secret file should raise ValueError."""
        with self.assertRaises(ValueError):
            _ = load_superexec_auth_secret(secret_file="/nonexistent/path/secret.txt")
