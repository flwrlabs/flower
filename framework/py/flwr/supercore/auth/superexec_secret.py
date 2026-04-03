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
"""Utilities for SuperExec shared-secret provisioning."""


from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path


def add_superexec_auth_secret_args(parser: argparse.ArgumentParser) -> None:
    """Add shared-secret arguments for SuperExec HMAC auth."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--superexec-auth-secret-file",
        type=str,
        default=None,
        help="Path to a file containing the SuperExec shared secret.",
    )
    group.add_argument(
        "--superexec-auth-secret-stdin",
        action="store_true",
        help="Read the SuperExec shared secret from stdin.",
    )


def load_superexec_auth_secret(
    *,
    secret_file: str | None,
    secret_stdin: bool,
) -> bytes | None:
    """Load the SuperExec shared secret from file or stdin."""
    secret: bytes | None = None
    if secret_file is not None:
        secret = Path(secret_file).expanduser().read_bytes()
    elif secret_stdin:
        secret = sys.stdin.buffer.read()

    if secret is None:
        return None

    normalized = secret.strip()
    if normalized == b"":
        raise ValueError("SuperExec auth secret must not be empty")
    return normalized


def generate_superexec_auth_secret(num_bytes: int = 32) -> bytes:
    """Generate a random SuperExec shared secret."""
    return secrets.token_bytes(num_bytes)
