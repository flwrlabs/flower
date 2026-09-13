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
from pathlib import Path


def add_superexec_auth_secret_args(parser: argparse.ArgumentParser) -> None:
    """Add shared-secret arguments for SuperExec HMAC auth."""
    parser.add_argument(
        "--superexec-auth-secret-file",
        type=str,
        default=None,
        help=(
            "Path to a file containing the SuperExec shared secret. The file "
            "is read as exact raw bytes."
        ),
    )


def load_superexec_auth_secret(secret_file: str | None) -> bytes | None:
    """Load the SuperExec shared secret from file."""
    secret: bytes | None = None
    if secret_file is not None:
        # File input is treated as exact raw bytes.
        secret_path = Path(secret_file).expanduser()
        try:
            secret = secret_path.read_bytes()
        except OSError as err:
            raise ValueError(
                f"Failed to read SuperExec auth secret from file '{secret_path}': "
                f"{err}"
            ) from err

    if secret is None:
        return None

    if secret == b"":
        raise ValueError("SuperExec auth secret must not be empty")
    return secret
