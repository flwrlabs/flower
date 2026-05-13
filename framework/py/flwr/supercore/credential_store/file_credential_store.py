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
"""File-based credential store implementation."""


import base64
import binascii
import tempfile
from pathlib import Path
from typing import cast

import yaml

from ..utils import get_flwr_home
from .credential_store import CredentialStore

CREDENTIAL_FILE_PATH = get_flwr_home() / "credentials.yaml"


class FileCredentialStore(CredentialStore):
    """File-based credential store implementation."""

    def __init__(self, file_path: Path | None = None) -> None:
        """Initialize the file credential store.

        Parameters
        ----------
        file_path : Path | None
            Path to the credentials file. If None, uses default path.
        """
        self.file_path = file_path or CREDENTIAL_FILE_PATH

    def _write_credentials_atomically(self, credentials: dict[str, str]) -> None:
        """Write credentials atomically by replacing the target file."""
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.file_path.parent,
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                yaml.safe_dump(credentials, temp_file)
            temp_path.replace(self.file_path)
        except (OSError, yaml.YAMLError):
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

    def _load_credentials(self) -> dict[str, str]:
        """Load credentials from file."""
        try:
            with self.file_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return cast(dict[str, str], data)
        except (OSError, yaml.YAMLError):
            pass
        self.file_path.unlink(missing_ok=True)
        return {}

    def _save_credentials(self, credentials: dict[str, str]) -> None:
        """Save credentials to file."""
        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_credentials_atomically(credentials)
        except (OSError, yaml.YAMLError):
            # Best-effort recovery: replace with an empty store and avoid raising.
            self.file_path.unlink(missing_ok=True)
            try:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                self._write_credentials_atomically({})
            except (OSError, yaml.YAMLError):
                pass

    def set(self, key: str, value: bytes) -> None:
        """Set a credential in the store."""
        credentials = self._load_credentials()
        credentials[key] = base64.b64encode(value).decode("utf-8")
        self._save_credentials(credentials)

    def get(self, key: str) -> bytes | None:
        """Get a credential from the store."""
        credentials = self._load_credentials()
        encoded_value = credentials.get(key)
        if encoded_value is None:
            return None
        try:
            return base64.b64decode(encoded_value, validate=True)
        except (binascii.Error, ValueError):
            self.file_path.unlink(missing_ok=True)
            return None

    def delete(self, key: str) -> None:
        """Delete a credential from the store."""
        credentials = self._load_credentials()
        if key in credentials:
            del credentials[key]
            self._save_credentials(credentials)
