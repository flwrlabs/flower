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
"""Tests for CredentialStore."""


import tempfile
import unittest
from abc import abstractmethod
from pathlib import Path
from unittest.mock import patch

import yaml

from .credential_store import CredentialStore
from .file_credential_store import FileCredentialStore


class CredentialStoreTest(unittest.TestCase):
    """Test all CredentialStore implementations."""

    # This is set to True in each child class
    __test__ = False

    @abstractmethod
    def credential_store_factory(self) -> CredentialStore:
        """Provide CredentialStore implementation to test."""
        raise NotImplementedError()

    def test_get_non_existent_key(self) -> None:
        """Test get method with a non-existent key."""
        # Prepare
        store = self.credential_store_factory()
        key = "non_existent_key"

        # Execute
        retrieved_value = store.get(key)

        # Assert
        self.assertIsNone(retrieved_value)

    def test_set_and_get(self) -> None:
        """Test set and get methods."""
        # Prepare
        store = self.credential_store_factory()
        key = "test_key"
        value = b"test_value"

        # Execute
        store.set(key, value)
        retrieved_value = store.get(key)

        # Assert
        self.assertEqual(value, retrieved_value)

    def test_set_overwrite(self) -> None:
        """Test set method with an existing key."""
        # Prepare
        store = self.credential_store_factory()
        key = "test_key"
        value1 = b"test_value1"
        value2 = b"test_value2"

        # Execute
        store.set(key, value1)
        store.set(key, value2)
        retrieved_value = store.get(key)

        # Assert
        self.assertEqual(value2, retrieved_value)

    def test_delete(self) -> None:
        """Test delete method."""
        # Prepare
        store = self.credential_store_factory()
        key = "test_key"
        value = b"test_value"
        store.set(key, value)

        # Execute
        store.delete(key)
        retrieved_value = store.get(key)

        # Assert
        self.assertIsNone(retrieved_value)

    def test_delete_non_existent_key(self) -> None:
        """Test delete method with a non-existent key."""
        # Prepare
        store = self.credential_store_factory()
        key = "non_existent_key"

        # Execute
        store.delete(key)
        # No exception should be raised

    def test_multiple_keys(self) -> None:
        """Test handling multiple keys."""
        # Prepare
        store = self.credential_store_factory()
        key1 = "key1"
        value1 = b"value1"
        key2 = "key2"
        value2 = b"value2"

        # Execute
        store.set(key1, value1)
        store.set(key2, value2)

        # Assert
        self.assertEqual(value1, store.get(key1))
        self.assertEqual(value2, store.get(key2))

        # Execute: Delete one key
        store.delete(key1)

        # Assert: Only key2 should remain
        self.assertIsNone(store.get(key1))
        self.assertEqual(value2, store.get(key2))

    def test_empty_value(self) -> None:
        """Test storing empty bytes."""
        # Prepare
        store = self.credential_store_factory()
        key = "empty_key"
        value = b""

        # Execute
        store.set(key, value)
        retrieved_value = store.get(key)

        # Assert
        self.assertEqual(value, retrieved_value)

    def test_binary_value(self) -> None:
        """Test storing binary data."""
        # Prepare
        store = self.credential_store_factory()
        key = "binary_key"
        value = bytes([0, 1, 2, 255, 254, 253])

        # Execute
        store.set(key, value)
        retrieved_value = store.get(key)

        # Assert
        self.assertEqual(value, retrieved_value)


class FileCredentialStoreTest(CredentialStoreTest):
    """Test FileCredentialStore implementation."""

    __test__ = True

    def setUp(self) -> None:
        """Set up the test case."""
        self.temp_file = tempfile.NamedTemporaryFile(  # pylint: disable=R1732
            delete=False, suffix=".yaml"
        )

    def tearDown(self) -> None:
        """Tear down the test case."""
        self.temp_file.close()

    def credential_store_factory(self) -> CredentialStore:
        """Return FileCredentialStore."""
        return FileCredentialStore(Path(self.temp_file.name))

    def test_get_with_invalid_yaml_removes_file(self) -> None:
        """Test invalid YAML is removed and treated as empty credentials."""
        with Path(self.temp_file.name).open("w", encoding="utf-8") as f:
            f.write("test_key: dmFsdWU=\nbroken_line_without_colon")

        store = self.credential_store_factory()
        self.assertIsNone(store.get("test_key"))
        self.assertFalse(Path(self.temp_file.name).exists())

    def test_set_does_not_raise_on_write_failure(self) -> None:
        """Test write failures are tolerated without raising."""
        store = self.credential_store_factory()

        with patch(
            "flwr.supercore.credential_store.file_credential_store.yaml.safe_dump",
            side_effect=yaml.YAMLError("failed to dump"),
        ):
            store.set("test_key", b"test_value")

        self.assertIsNone(store.get("test_key"))

    def test_get_with_invalid_base64_removes_file(self) -> None:
        """Test malformed encoded values are treated as corruption."""
        with Path(self.temp_file.name).open("w", encoding="utf-8") as f:
            f.write('test_key: "not_base64_encoded_value"')

        store = self.credential_store_factory()
        self.assertIsNone(store.get("test_key"))
        self.assertFalse(Path(self.temp_file.name).exists())
