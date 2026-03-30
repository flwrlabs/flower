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
"""Abstract base classes for credential store."""


from abc import ABC, abstractmethod


class CredentialStore(ABC):
    """Abstract base class for credential store."""

    @abstractmethod
    def set(self, key: str, value: bytes) -> None:
        """Set a credential in the store."""

    @abstractmethod
    def get(self, key: str) -> bytes | None:
        """Get a credential from the store."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete a credential from the store."""
