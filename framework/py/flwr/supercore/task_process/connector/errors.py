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
"""Secret-safe connector errors that may be exposed to an agent."""

from flwr.supercore.typing import JSONObject


class PublicConnectorError(Exception):
    """Base class for connector failures that are safe to expose."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retryable = retryable

    def to_json(self) -> JSONObject:
        """Return the model-facing structured error."""
        error: JSONObject = {
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
        }
        if self.status_code is not None:
            error["status_code"] = self.status_code
        return error


class ConnectorArgumentError(PublicConnectorError, ValueError):
    """An invalid connector argument that an agent can correct."""

    def __init__(self, message: str) -> None:
        super().__init__("invalid_arguments", message)
