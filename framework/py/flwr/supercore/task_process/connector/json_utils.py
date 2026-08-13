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
"""Validation helpers shared by account-scoped connectors."""


from collections.abc import Callable
from typing import cast

from flwr.supercore.typing import JSONObject

from .errors import ConnectorArgumentError

ErrorFactory = Callable[[str], Exception]


def require_string(value: object, provider: str, name: str) -> str:
    """Validate and normalize a required connector argument."""
    if not isinstance(value, str) or not value.strip():
        raise ConnectorArgumentError(f"{provider} {name} must be a non-empty string.")
    return value.strip()


def optional_string(value: object, provider: str, name: str) -> str | None:
    """Validate and normalize an optional connector argument."""
    if value is None:
        return None
    return require_string(value, provider, name)


def require_int_range(
    value: object,
    provider: str,
    name: str,
    *,
    minimum: int = 1,
    maximum: int,
) -> int:
    """Validate an integer connector argument with inclusive bounds."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConnectorArgumentError(f"{provider} {name} must be an integer.")
    if value < minimum or value > maximum:
        raise ConnectorArgumentError(
            f"{provider} {name} must be between {minimum} and {maximum}."
        )
    return value


def require_bool(value: object, provider: str, name: str) -> bool:
    """Validate a boolean connector argument."""
    if not isinstance(value, bool):
        raise ConnectorArgumentError(f"{provider} {name} must be a boolean.")
    return value


def object_field(payload: JSONObject, key: str, *, error: ErrorFactory) -> JSONObject:
    """Read a required JSON object from a provider response."""
    value = payload.get(key)
    if not isinstance(value, dict):
        raise error("invalid_response")
    return value


def object_list_field(
    payload: JSONObject, key: str, *, error: ErrorFactory
) -> list[JSONObject]:
    """Read a required list of JSON objects from a provider response."""
    value = payload.get(key)
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise error("invalid_response")
    return cast(list[JSONObject], value)


def required_string_field(payload: JSONObject, key: str, *, error: ErrorFactory) -> str:
    """Read a required non-empty string from a provider response."""
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise error("invalid_response")
    return value


def string_field(payload: JSONObject, key: str) -> str:
    """Return a response string field or an empty string."""
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def integer_field(payload: JSONObject, key: str) -> int | None:
    """Return a response integer field or None."""
    value = payload.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None
