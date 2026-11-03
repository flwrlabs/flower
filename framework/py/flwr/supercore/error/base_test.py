# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for shared FlowerError serialization."""


import json

import pytest

from .base import ApiErrorCode, FlowerError
from .exceptions import EntitlementError


def test_flower_error_to_json_and_from_json() -> None:
    """Serialize and deserialize the canonical HTTP FlowerError payload."""
    err = FlowerError(
        ApiErrorCode.RUNTIME_VERSION_INCOMPATIBLE,
        "internal diagnostic message",
        public_details="public details",
    )

    payload = err.to_json("public message")
    parsed = FlowerError.from_json(json.dumps(payload))

    assert payload == {
        "code": ApiErrorCode.RUNTIME_VERSION_INCOMPATIBLE,
        "detail": "public message",
        "extra": "public details",
    }
    assert parsed is not None
    assert parsed.code == ApiErrorCode.RUNTIME_VERSION_INCOMPATIBLE
    assert parsed.message == "public message"
    assert parsed.public_details == "public details"




def test_flower_error_from_json_returns_base_error_for_subclass_call() -> None:
    """Use the base HTTP representation and deserialize into a base FlowerError."""
    err = EntitlementError(
        "internal diagnostic message",
        public_details="public details",
        entitlement_code=123,
    )

    payload = err.to_json("public message")
    parsed = EntitlementError.from_json(json.dumps(payload))

    assert payload == {
        "code": ApiErrorCode.ENTITLEMENT_ERROR,
        "detail": "public message",
        "extra": "public details",
        "entitlement_code": 123,
    }
    assert isinstance(parsed, FlowerError)
    assert not isinstance(parsed, EntitlementError)
    assert parsed.code == ApiErrorCode.ENTITLEMENT_ERROR
    assert parsed.message == "public message"
    assert parsed.public_details == "public details"


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not json",
        "[]",
        '{"public_message": "missing code"}',
        '{"code": 1}',
        '{"code": "1", "public_message": "wrong code type"}',
        '{"code": true, "public_message": "wrong code type"}',
        '{"code": 1, "public_message": "ok", "public_details": 1}',
        '{"detail": "missing code"}',
        '{"code": "1", "detail": "wrong code type"}',
        '{"code": true, "detail": "wrong code type"}',
        '{"code": 1, "detail": 1}',
        '{"code": 1, "detail": "ok", "extra": 1}',
    ],
)
def test_flower_error_from_json_returns_none_for_invalid_payloads(
    value: str | None,
) -> None:
    """Invalid public FlowerError payloads should not raise."""
    assert FlowerError.from_json(value) is None
