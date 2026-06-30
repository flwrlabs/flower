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
"""Tests for HTTP error translation utilities."""


import json

import pytest
from fastapi import HTTPException, status

from .base import ApiErrorCode, FlowerError
from .catalog import API_ERROR_MAP
from .exceptions import EntitlementError
from .http import INTERNAL_SERVER_ERROR_MESSAGE, http_error_translator


def test_http_error_translator_mapped_flower_error() -> None:
    """Translate a mapped FlowerError into its configured HTTP contract."""
    with pytest.raises(HTTPException) as exc:
        with http_error_translator("MockRoute"):
            raise FlowerError(
                ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT,
                "internal diagnostic message",
            )

    spec = API_ERROR_MAP[ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT]
    assert exc.value.status_code == spec.http_status_code
    assert isinstance(exc.value.detail, str)
    assert json.loads(exc.value.detail) == {
        "code": ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT,
        "public_message": spec.public_message,
        "public_details": None,
    }


def test_http_error_translator_unmapped_flower_error() -> None:
    """Translate an unmapped FlowerError into INTERNAL."""
    with pytest.raises(HTTPException) as exc:
        with http_error_translator("MockRoute"):
            raise FlowerError(999, "internal diagnostic message")

    assert exc.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert isinstance(exc.value.detail, str)
    assert json.loads(exc.value.detail) == {
        "code": 999,
        "public_message": INTERNAL_SERVER_ERROR_MESSAGE,
        "public_details": None,
    }


def test_http_error_translator_entitlement_error_preserves_error_message() -> None:
    """Keep entitlement details in the translated payload."""
    error_message = "Entitlement check failed: plan does not allow this action."
    entitlement_code = 101

    with pytest.raises(HTTPException) as exc:
        with http_error_translator("MockRoute"):
            raise EntitlementError(
                "internal diagnostic message",
                public_details=error_message,
                entitlement_code=entitlement_code,
            )

    spec = API_ERROR_MAP[ApiErrorCode.ENTITLEMENT_ERROR]
    assert exc.value.status_code == spec.http_status_code
    assert isinstance(exc.value.detail, str)
    assert json.loads(exc.value.detail) == {
        "code": ApiErrorCode.ENTITLEMENT_ERROR,
        "public_message": spec.public_message,
        "public_details": error_message,
        "entitlement_code": entitlement_code,
    }


def test_http_error_translator_http_exception() -> None:
    """Allow existing HTTPException to propagate unmodified."""
    http_error = HTTPException(
        status_code=status.HTTP_418_IM_A_TEAPOT,
        detail={"message": "short and stout"},
    )

    with pytest.raises(HTTPException) as exc:
        with http_error_translator("MockRoute"):
            raise http_error

    assert exc.value is http_error


def test_http_error_translator_unexpected_error() -> None:
    """Translate unexpected errors into INTERNAL."""
    with pytest.raises(HTTPException) as exc:
        with http_error_translator("MockRoute"):
            raise RuntimeError("unexpected failure")

    assert exc.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert exc.value.detail == INTERNAL_SERVER_ERROR_MESSAGE
