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


import asyncio
import json

from fastapi import HTTPException, Request, Response, status
from starlette.datastructures import State

from .base import ApiErrorCode, FlowerError
from .catalog import API_ERROR_MAP
from .exceptions import EntitlementError
from .http import INTERNAL_SERVER_ERROR_MESSAGE, http_error_translator


def _run_translator(exception: Exception) -> Response:
    """Run the HTTP error translator with a failing request handler."""

    async def call_next(_: Request[State]) -> Response:
        raise exception

    request = Request({"type": "http", "path": "/mock-route", "headers": []})
    return asyncio.run(http_error_translator(request, call_next))


def test_http_error_translator_mapped_flower_error() -> None:
    """Translate a mapped FlowerError into its configured HTTP contract."""
    response = _run_translator(
        FlowerError(
            ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT,
            "internal diagnostic message",
        )
    )

    spec = API_ERROR_MAP[ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT]
    assert response.status_code == spec.http_status_code
    assert json.loads(response.body) == {
        "code": ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT,
        "public_message": spec.public_message,
        "public_details": None,
    }


def test_http_error_translator_unmapped_flower_error() -> None:
    """Translate an unmapped FlowerError into INTERNAL."""
    response = _run_translator(FlowerError(999, "internal diagnostic message"))

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert json.loads(response.body) == {
        "code": 999,
        "public_message": INTERNAL_SERVER_ERROR_MESSAGE,
        "public_details": None,
    }


def test_http_error_translator_entitlement_error_preserves_error_message() -> None:
    """Keep entitlement details in the translated payload."""
    error_message = "Entitlement check failed: plan does not allow this action."
    entitlement_code = 101

    response = _run_translator(
        EntitlementError(
            "internal diagnostic message",
            public_details=error_message,
            entitlement_code=entitlement_code,
        )
    )

    spec = API_ERROR_MAP[ApiErrorCode.ENTITLEMENT_ERROR]
    assert response.status_code == spec.http_status_code
    assert json.loads(response.body) == {
        "code": ApiErrorCode.ENTITLEMENT_ERROR,
        "public_message": spec.public_message,
        "public_details": error_message,
        "entitlement_code": entitlement_code,
    }


def test_http_error_translator_http_exception() -> None:
    """Translate an HTTPException into a response."""
    http_error = HTTPException(
        status_code=status.HTTP_418_IM_A_TEAPOT,
        detail={"message": "short and stout"},
    )

    response = _run_translator(http_error)

    assert response.status_code == status.HTTP_418_IM_A_TEAPOT
    assert json.loads(response.body) == {"detail": {"message": "short and stout"}}


def test_http_error_translator_unexpected_error() -> None:
    """Translate unexpected errors into INTERNAL."""
    response = _run_translator(RuntimeError("unexpected failure"))

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert response.body == INTERNAL_SERVER_ERROR_MESSAGE.encode()
