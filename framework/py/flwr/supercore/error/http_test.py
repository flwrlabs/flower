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
from .http import (
    INTERNAL_SERVER_ERROR_MESSAGE,
    BearerAuthenticationError,
    http_error_translator,
)


def _run_translator(exception: Exception) -> Response:
    """Run the HTTP error translator with a failing request handler."""

    async def call_next(_: Request[State]) -> Response:
        raise exception

    request = Request({"type": "http", "path": "/mock-route", "headers": []})
    return asyncio.run(http_error_translator(request, call_next))


def _assert_json_response(
    response: Response, status_code: int, body: dict[str, object]
) -> None:
    """Assert the complete JSON response contract."""
    assert response.status_code == status_code
    assert response.headers["content-type"] == "application/json"
    assert json.loads(bytes(response.body)) == body


def test_http_error_translator_mapped_flower_error() -> None:
    """Include the public message and numeric code for a mapped FlowerError."""
    error_code = ApiErrorCode.NO_FEDERATION_MANAGEMENT_SUPPORT
    response = _run_translator(FlowerError(error_code, "internal diagnostic message"))

    spec = API_ERROR_MAP[error_code]
    _assert_json_response(
        response,
        spec.http_status_code,
        {"detail": spec.public_message, "code": error_code.value},
    )
    assert b"internal diagnostic message" not in response.body


def test_http_error_translator_includes_public_details() -> None:
    """Expose public details separately from the catalog message."""
    error_code = ApiErrorCode.INVALID_FEDERATION_NAME
    public_details = "The requested federation name is invalid."
    response = _run_translator(
        FlowerError(
            error_code,
            "internal diagnostic message",
            public_details=public_details,
        )
    )

    spec = API_ERROR_MAP[error_code]
    _assert_json_response(
        response,
        spec.http_status_code,
        {
            "detail": spec.public_message,
            "code": error_code.value,
            "extra": public_details,
        },
    )
    assert b"internal diagnostic message" not in response.body


def test_http_error_translator_includes_empty_public_details() -> None:
    """Include an empty public-details string in a FlowerError response."""
    error_code = ApiErrorCode.INVALID_FEDERATION_NAME
    response = _run_translator(
        FlowerError(
            error_code,
            "internal diagnostic message",
            public_details="",
        )
    )

    spec = API_ERROR_MAP[error_code]
    _assert_json_response(
        response,
        spec.http_status_code,
        {"detail": spec.public_message, "code": error_code.value, "extra": ""},
    )


def test_http_error_translator_adds_bearer_authentication_challenge() -> None:
    """Challenge clients when HTTP account authentication fails."""
    response = _run_translator(
        FlowerError(
            ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED,
            "internal authentication failure",
        )
    )

    spec = API_ERROR_MAP[ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED]
    _assert_json_response(
        response,
        spec.http_status_code,
        {
            "detail": spec.public_message,
            "code": ApiErrorCode.ACCOUNT_AUTHENTICATION_FAILED.value,
        },
    )
    assert response.headers["www-authenticate"] == "Bearer"
    assert b"internal authentication failure" not in response.body


def test_http_error_translator_unmapped_flower_error() -> None:
    """Translate an unmapped FlowerError into INTERNAL."""
    response = _run_translator(FlowerError(999, "internal diagnostic message"))

    _assert_json_response(
        response,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        {"detail": INTERNAL_SERVER_ERROR_MESSAGE},
    )
    assert b"internal diagnostic message" not in response.body


def test_http_error_translator_entitlement_error() -> None:
    """Expose entitlement details and its application-level code."""
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
    _assert_json_response(
        response,
        spec.http_status_code,
        {
            "detail": spec.public_message,
            "code": ApiErrorCode.ENTITLEMENT_ERROR,
            "extra": error_message,
            "entitlement_code": entitlement_code,
        },
    )
    assert b"internal diagnostic message" not in response.body


def test_http_error_translator_http_exception() -> None:
    """Translate an HTTPException into a response."""
    http_error = HTTPException(
        status_code=status.HTTP_418_IM_A_TEAPOT,
        detail={"message": "short and stout"},
    )

    response = _run_translator(http_error)

    _assert_json_response(
        response,
        status.HTTP_418_IM_A_TEAPOT,
        {"detail": {"message": "short and stout"}},
    )


def test_http_error_translator_bearer_authentication_error() -> None:
    """Translate Bearer authentication failures using FastAPI's error contract."""
    response = _run_translator(BearerAuthenticationError())

    _assert_json_response(
        response,
        status.HTTP_401_UNAUTHORIZED,
        {"detail": "Not authenticated"},
    )
    assert response.headers["www-authenticate"] == "Bearer"


def test_http_error_translator_unexpected_error() -> None:
    """Translate unexpected errors into INTERNAL."""
    response = _run_translator(RuntimeError("unexpected failure"))

    _assert_json_response(
        response,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        {"detail": INTERNAL_SERVER_ERROR_MESSAGE},
    )
    assert b"unexpected failure" not in response.body
