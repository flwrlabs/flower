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
"""HTTP-specific translation utilities for Flower API errors."""


from logging import ERROR

from fastapi import HTTPException, Request, Response, status
from fastapi.exception_handlers import http_exception_handler
from starlette.datastructures import State
from starlette.middleware.base import RequestResponseEndpoint

from flwr.common.logger import log

from .base import FlowerError
from .catalog import API_ERROR_MAP

INTERNAL_SERVER_ERROR_MESSAGE = "Internal server error."


async def http_error_translator(
    request: Request[State], call_next: RequestResponseEndpoint
) -> Response:
    """Translate FlowerError into a sanitized HTTP response."""
    try:
        return await call_next(request)
    except FlowerError as err:
        try:
            error_spec = API_ERROR_MAP[err.code]
            http_status = error_spec.http_status_code
            public_message = error_spec.public_message
        except (ValueError, KeyError):
            http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
            public_message = INTERNAL_SERVER_ERROR_MESSAGE

        # Log error as is
        msg = f"[{request.url.path}][ApiError:{err.code}] {err.message}"
        log(ERROR, msg)
        # Return sanitized error to client
        return Response(
            status_code=http_status,
            content=err.to_json(public_message),
            media_type="application/json",
        )
    except HTTPException as err:
        return await http_exception_handler(request, err)
    except Exception as err:  # pylint: disable=broad-exception-caught
        # Log unexpected exceptions and translate into INTERNAL
        msg = f"[{request.url.path}][UnexpectedError:{type(err).__name__}] {err}"
        log(ERROR, msg)
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=INTERNAL_SERVER_ERROR_MESSAGE,
        )
