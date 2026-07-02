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


from collections.abc import Iterator
from contextlib import contextmanager
from logging import ERROR

from fastapi import HTTPException, status

from flwr.common.logger import log

from .base import FlowerError
from .catalog import API_ERROR_MAP

INTERNAL_SERVER_ERROR_MESSAGE = "Internal server error."


@contextmanager
def http_error_translator(route_name: str) -> Iterator[None]:
    """Translate FlowerError into a sanitized HTTP error."""
    try:
        yield
    except FlowerError as err:
        try:
            error_spec = API_ERROR_MAP[err.code]
            http_status = error_spec.http_status_code
            public_message = error_spec.public_message
        except (ValueError, KeyError):
            http_status = status.HTTP_500_INTERNAL_SERVER_ERROR
            public_message = INTERNAL_SERVER_ERROR_MESSAGE

        # Log error as is
        msg = f"[{route_name}][ApiError:{err.code}] {err.message}"
        log(ERROR, msg)
        # Return sanitized error to client
        raise HTTPException(
            status_code=http_status,
            detail=err.to_json(public_message),
        ) from None
    except HTTPException:
        raise
    except Exception as err:
        # Log unexpected exceptions and translate into INTERNAL
        msg = f"[{route_name}][UnexpectedError:{type(err).__name__}] {err}"
        log(ERROR, msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=INTERNAL_SERVER_ERROR_MESSAGE,
        ) from None
