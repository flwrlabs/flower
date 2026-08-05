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
"""Secret-safe JSON HTTP handling for connectors."""


from collections.abc import Callable, Mapping
from typing import cast

import requests

from flwr.supercore.typing import JSONObject

ConnectorErrorFactory = Callable[[str, int | None], RuntimeError]
HttpErrorCode = Callable[[requests.Response], str]


class ConnectorApiError(RuntimeError):
    """Base class for secret-safe connector API failures."""

    provider: str

    def __init__(self, code: str, status_code: int | None = None) -> None:
        self.code = code
        self.status_code = status_code
        detail = code if status_code is None else f"{code} ({status_code})"
        super().__init__(f"{self.provider} API request failed: {detail}.")


# pylint: disable-next=too-many-arguments
def request_json_object(
    method: str,
    url: str,
    *,
    error: ConnectorErrorFactory,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    json: JSONObject | None = None,
    timeout: float = 30.0,
    http_error_code: HttpErrorCode | None = None,
) -> JSONObject:
    """Send one request and return its JSON object response."""
    try:
        response = requests.request(
            method,
            url,
            headers=headers,
            params=params,
            json=json,
            timeout=timeout,
        )
    except requests.RequestException:
        raise error("request_failed", None) from None
    if response.status_code >= 400:
        code = http_error_code(response) if http_error_code else "http_error"
        raise error(code, response.status_code)
    try:
        payload = response.json()
    except ValueError:
        raise error("invalid_response", None) from None
    if not isinstance(payload, dict):
        raise error("invalid_response", None)
    return cast(JSONObject, payload)
