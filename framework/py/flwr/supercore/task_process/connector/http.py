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

    provider = "Connector"

    def __init__(
        self,
        code: str,
        status_code: int | None = None,
        *,
        provider: str | None = None,
    ) -> None:
        self.code = code
        self.status_code = status_code
        detail = code if status_code is None else f"{code} ({status_code})"
        super().__init__(f"{provider or self.provider} API request failed: {detail}.")


class ConnectorHttpClient:
    """Make authenticated, secret-safe requests to one provider API."""

    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        credentials: JSONObject,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not base_url.startswith("https://"):
            raise ValueError("Connector API base URL must use HTTPS.")
        token = credentials.get("access_token")
        self._provider = provider
        self._base_url = base_url.rstrip("/")
        self._token = token if isinstance(token, str) and token else None
        self._headers = dict(headers or {})

    # pylint: disable-next=too-many-arguments
    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json: JSONObject | None = None,
        headers: Mapping[str, str] | None = None,
        error: ConnectorErrorFactory | None = None,
        http_error_code: HttpErrorCode | None = None,
    ) -> JSONObject:
        """Request one provider-relative JSON object endpoint."""
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Connector API path must be provider-relative.")
        error_factory = error or self._error
        if self._token is None:
            raise error_factory("invalid_credentials", None)
        return request_json_object(
            method,
            f"{self._base_url}{path}",
            error=error_factory,
            headers={
                **self._headers,
                **(headers or {}),
                "Authorization": f"Bearer {self._token}",
            },
            params=params,
            json=json,
            http_error_code=http_error_code,
        )

    def _error(self, code: str, status_code: int | None) -> ConnectorApiError:
        """Build a provider-labelled, secret-safe error."""
        return ConnectorApiError(code, status_code, provider=self._provider)


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
