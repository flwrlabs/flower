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
"""Trafilatura-backed web fetch provider."""

from __future__ import annotations

import ipaddress
import json
import socket
from urllib.parse import urljoin, urlparse

import requests

from flwr.supercore.typing import JSONObject, JSONValue

_MAX_RESPONSE_BYTES = 1024 * 1024
_TIMEOUT = 30.0
_MAX_REDIRECTS = 10
_READ_CHUNK_SIZE = 64 * 1024
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})


class WebFetchProviderError(RuntimeError):
    """Error returned by the web fetch provider."""

    def __init__(
        self,
        *,
        code: str,
        detail: JSONValue,
        status_code: int | None = None,
    ) -> None:
        """Initialize the provider error."""
        self.code = code
        self.status_code = status_code
        self.detail = detail
        if isinstance(detail, str):
            formatted_detail = detail
        else:
            formatted_detail = json.dumps(detail, separators=(",", ":"))
        if status_code is not None:
            formatted_detail = f"{status_code} {formatted_detail}"

        super().__init__(f"Web fetch provider request failed: {formatted_detail}")


def invoke_web_fetch_provider(url: str) -> JSONObject:
    """Fetch a URL and extract web page content with trafilatura.

    The provider validates every redirect target before requesting it and rejects
    local/private hosts before DNS-resolved requests are made.
    """
    url = _validate_url(url)
    response = _fetch_url(url)
    final_url = url
    try:
        final_url = _validate_url(response.url or url)
        body = _read_response_body(response)
        text = body.decode(
            response.encoding or response.apparent_encoding or "utf-8",
            errors="replace",
        )
        if response.status_code >= 400:
            raise WebFetchProviderError(
                code="http_error",
                status_code=response.status_code,
                detail=text,
            )
    finally:
        response.close()

    try:
        import trafilatura  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise WebFetchProviderError(
            code="missing_dependency",
            detail="Install the 'agent' extra to use the web fetch provider.",
        ) from exc

    content = (
        trafilatura.extract(
            text,
            url=final_url,
            output_format="markdown",
            include_comments=False,
            include_tables=True,
        )
        or text
    )

    return {
        "object": "web_fetch.response",
        "status": "completed",
        "url": url,
        "final_url": final_url,
        "status_code": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "content": content,
    }


def _fetch_url(url: str) -> requests.Response:
    """Fetch a URL while validating each redirect target before following it."""
    current_url = url
    for redirect_count in range(_MAX_REDIRECTS + 1):
        current_url = _validate_url(current_url)
        try:
            response = requests.get(
                current_url,
                timeout=_TIMEOUT,
                stream=True,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise WebFetchProviderError(
                code="fetch_failed",
                detail=str(exc),
            ) from exc

        if response.status_code not in _REDIRECT_STATUS_CODES:
            return response

        location = response.headers.get("Location")
        if not location:
            return response

        response_url = response.url or current_url
        response.close()
        if redirect_count == _MAX_REDIRECTS:
            raise WebFetchProviderError(
                code="too_many_redirects",
                detail=f"Web fetch exceeded {_MAX_REDIRECTS} redirects.",
            )
        current_url = urljoin(response_url, location)

    raise RuntimeError("This line should never be reached.")


def _validate_url(url: str) -> str:
    """Return a validated URL."""
    url = url.strip()
    if not url:
        raise WebFetchProviderError(
            code="invalid_request",
            detail="URL must not be empty.",
        )

    parsed = urlparse(url)
    hostname = parsed.hostname
    if parsed.scheme not in {"http", "https"} or hostname is None:
        raise WebFetchProviderError(
            code="invalid_request",
            detail="URL must use the http or https scheme.",
        )

    hostname = hostname.rstrip(".").lower()
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise WebFetchProviderError(
            code="blocked_url",
            detail="URL host is not allowed.",
        )

    try:
        ip_addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            ip_addresses = [
                ipaddress.ip_address(addr[4][0])
                for addr in socket.getaddrinfo(
                    hostname,
                    None,
                    type=socket.SOCK_STREAM,
                )
            ]
        except socket.gaierror as exc:
            raise WebFetchProviderError(
                code="fetch_failed",
                detail=f"Could not resolve URL host: {hostname}",
            ) from exc

    if any(
        ip_address.is_private
        or ip_address.is_loopback
        or ip_address.is_link_local
        or ip_address.is_multicast
        or ip_address.is_reserved
        or ip_address.is_unspecified
        for ip_address in ip_addresses
    ):
        raise WebFetchProviderError(
            code="blocked_url",
            detail="URL host is not allowed.",
        )
    return url


def _read_response_body(response: requests.Response) -> bytes:
    """Read a bounded response body."""
    body = bytearray()
    try:
        chunks = response.iter_content(chunk_size=_READ_CHUNK_SIZE)
        for chunk in chunks:
            if not chunk:
                continue
            if len(body) + len(chunk) > _MAX_RESPONSE_BYTES:
                raise WebFetchProviderError(
                    code="response_too_large",
                    status_code=response.status_code,
                    detail=(
                        f"Response body exceeds maximum size ({_MAX_RESPONSE_BYTES})."
                    ),
                )
            body.extend(chunk)
    except requests.RequestException as exc:
        raise WebFetchProviderError(
            code="fetch_failed",
            detail=str(exc),
        ) from exc
    return bytes(body)
