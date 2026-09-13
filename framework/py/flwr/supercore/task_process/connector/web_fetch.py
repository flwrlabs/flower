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
"""Built-in web fetch connector."""

from __future__ import annotations

import codecs
import ipaddress
import os
import socket
from typing import cast
from urllib.parse import urljoin, urlparse

import requests

from flwr.supercore.task_process.usage import TaskUsageRecorder
from flwr.supercore.typing import JSONObject, JSONValue

WEB_FETCH_CONNECTOR_NAME = "web_fetch"
WEB_FETCH_ENDPOINT_ENV = "FLWR_WEB_FETCH_ENDPOINT"


def make_web_fetch_tool() -> JSONObject:
    """Return the web fetch function tool schema."""
    return {
        "type": "function",
        "name": WEB_FETCH_CONNECTOR_NAME,
        "description": "Fetch a web page and extract readable content.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    }


_MAX_RESPONSE_BYTES = 1024 * 1024
_TIMEOUT = 30.0
_MAX_REDIRECTS = 10
_READ_CHUNK_SIZE = 64 * 1024
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})
_PROXY_REQUEST_TIMEOUT = 60.0
_PROXY_WEB_FETCH_PROVIDER = "proxy"


class WebFetchProviderError(RuntimeError):
    """Error returned by the web fetch provider."""

    def __init__(
        self,
        *,
        code: str,
        detail: str,
        status_code: int | None = None,
    ) -> None:
        """Initialize the provider error."""
        self.code = code
        self.status_code = status_code
        self.detail = detail
        formatted_detail = detail
        if status_code is not None:
            formatted_detail = f"{status_code} {formatted_detail}"

        super().__init__(f"Web fetch provider request failed: {formatted_detail}")


def invoke_web_fetch_provider(
    url: str, *, usage_recorder: TaskUsageRecorder
) -> JSONObject:
    """Execute one web fetch request."""
    del usage_recorder
    if proxy_endpoint := os.getenv(WEB_FETCH_ENDPOINT_ENV, "").strip():
        return ProxyWebFetchProvider(proxy_endpoint).fetch(url)
    return _invoke_direct_web_fetch_provider(url)


class ProxyWebFetchProvider:
    """Proxy web fetch adapter."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint

    def fetch(self, url: str) -> JSONObject:
        """Execute one proxy web fetch request."""
        # This framework-side validation is best-effort. The proxy remains the
        # SSRF enforcement point because it validates DNS/IPs and redirects
        # immediately before fetching.
        url = _validate_url_syntax(url)
        try:
            response = requests.post(
                self._endpoint,
                json={"url": url},
                timeout=_PROXY_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:
            raise WebFetchProviderError(
                code="fetch_failed",
                detail=f"{_PROXY_WEB_FETCH_PROVIDER} web fetch request failed: {exc}",
            ) from exc
        if response.status_code >= 400:
            try:
                detail = str(cast(JSONValue, response.json()))
            except ValueError:
                detail = response.text
            raise WebFetchProviderError(
                code="http_error",
                status_code=response.status_code,
                detail=(
                    f"{_PROXY_WEB_FETCH_PROVIDER} web fetch request failed: {detail}"
                ),
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise WebFetchProviderError(
                code="invalid_response",
                detail=f"{_PROXY_WEB_FETCH_PROVIDER} web fetch returned invalid JSON.",
            ) from exc
        if not isinstance(payload, dict):
            raise WebFetchProviderError(
                code="invalid_response",
                detail=f"{_PROXY_WEB_FETCH_PROVIDER} web fetch returned invalid JSON.",
            )
        if not isinstance(payload.get("content"), str):
            raise WebFetchProviderError(
                code="invalid_response",
                detail=(
                    f"{_PROXY_WEB_FETCH_PROVIDER} web fetch response must contain "
                    "content."
                ),
            )
        return cast(JSONObject, payload)


def _invoke_direct_web_fetch_provider(url: str) -> JSONObject:
    """Fetch a URL and extract web page content with trafilatura.

    The direct provider validates every redirect target before requesting it and
    rejects local/private hosts before DNS-resolved requests are made.
    """
    url = _validate_url(url)
    response = _fetch_url(url)
    final_url = url
    try:
        final_url = _validate_url(response.url or url)
        body = _read_response_body(response)
        encoding = response.encoding or "utf-8"
        try:
            codecs.lookup(encoding)
        except LookupError:
            encoding = "utf-8"
        text = body.decode(encoding, errors="replace")
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

    content = trafilatura.extract(
        text,
        url=final_url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
    )
    if content is None:
        content = text

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
            # Follow redirects manually so every hop is validated before connect.
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
    """Return a URL allowed by the direct web-fetch guardrails.

    Only HTTP(S) URLs with a resolvable, globally routable host are accepted.
    Localhost names, private/reserved IP literals, and hostnames that resolve to
    non-public addresses are rejected before any direct request is made.
    """
    url, hostname = _validate_url_syntax_with_hostname(url)

    # Block localhost aliases explicitly; they should never reach DNS resolution.
    if hostname == "localhost" or hostname.endswith(".localhost"):
        raise WebFetchProviderError(
            code="blocked_url",
            detail="URL host is not allowed.",
        )

    try:
        ip_addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        try:
            # DNS can hide private targets behind public-looking hostnames.
            ip_addresses = {
                ipaddress.ip_address(addr[4][0])
                for addr in socket.getaddrinfo(
                    hostname,
                    None,
                    type=socket.SOCK_STREAM,
                )
            }
        except socket.gaierror as exc:
            raise WebFetchProviderError(
                code="fetch_failed",
                detail=f"Could not resolve URL host: {hostname}",
            ) from exc

    # Allow only public, globally routable targets to avoid SSRF-style fetches.
    if any(ip_addr.is_multicast or not ip_addr.is_global for ip_addr in ip_addresses):
        raise WebFetchProviderError(
            code="blocked_url",
            detail="URL host is not allowed.",
        )
    return url


def _validate_url_syntax(url: str) -> str:
    """Return a URL with a supported scheme and default port."""
    return _validate_url_syntax_with_hostname(url)[0]


def _validate_url_syntax_with_hostname(url: str) -> tuple[str, str]:
    """Return a URL and hostname with a supported scheme and default port."""
    url = url.strip()
    if not url:
        raise WebFetchProviderError(
            code="invalid_request",
            detail="URL must not be empty.",
        )

    try:
        # Parse once up front so malformed hosts are rejected before any network I/O.
        parsed = urlparse(url)
        hostname = parsed.hostname.rstrip(".").lower() if parsed.hostname else None
    except ValueError as exc:
        raise WebFetchProviderError(
            code="invalid_request",
            detail="URL must use the http or https scheme with a valid port.",
        ) from exc
    if parsed.scheme not in {"http", "https"} or hostname is None:
        raise WebFetchProviderError(
            code="invalid_request",
            detail="URL must use the http or https scheme.",
        )
    if parsed.username is not None or parsed.password is not None:
        raise WebFetchProviderError(
            code="invalid_request",
            detail="URL userinfo is not allowed.",
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise WebFetchProviderError(
            code="invalid_request",
            detail="URL must use the http or https scheme with a valid port.",
        ) from exc
    if port is None and parsed.netloc.endswith(":"):
        raise WebFetchProviderError(
            code="invalid_request",
            detail="URL must use the http or https scheme with a valid port.",
        )
    default_port = 80 if parsed.scheme == "http" else 443
    if port is not None and port != default_port:
        raise WebFetchProviderError(
            code="invalid_request",
            detail="URL must use the default port for its scheme.",
        )
    return url, hostname


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
