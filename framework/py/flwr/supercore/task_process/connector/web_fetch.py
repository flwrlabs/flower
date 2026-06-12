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
import os
import socket
from urllib.parse import urljoin, urlparse

import requests
import trafilatura

from flwr.supercore.typing import JSONObject, JSONValue

DEFAULT_WEB_FETCH_MAX_LENGTH = 5000
DEFAULT_WEB_FETCH_MAX_RESPONSE_BYTES = 1024 * 1024
DEFAULT_WEB_FETCH_TIMEOUT = 30.0
DEFAULT_WEB_FETCH_USER_AGENT = "FlowerWebFetch/1.0"
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
        message: str = "Web fetch provider request failed",
    ) -> None:
        """Initialize the provider error."""
        self.code = code
        self.status_code = status_code
        self.detail = detail
        if isinstance(detail, str):
            formatted_detail = detail
        else:
            formatted_detail = json.dumps(detail, separators=(",", ":"))
        if status_code is None:
            super().__init__(f"{message}: {formatted_detail}")
        else:
            super().__init__(f"{message}: {status_code} {formatted_detail}")


def invoke_web_fetch_provider(request: JSONObject) -> JSONObject:
    """Fetch a URL and extract web page content with trafilatura.

    Control flow:
    1. Validate request and provider environment settings.
    2. Fetch the URL with bounded timeout and response size.
    3. Return raw text or trafilatura-extracted markdown, chunked by character index.
    """
    request_payload = dict(request)

    url = _get_required_url(request_payload)
    max_length = _get_int(
        request_payload,
        "max_length",
        _get_env_int("FLWR_WEB_FETCH_MAX_LENGTH", DEFAULT_WEB_FETCH_MAX_LENGTH),
    )
    start_index = _get_int(request_payload, "start_index", 0)
    raw = _get_bool(request_payload, "raw", False)

    if max_length <= 0:
        raise WebFetchProviderError(
            code="invalid_request",
            detail="Field 'max_length' must be a positive integer.",
        )
    if start_index < 0:
        raise WebFetchProviderError(
            code="invalid_request",
            detail="Field 'start_index' must be a non-negative integer.",
        )

    timeout = _get_env_float("FLWR_WEB_FETCH_TIMEOUT", DEFAULT_WEB_FETCH_TIMEOUT)
    max_response_bytes = _get_env_int(
        "FLWR_WEB_FETCH_MAX_RESPONSE_BYTES",
        DEFAULT_WEB_FETCH_MAX_RESPONSE_BYTES,
    )
    user_agent = os.getenv("FLWR_WEB_FETCH_USER_AGENT", "").strip()
    if not user_agent:
        user_agent = DEFAULT_WEB_FETCH_USER_AGENT

    response = _fetch_url(
        url=url,
        headers={"User-Agent": user_agent},
        timeout=timeout,
    )

    final_url = url
    try:
        final_url = _get_required_url({"url": response.url or url})
        body = _read_response_body(response, max_response_bytes)
        text = _decode_response_body(response, body)
        if response.status_code >= 400:
            raise WebFetchProviderError(
                code="http_error",
                status_code=response.status_code,
                detail=text,
            )
    finally:
        response.close()

    content = text if raw else _extract_markdown(text, final_url)
    if not content:
        content = text

    content_slice, truncated, next_start_index = _slice_content(
        content,
        start_index,
        max_length,
    )

    return {
        "object": "web_fetch.response",
        "status": "completed",
        "url": url,
        "final_url": final_url,
        "status_code": response.status_code,
        "content_type": response.headers.get("Content-Type", ""),
        "content": content_slice,
        "start_index": start_index,
        "truncated": truncated,
        "next_start_index": next_start_index,
    }


def _fetch_url(
    *,
    url: str,
    headers: dict[str, str],
    timeout: float,
) -> requests.Response:
    """Fetch a URL while validating each redirect target before following it."""
    current_url = url

    for redirect_count in range(_MAX_REDIRECTS + 1):
        current_url = _get_required_url({"url": current_url})
        response = _request_once(url=current_url, headers=headers, timeout=timeout)

        response_url = response.url or current_url
        if response.status_code not in _REDIRECT_STATUS_CODES:
            return response

        location = response.headers.get("Location")
        if not location:
            return response

        if redirect_count == _MAX_REDIRECTS:
            response.close()
            raise WebFetchProviderError(
                code="too_many_redirects",
                detail=f"Web fetch exceeded {_MAX_REDIRECTS} redirects.",
            )

        next_url = urljoin(response_url, location)
        response.close()
        current_url = next_url

    raise RuntimeError("This line should never be reached.")


def _get_required_url(request: JSONObject) -> str:
    """Return the validated URL from a provider request."""
    raw_url = request.get("url")
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise WebFetchProviderError(
            code="invalid_request",
            detail="Request requires a non-empty string field 'url'.",
        )

    url = raw_url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise WebFetchProviderError(
            code="invalid_request",
            detail="URL must use the http or https scheme.",
        )

    hostname = parsed.hostname
    if hostname is None or _is_blocked_host(hostname):
        raise WebFetchProviderError(
            code="blocked_url",
            detail="URL host is not allowed.",
        )
    return url


def _request_once(
    *,
    url: str,
    headers: dict[str, str],
    timeout: float,
) -> requests.Response:
    """Request one URL without following redirects."""
    try:
        return requests.get(
            url,
            headers=headers,
            timeout=timeout,
            stream=True,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise WebFetchProviderError(
            code="fetch_failed",
            detail=str(exc),
        ) from exc


def _get_bool(request: JSONObject, field: str, default: bool) -> bool:
    """Return a boolean request field."""
    value = request.get(field, default)
    if not isinstance(value, bool):
        raise WebFetchProviderError(
            code="invalid_request",
            detail=f"Field '{field}' must be a boolean.",
        )
    return value


def _get_int(request: JSONObject, field: str, default: int) -> int:
    """Return an integer request field."""
    value = request.get(field, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise WebFetchProviderError(
            code="invalid_request",
            detail=f"Field '{field}' must be an integer.",
        )
    return value


def _get_env_float(name: str, default: float) -> float:
    """Return a positive float from the environment."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return max(1.0, value)


def _get_env_int(name: str, default: int) -> int:
    """Return a positive integer from the environment."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _is_blocked_host(hostname: str) -> bool:
    """Return whether a hostname is blocked by local and DNS checks."""
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True

    try:
        ip_address = ipaddress.ip_address(normalized)
    except ValueError:
        ip_address = None

    if ip_address is not None:
        return _is_blocked_ip_address(ip_address)

    try:
        address_info = socket.getaddrinfo(
            normalized,
            None,
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise WebFetchProviderError(
            code="fetch_failed",
            detail=f"Could not resolve URL host: {hostname}",
        ) from exc

    for addr in address_info:
        sockaddr = addr[4]
        try:
            resolved_ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return True
        if _is_blocked_ip_address(resolved_ip):
            return True

    return False


def _is_blocked_ip_address(
    ip_address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Return whether an IP address is blocked for web fetches."""
    return (
        ip_address.is_private
        or ip_address.is_loopback
        or ip_address.is_link_local
        or ip_address.is_multicast
        or ip_address.is_reserved
        or ip_address.is_unspecified
    )


def _read_response_body(response: requests.Response, max_response_bytes: int) -> bytes:
    """Read a bounded response body."""
    body = bytearray()
    try:
        chunks = response.iter_content(chunk_size=_READ_CHUNK_SIZE)
        for chunk in chunks:
            if not chunk:
                continue
            if len(body) + len(chunk) > max_response_bytes:
                raise WebFetchProviderError(
                    code="response_too_large",
                    status_code=response.status_code,
                    detail=(
                        "Response body exceeds FLWR_WEB_FETCH_MAX_RESPONSE_BYTES "
                        f"({max_response_bytes})."
                    ),
                )
            body.extend(chunk)
    except requests.RequestException as exc:
        raise WebFetchProviderError(
            code="fetch_failed",
            detail=str(exc),
        ) from exc
    return bytes(body)


def _decode_response_body(response: requests.Response, body: bytes) -> str:
    """Decode response bytes to text."""
    encoding = response.encoding or response.apparent_encoding or "utf-8"
    return body.decode(encoding, errors="replace")


def _extract_markdown(html: str, url: str) -> str:
    """Extract markdown content with trafilatura."""
    extracted = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
    )
    if extracted is None:
        return ""
    return extracted


def _slice_content(
    content: str,
    start_index: int,
    max_length: int,
) -> tuple[str, bool, int | None]:
    """Return a character slice and continuation metadata."""
    original_length = len(content)
    if start_index >= original_length:
        return "", False, None

    end_index = min(start_index + max_length, original_length)
    truncated = end_index < original_length
    return content[start_index:end_index], truncated, end_index if truncated else None
