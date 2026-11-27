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
"""Complete account signup during CLI login."""

import ssl
import sys
import time
import webbrowser
from collections.abc import Callable
from ipaddress import ip_address
from urllib.parse import urlsplit

import click
import httpx
import typer

from flwr.cli.config_utils import load_certificate_in_connection
from flwr.cli.typing import SuperLinkConnection
from flwr.cli.utils import (
    flwr_cli_exc_handler,
    prompt_text,
    refresh_account_auth_tokens,
)
from flwr.supercore.auth.typing import AccountAuthCredentials
from flwr.supercore.control import ControlHttpClient

_INVALID_RESPONSE = (
    "SuperLink returned an invalid signup response. Run flwr login again."
)
_CONNECTION_ERROR = (
    "Could not reach SuperLink to complete signup. Run flwr login again."
)
_RESUME_SIGNUP = (
    "Finish signup in your browser, then retry your Flower command. "
    "Run flwr login again to reopen signup."
)
_SIGNUP_POLL_INTERVAL = 3
_SIGNUP_TIMEOUT = 300


def complete_signup(
    connection: SuperLinkConnection,
    credentials: AccountAuthCredentials,
    control_client: ControlHttpClient,
) -> AccountAuthCredentials:
    """Return the current credentials after checking or completing handle signup."""
    handle_url = "/v1/user/handle"
    signup_url = "/v1/user/signup"
    try:
        with _signup_client(connection) as client:

            def send_with_refresh(request: httpx.Request) -> httpx.Response:
                nonlocal credentials
                response, credentials = _send_with_refresh(
                    client, request, credentials, control_client
                )
                return response

            response = send_with_refresh(client.build_request("GET", handle_url))
            if _read_handle(response) is not None:
                return credentials

            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                raise click.ClickException(
                    "Your account needs a handle. Run flwr login again in an "
                    "interactive terminal to choose one."
                )

            while True:
                handle = prompt_text("Choose a handle for your Flower account")
                response = send_with_refresh(
                    client.build_request("POST", signup_url, json={"handle": handle})
                )
                if response.status_code == httpx.codes.CONFLICT:
                    typer.echo(
                        "That handle is already in use. Choose another.", err=True
                    )
                    continue
                if response.status_code == httpx.codes.UNPROCESSABLE_ENTITY:
                    typer.echo("That handle is invalid. Choose another.", err=True)
                    continue
                if _read_handle(response) is None:
                    raise click.ClickException(_INVALID_RESPONSE)
                return credentials
    except httpx.RequestError:
        raise click.ClickException(_CONNECTION_ERROR) from None


def _read_handle(response: httpx.Response) -> str | None:
    payload = _read_signup_response(response)
    if "handle" in payload:
        handle = payload["handle"]
        if handle is None or (isinstance(handle, str) and handle.strip()):
            return handle
    raise click.ClickException(_INVALID_RESPONSE)


def wait_for_web_signup(
    connection: SuperLinkConnection,
    credentials: AccountAuthCredentials,
    control_client: ControlHttpClient,
    web_signup_url: str,
    store_tokens: Callable[[AccountAuthCredentials], None],
) -> None:
    """Open browser signup when needed and wait for SuperLink to confirm completion.

    Parameters
    ----------
    connection : SuperLinkConnection
        The configured SuperLink connection used to check signup status.
    credentials : AccountAuthCredentials
        The already saved credentials from device authentication.
    control_client : ControlHttpClient
        The Control client used to refresh expired credentials.
    web_signup_url : str
        The browser signup URL advertised by SuperLink.
    store_tokens : Callable[[AccountAuthCredentials], None]
        Callback for saving refreshed credentials immediately.
    """
    try:
        with _signup_client(connection) as client:

            def signup_complete() -> bool:
                nonlocal credentials
                response, credentials = _send_with_refresh(
                    client,
                    client.build_request("GET", "/v1/user/signup/status"),
                    credentials,
                    control_client,
                    store_tokens,
                )
                complete = _read_signup_response(response).get("complete")
                if not isinstance(complete, bool):
                    raise click.ClickException(_INVALID_RESPONSE)
                return complete

            if signup_complete():
                return

            if not _valid_web_signup_url(web_signup_url):
                raise click.ClickException(
                    "SuperLink returned an invalid browser signup URL. "
                    "Contact your administrator."
                )
            typer.echo(
                "Device approved. Finish setting up your Flower account in your "
                "browser using the same sign-in:\n"
                f"{web_signup_url}\n"
                "Waiting for signup to complete. Press Ctrl+C to stop waiting."
            )
            try:
                webbrowser.open(web_signup_url)
            except (OSError, webbrowser.Error):
                # The URL above also works when a browser cannot be opened here.
                pass

            deadline = time.monotonic() + _SIGNUP_TIMEOUT
            while (remaining := deadline - time.monotonic()) > 0:
                time.sleep(min(_SIGNUP_POLL_INTERVAL, remaining))
                if time.monotonic() >= deadline:
                    break
                if signup_complete():
                    return
    except httpx.RequestError:
        raise click.ClickException(_CONNECTION_ERROR) from None
    except KeyboardInterrupt:
        raise click.ClickException(
            f"Stopped waiting for signup. {_RESUME_SIGNUP}"
        ) from None

    raise click.ClickException(f"Timed out waiting for signup. {_RESUME_SIGNUP}")


def _valid_web_signup_url(url: str) -> bool:
    if not url.isprintable() or " " in url or "\\" in url:
        return False
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        if not host or parsed.username is not None or parsed.port == 0:
            return False
        return parsed.scheme == "https" or (
            parsed.scheme == "http"
            and (host == "localhost" or ip_address(host).is_loopback)
        )
    except ValueError:
        return False


def _signup_client(connection: SuperLinkConnection) -> httpx.Client:
    try:
        origin = httpx.URL(f"https://{connection.address or ''}")
    except httpx.InvalidURL:
        raise click.ClickException(
            "Signup requires a valid SuperLink address."
        ) from None
    if connection.insecure or not origin.host or origin.userinfo:
        raise click.ClickException("Signup requires a valid HTTPS SuperLink address.")

    verify: ssl.SSLContext | bool = True
    if root_certificates := load_certificate_in_connection(connection):
        verify = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        verify.load_verify_locations(cadata=root_certificates.decode("ascii"))

    return httpx.Client(
        base_url=origin.copy_with(path="/", query=None, fragment=None),
        verify=verify,
        follow_redirects=False,
    )


def _send_with_refresh(
    client: httpx.Client,
    request: httpx.Request,
    credentials: AccountAuthCredentials,
    control_client: ControlHttpClient,
    store_tokens: Callable[[AccountAuthCredentials], None] | None = None,
) -> tuple[httpx.Response, AccountAuthCredentials]:
    request.headers["Authorization"] = f"Bearer {credentials.access_token}"
    response = client.send(request)
    if (
        response.status_code != httpx.codes.UNAUTHORIZED
        or response.headers.get("WWW-Authenticate") != "Bearer"
    ):
        return response, credentials

    with flwr_cli_exc_handler():
        credentials = refresh_account_auth_tokens(
            control_client, credentials.refresh_token
        )
    if store_tokens is not None:
        store_tokens(credentials)
    request.headers["Authorization"] = f"Bearer {credentials.access_token}"
    return client.send(request), credentials


def _read_signup_response(response: httpx.Response) -> dict[str, object]:
    if response.status_code in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
        raise click.ClickException(
            "SuperLink denied signup. Sign in again or contact your administrator "
            "if the problem persists."
        )
    if response.status_code != httpx.codes.OK:
        raise click.ClickException(
            f"SuperLink could not complete signup (HTTP {response.status_code}). "
            "Run flwr login again or contact your administrator."
        )
    try:
        payload = response.json()
    except ValueError:
        raise click.ClickException(_INVALID_RESPONSE) from None
    if isinstance(payload, dict):
        return payload
    raise click.ClickException(_INVALID_RESPONSE)
