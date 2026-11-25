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
"""Complete handle signup during CLI login."""

import ssl
import sys

import click
import httpx
import typer

from flwr.cli.config_utils import load_certificate_in_connection
from flwr.cli.typing import SuperLinkConnection
from flwr.cli.utils import prompt_text

_INVALID_RESPONSE = (
    "SuperLink returned an invalid signup response. Run flwr login again."
)


def complete_signup(connection: SuperLinkConnection, access_token: str) -> None:
    """Check signup status and prompt for a handle when the account needs one."""
    try:
        origin = httpx.URL(f"https://{connection.address or ''}")
    except httpx.InvalidURL:
        raise click.ClickException(
            "Signup requires a valid SuperLink address."
        ) from None
    if connection.insecure or not origin.host or origin.userinfo:
        raise click.ClickException("Signup requires a valid HTTPS SuperLink address.")
    handle_url = origin.copy_with(path="/v1/user/handle", query=None, fragment=None)
    signup_url = origin.copy_with(path="/v1/user/signup", query=None, fragment=None)

    verify: ssl.SSLContext | bool = True
    if root_certificates := load_certificate_in_connection(connection):
        verify = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        verify.load_verify_locations(cadata=root_certificates.decode("ascii"))

    try:
        with httpx.Client(
            verify=verify,
            follow_redirects=False,
            headers={"Authorization": f"Bearer {access_token}"},
        ) as client:
            if _read_handle(client.get(handle_url)) is not None:
                return

            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                raise click.ClickException(
                    "Your account needs a handle. Run flwr login again in an "
                    "interactive terminal to choose one."
                )

            while True:
                handle = prompt_text("Choose a handle for your Flower account")
                response = client.post(signup_url, json={"handle": handle})
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
                return
    except httpx.RequestError:
        raise click.ClickException(
            "Could not reach SuperLink to complete signup. Run flwr login again."
        ) from None


def _read_handle(response: httpx.Response) -> str | None:
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
    if isinstance(payload, dict) and "handle" in payload:
        handle = payload["handle"]
        if handle is None or (isinstance(handle, str) and handle.strip()):
            return handle
    raise click.ClickException(_INVALID_RESPONSE)
