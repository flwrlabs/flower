# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower command line interface `app review` command."""


import base64
import hashlib
import re
from pathlib import Path
from typing import Annotated

import click
import requests
import typer
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric import ed25519

from flwr.common import now
from flwr.supercore.constant import PLATFORM_API_URL, SUPERGRID_ADDRESS
from flwr.supercore.primitives.asymmetric_ed25519 import (
    create_message_to_sign,
    load_private_key,
    sign_message,
)
from flwr.supercore.utils import get_flwr_home, parse_app_spec, request_download_link
from flwr.supercore.version import package_version as flwr_version

from ..auth_plugin.oidc_cli_plugin import OidcCliPlugin
from ..install import install_from_fab
from ..utils import load_cli_auth_plugin_from_connection

TRY_AGAIN_MESSAGE = "Please try again or press CTRL+C to abort.\n"


# pylint: disable-next=too-many-locals, too-many-statements
def review(
    app_spec: Annotated[
        str,
        typer.Argument(
            help="App specifier (e.g., '@account/app' or '@account/app==1.0.0'). "
            "Version is optional; defaults to the latest."
        ),
    ],
) -> None:
    """Download a FAB for <APP-ID>, unpack it for manual review, and upon confirmation
    sign & submit the review to Flower Hub."""
    auth_plugin = load_cli_auth_plugin_from_connection(SUPERGRID_ADDRESS)

    auth_plugin.load_tokens()
    if not isinstance(auth_plugin, OidcCliPlugin) or not auth_plugin.access_token:
        raise click.ClickException("Please log in before reviewing app.")

    # Load token from the plugin
    token = auth_plugin.access_token

    # Validate app version and ID format
    try:
        app_id, app_version = parse_app_spec(app_spec)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    # Download FAB
    typer.secho("Downloading FAB... ", fg=typer.colors.BLUE)
    url = f"{PLATFORM_API_URL}/hub/fetch-fab"
    try:
        presigned_url, _ = request_download_link(app_id, app_version, url, "fab_url")

        fab_bytes = _download_fab(presigned_url)

        # Unpack FAB
        typer.secho("Unpacking FAB... ", fg=typer.colors.BLUE)
        review_dir = _create_review_dir()
        review_app_path = install_from_fab(fab_bytes, review_dir)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    # Extract app version
    version_pattern = re.compile(r"\b(\d+\.\d+\.\d+)\b")
    match = version_pattern.search(str(review_app_path))
    assert match is not None
    app_version = match.group(1)

    # Prompt to ask for sign
    typer.secho(
        f"""
    Review the unpacked app in the following directory:

        {typer.style(review_app_path, fg=typer.colors.GREEN, bold=True)}

    If you have reviewed the app and want to continue to sign it,
    type {typer.style("SIGN", fg=typer.colors.GREEN, bold=True)} or abort with CTRL+C.
    """,
        fg=typer.colors.BLUE,
    )

    confirmation = typer.prompt("Type SIGN to continue").strip()
    if confirmation.upper() != "SIGN":
        typer.secho("Aborted (user did not type SIGN).", fg=typer.colors.YELLOW)
        raise typer.Exit(code=130)

    # Ask for private key path (retry until valid)
    while True:
        try:
            key_path_str = typer.prompt(
                "Please specify the path of Ed25519 OpenSSH private key for signing"
            )
        except typer.Abort as e:
            typer.secho("Aborted by user.", fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(code=130) from e

        key_path = Path(key_path_str).expanduser().resolve()

        if not key_path.is_file():
            typer.secho(
                f"❌ Private key not found: {key_path}",
                fg=typer.colors.RED,
                err=True,
            )
            typer.secho(TRY_AGAIN_MESSAGE, fg=typer.colors.YELLOW)
            continue

        # Load private key
        try:
            private_key = load_private_key(key_path)
        except (OSError, ValueError, UnsupportedAlgorithm) as e:
            typer.secho(
                f"❌ Failed to load the private key: {e}", fg=typer.colors.RED, err=True
            )
            typer.secho(TRY_AGAIN_MESSAGE, fg=typer.colors.YELLOW)
            continue
        break  # valid

    # Sign FAB
    signature, signed_at = _sign_fab(fab_bytes, private_key)

    # Submit review
    _submit_review(app_id, app_version, signature, signed_at, token)


def _create_review_dir() -> Path:
    """Create a directory for reviewing code."""
    review_dir = get_flwr_home() / "reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    return review_dir


def _download_fab(url: str) -> bytes:
    """Download FAB file from given URL."""
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        raise click.ClickException(f"FAB download failed: {e}") from e
    return r.content


def _sign_fab(
    fab_bytes: bytes, private_key: ed25519.Ed25519PrivateKey
) -> tuple[bytes, int]:
    """Sign the given FAB hash bytes."""
    # Get current timestamp
    timestamp = int(now().timestamp())
    message_to_sign = create_message_to_sign(
        hashlib.sha256(fab_bytes).digest(),
        timestamp,
    )
    return sign_message(private_key, message_to_sign), timestamp


def _submit_review(
    app_id: str, app_version: str, signature: bytes, signed_at: int, token: str
) -> None:
    """Submit review to Flower Hub."""
    signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
    url = f"{PLATFORM_API_URL}/hub/apps/signature"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {
        "app_id": app_id,
        "app_version": app_version,
        "signature_b64": signature_b64,
        "signed_at": signed_at,
        "flwr_version": flwr_version,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
    except requests.RequestException as e:
        raise click.ClickException(f"Network error while submitting review: {e}") from e

    if resp.ok:
        typer.secho("🎊 Review submitted", fg=typer.colors.GREEN, bold=True)
        return

    # Error path:
    msg = f"Review submission failed (HTTP {resp.status_code})"
    if resp.text:
        msg += f": {resp.text}"
    raise click.ClickException(msg)
