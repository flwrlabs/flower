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
"""Flower command line interface `new` command."""


import io
import zipfile
from pathlib import Path
from typing import Annotated, cast

import click
import requests
import typer

from flwr.supercore.constant import PLATFORM_API_URL
from flwr.supercore.utils import parse_app_spec, request_download_link

from ..archive_utils import safe_extract_zip
from ..utils import prompt_options, prompt_text


# pylint: disable=too-many-locals,too-many-branches,too-many-statements
def new(
    app_spec: Annotated[
        str | None,
        typer.Argument(
            help="Flower app specifier. Use the format "
            "'@account_name/app_name' or '@account_name/app_name==x.y.z'. "
            "Version is optional (defaults to latest)."
        ),
    ] = None,
    framework: Annotated[
        str | None,
        typer.Option(case_sensitive=False, help="Deprecated. The ML framework to use"),
    ] = None,
    username: Annotated[
        str | None,
        typer.Option(
            case_sensitive=False, help="Deprecated. The Flower username of the author"
        ),
    ] = None,
) -> None:
    """Create new Flower App."""
    if framework is not None or username is not None:
        raise click.ClickException(
            "The --framework and --username options are deprecated and will be "
            "removed in future versions of Flower. Please provide an app specifier "
            "after `flwr new` instead, e.g., '@account_name/app_name' or "
            "'@account_name/app_name==x.y.z'."
        )

    if app_spec is None:
        # Fetch recommended apps
        print(
            typer.style(
                "\n🌸 Fetching recommended apps...",
                fg=typer.colors.GREEN,
                bold=True,
            )
        )
        apps = fetch_recommended_apps()

        if not apps:
            typer.secho(
                "No recommended apps found. Please provide an app specifier manually.",
                fg=typer.colors.YELLOW,
            )
            app_spec = prompt_text("Please provide the app specifier")
        else:
            # Extract app_ids and show selection menu
            app_ids = [app["app_id"] for app in apps]
            app_spec = prompt_options(
                "Select a Flower App to create by entering "
                "the number from the list below:",
                app_ids,
            )

    # Download remote app
    download_remote_app_via_api(app_spec)


def print_success_prompt(package_name: str) -> None:
    """Print styled setup instructions for running a new Flower App after creation."""
    prompt = typer.style(
        "🎊 Flower App creation successful.\n\n"
        "To run your Flower App, first install its dependencies:\n\n",
        fg=typer.colors.GREEN,
        bold=True,
    )

    prompt += typer.style(
        f"	cd {package_name} && pip install -e .\n\n",
        fg=typer.colors.BRIGHT_CYAN,
        bold=True,
    )

    prompt += typer.style(
        "then, run the app:\n\n ",
        fg=typer.colors.GREEN,
        bold=True,
    )

    prompt += typer.style(
        "\tflwr run .\n\n",
        fg=typer.colors.BRIGHT_CYAN,
        bold=True,
    )

    prompt += typer.style(
        "💡 Check the README in your app directory to learn how to\n"
        "customize it and how to run it using the Deployment Runtime.\n",
        fg=typer.colors.GREEN,
        bold=True,
    )

    print(prompt)


def fetch_recommended_apps() -> list[dict[str, str]]:
    """Fetch recommended apps from Platform API."""
    url = f"{PLATFORM_API_URL}/hub/apps?tag=recommended"
    try:
        response = requests.get(url, headers={"accept": "application/json"}, timeout=10)
        response.raise_for_status()
        data = response.json()
        apps = data.get("apps", [])
        return cast(list[dict[str, str]], apps)

    except requests.RequestException as e:
        raise click.ClickException(f"Failed to fetch recommended apps: {e}") from e


def _download_zip_to_memory(presigned_url: str) -> io.BytesIO:
    """Download ZIP file from Platform API to memory."""
    try:
        r = requests.get(presigned_url, timeout=60)
        r.raise_for_status()
    except requests.RequestException as e:
        raise click.ClickException(f"ZIP download failed: {e}") from e

    buf = io.BytesIO(r.content)
    # Validate it's a zip
    if not zipfile.is_zipfile(buf):
        raise click.ClickException("Downloaded file is not a valid ZIP")
    buf.seek(0)
    return buf


def download_remote_app_via_api(app_spec: str) -> None:
    """Download App from Platform API."""
    # Validate app version and ID format
    try:
        app_id, app_version = parse_app_spec(app_spec)
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    app_name = app_id.split("/")[1]

    project_dir = Path.cwd() / app_name
    if project_dir.exists():
        if not typer.confirm(
            typer.style(
                f"\n💬 {app_name} already exists, do you want to override it?",
                fg=typer.colors.MAGENTA,
                bold=True,
            )
        ):
            return

    typer.secho(
        f"\n🔗 Requesting download link for {app_id}...",
        fg=typer.colors.GREEN,
        bold=True,
    )
    # Fetch ZIP downloading URL
    url = f"{PLATFORM_API_URL}/hub/fetch-zip"
    try:
        presigned_url, _ = request_download_link(app_id, app_version, url, "zip_url")
    except ValueError as e:
        raise click.ClickException(str(e)) from e

    typer.secho(
        "🔽 Downloading ZIP into memory...",
        fg=typer.colors.GREEN,
        bold=True,
    )
    zip_buf = _download_zip_to_memory(presigned_url)

    typer.secho(
        f"📦 Unpacking into {project_dir}...",
        fg=typer.colors.GREEN,
        bold=True,
    )
    with zipfile.ZipFile(zip_buf) as zf:
        safe_extract_zip(zf, Path.cwd())

    print_success_prompt(app_name)
