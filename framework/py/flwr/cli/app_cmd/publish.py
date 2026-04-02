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
"""Flower command line interface `app publish` command."""


from contextlib import ExitStack
from pathlib import Path
from typing import IO, Annotated, Any, cast

import click
import requests
import typer
from requests import Response

from flwr.cli.config_utils import load_and_validate
from flwr.common.constant import FAB_CONFIG_FILE
from flwr.supercore.constant import (
    APP_PUBLISH_ALLOWED_LICENSE_FILES,
    MAX_FILE_BYTES,
    MAX_FILE_COUNT,
    MAX_TOTAL_BYTES,
    MIME_MAP,
    PLATFORM_API_URL,
    SUPERGRID_ADDRESS,
    UTF8,
)
from flwr.supercore.version import package_version as flwr_version

from ..auth_plugin.oidc_cli_plugin import OidcCliPlugin
from ..config_utils import load as load_toml
from ..utils import (
    collect_files,
    filter_paths_for_publish,
    load_cli_auth_plugin_from_connection,
    validate_project_name,
)


# pylint: disable=too-many-locals
def publish(
    app: Annotated[
        Path,
        typer.Argument(
            help="Project directory to upload (defaults to current directory)."
        ),
    ] = Path("."),
) -> None:
    """Publish a Flower App to Flower Hub.

    This command uploads your app project to Flower Hub. Files are filtered based on
    .gitignore patterns and allowed file extensions.
    """
    auth_plugin = load_cli_auth_plugin_from_connection(SUPERGRID_ADDRESS)
    auth_plugin.load_tokens()
    if not isinstance(auth_plugin, OidcCliPlugin) or not auth_plugin.access_token:
        raise click.ClickException("Please log in before publishing app.")

    # Load token from the plugin
    token = auth_plugin.access_token

    # Resolve app path
    app = app.expanduser().resolve()

    # Validate app description from config
    config, _ = load_and_validate(app / FAB_CONFIG_FILE, check_module=False)
    _validate_app_name(app.name, "Flower App directory name")
    _validate_description(config["project"].get("description", ""))

    # Collect & validate app files
    file_paths = _collect_file_paths(app)
    _validate_files(file_paths)

    # Build and POST multipart
    with ExitStack() as stack:
        files_param = _build_multipart_files_param(app, file_paths, stack)
        try:
            resp = _post_files(files_param, token)
        except requests.RequestException as err:
            raise click.ClickException(f"Network error: {err}") from err

    if resp.ok:
        typer.secho("🎊 Upload successful", fg=typer.colors.GREEN, bold=True)
        return  # success

    # Error path:
    msg = f"Upload failed with status {resp.status_code}"
    if resp.text:
        msg += f": {resp.text}"
    raise click.ClickException(msg)


def _validate_description(description: Any) -> None:
    """Validate app description before publishing."""
    if not isinstance(description, str):
        raise click.ClickException(
            "Missing or invalid app description. "
            "Please set `description` in [project] of pyproject.toml."
        )

    if description.strip() == "":
        raise click.ClickException(
            "App description can't be empty. "
            "Please provide one with fewer than 200 characters."
        )

    if len(description) > 200:
        typer.secho(
            "Warning: the app description is more than 200 characters.",
            fg=typer.colors.YELLOW,
            bold=True,
        )
        should_continue = typer.confirm("Do you want to continue publishing anyway?")
        if not should_continue:
            raise click.ClickException("Publishing cancelled by user.")


def _validate_app_name(name: str, target: str) -> None:
    """Validate app and directory names used during publish."""
    try:
        validate_project_name(name, target)
    except ValueError as err:
        raise click.ClickException(str(err)) from None


def _detect_mime(path: Path) -> str:
    """Detect files' MIME."""
    return MIME_MAP.get(path.suffix.lower(), "text/plain; charset=utf-8")


def _get_declared_license_file(root: Path) -> Path | None:
    """Return validated absolute path from `[project].license.file`, else None."""
    # Read optional [project].license.file from pyproject.toml
    config = load_toml(root / "pyproject.toml")
    if config is None:
        return None
    project = config.get("project")
    if not isinstance(project, dict):
        return None
    license_entry = project.get("license")
    if not isinstance(license_entry, dict):
        return None
    if "file" not in license_entry:
        return None

    # Validate [project].license.file:
    # cannot be combined with `text`, must be a string,
    # must be an allowed filename, and must exist.
    license_file = license_entry["file"]
    if "file" in license_entry and "text" in license_entry:
        raise click.ClickException(
            "Invalid [project].license: `file` and `text` cannot be set together."
        )
    if not isinstance(license_file, str):
        raise click.ClickException(
            "Invalid [project].license.file: expected a string path."
        )
    if license_file not in APP_PUBLISH_ALLOWED_LICENSE_FILES:
        raise click.ClickException(
            "Invalid [project].license.file: only `LICENSE` or `LICENSE.md` "
            "are supported."
        )
    if not (root / license_file).is_file():
        raise click.ClickException(
            f"Invalid [project].license.file: `{license_file}` was declared "
            "but does not exist."
        )
    return (root / license_file).expanduser().resolve()


def _collect_file_paths(root: Path) -> list[Path]:
    """Return list of file paths that match include/exclude patterns."""
    declared_license_file = _get_declared_license_file(root)

    try:
        # Collect all files
        all_files = collect_files(root)
        # Filter files based on .gitignore and include/exclude patterns
        files = cast(dict[str, Path], filter_paths_for_publish(all_files))
        # Warn about skipped files (sorted for deterministic output)
        skipped_paths = sorted(set(all_files.keys()) - set(files.keys()))
        for path in skipped_paths:
            typer.secho(f"Skip: {path}", fg=typer.colors.YELLOW)
        # Build list of absolute file paths (sorted by relative path for stability)
        file_paths = [files[key].expanduser().resolve() for key in sorted(files.keys())]
    except ValueError as err:
        raise click.ClickException(str(err)) from err

    if declared_license_file and declared_license_file not in file_paths:
        raise click.ClickException(
            f"Invalid [project].license.file: `{declared_license_file.name}` is "
            "excluded by `.gitignore` or publish exclude rules."
        )

    return file_paths


def _validate_files(file_paths: list[Path]) -> None:
    """Validate files against upload constraints.

    Checks file count, individual file size, total size, and UTF-8 encoding.
    """
    if len(file_paths) == 0:
        raise click.ClickException(
            "Nothing to upload: no files matched after applying .gitignore and "
            "allowed extensions."
        )

    if len(file_paths) > MAX_FILE_COUNT:
        raise click.ClickException(
            f"Too many files: {len(file_paths)} > allowed maximum of {MAX_FILE_COUNT}."
        )

    # Calculate files size
    total_size = 0
    for path in file_paths:
        file_size = path.stat().st_size
        total_size += file_size

        # Check single file size
        if file_size > MAX_FILE_BYTES:
            raise click.ClickException(
                f"File too large: '{path.as_posix()}' is {file_size:,} bytes, "
                f"exceeding the per-file limit of {MAX_FILE_BYTES:,} bytes."
            )

        # Ensure we can decode as UTF-8.
        try:
            path.read_text(encoding=UTF8)
        except UnicodeDecodeError as err:
            raise click.ClickException(
                f"Encoding error: '{path}' is not UTF-8 encoded."
            ) from err

    # Check total files size
    if total_size > MAX_TOTAL_BYTES:
        raise click.ClickException(
            "Total size of all files is too large: "
            f"{total_size:,} bytes > {MAX_TOTAL_BYTES:,} bytes."
        )

    # Print validation passed prompt
    typer.echo(typer.style("✅ Validation passed", fg=typer.colors.GREEN, bold=True))
    typer.echo(f"{len(file_paths)} files, {total_size:,} bytes in total")


def _build_multipart_files_param(
    root: Path,
    file_paths: list[Path],
    stack: ExitStack,
) -> list[tuple[str, tuple[str, IO[bytes], str]]]:
    """Build multipart/form-data files parameter for HTTP upload.

    Returns list of tuples: (field_name, (filename, file_object, content_type)).
    File handles are registered with ExitStack for proper cleanup.
    """
    form: list[tuple[str, tuple[str, IO[bytes], str]]] = []
    for path in file_paths:
        # Detect MIME (content type)
        mime = _detect_mime(path)

        # Open file and register with ExitStack
        # pylint: disable-next=consider-using-with
        fobj = stack.enter_context(open(path.resolve(), "rb"))
        typer.echo(f"Attach {path} ({mime}, {path.stat().st_size:,} bytes)")

        # Get relative POSIX path
        relative_posix = path.relative_to(root).as_posix()

        # Append to form data (key, (filename, fileobj, mime))
        form.append(("files", (relative_posix, fobj, mime)))
    return form


def _post_files(
    files_param: list[tuple[str, tuple[str, IO[bytes], str]]],
    token: str,
) -> Response:
    """POST multipart with one part per file."""
    url = f"{PLATFORM_API_URL}/hub/apps/publish"
    headers = {"Authorization": f"Bearer {token}"}
    body = {"flwr_version": flwr_version}

    resp = requests.post(
        url, files=files_param, headers=headers, json=body, timeout=120
    )
    return resp
