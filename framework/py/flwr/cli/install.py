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
"""Flower command line interface `install` command."""


import hashlib
import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from typing import IO, Annotated

import click
import typer

from flwr.common.config import get_metadata_from_config
from flwr.common.constant import APP_DIR, FAB_HASH_TRUNCATION
from flwr.supercore.utils import get_flwr_home

from .archive_utils import safe_extract_zip
from .config_utils import load_and_validate
from .utils import get_sha256_hash


def install(
    source: Annotated[
        Path | None,
        typer.Argument(metavar="source", help="The source FAB file to install."),
    ] = None,
) -> None:
    """Install a Flower App Bundle.

    It can be run with a single FAB file argument:

        ``flwr install ./target_project.fab``

    Apps are installed to:

        - ``$FLWR_HOME/apps/`` if ``$FLWR_HOME`` is defined
        - ``$HOME/.flwr/apps/`` otherwise
    """
    if source is None:
        source = Path(typer.prompt("Enter the source FAB file"))

    source = source.expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise click.ClickException(
            f"The source {source} does not exist or is not a file."
        )

    if source.suffix != ".fab":
        raise click.ClickException(f"The source {source} is not a `.fab` file.")

    try:
        install_from_fab(source)
    except ValueError as e:
        raise click.ClickException(str(e)) from None


def install_from_fab(
    fab_file: Path | bytes,
    install_dir: Path | None = None,
    skip_prompt: bool = False,
) -> Path:
    """Install from a FAB file after extracting and validating.

    Parameters
    ----------
    fab_file : Path | bytes
        Either a path to the FAB file or the FAB file content as bytes.
    install_dir : Path | None
        Target installation directory, or None to use default.
    skip_prompt : bool
        If True, skip confirmation prompts. Default is False.

    Returns
    -------
    Path
        Path to the installed application directory.

    Raises
    ------
    click.ClickException
        If FAB format is invalid or hash verification fails.
    """
    fab_file_archive: Path | IO[bytes]
    fab_name: str | None
    if isinstance(fab_file, bytes):
        fab_file_archive = BytesIO(fab_file)
        fab_hash = hashlib.sha256(fab_file).hexdigest()
        fab_name = None
    elif isinstance(fab_file, Path):
        fab_file_archive = fab_file
        fab_hash = hashlib.sha256(fab_file.read_bytes()).hexdigest()
        fab_name = fab_file.stem
    else:
        raise ValueError("fab_file must be either a Path or bytes")

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(fab_file_archive, "r") as zipf:
            tmpdir_path = Path(tmpdir)
            safe_extract_zip(zipf, tmpdir_path)
            info_dir = tmpdir_path / ".info"
            if not info_dir.exists():
                raise click.ClickException("FAB file has incorrect format.")

            content_file = info_dir / "CONTENT"

            if not content_file.exists() or not _verify_hashes(
                content_file.read_text(), tmpdir_path
            ):
                raise click.ClickException("File hashes couldn't be verified.")

            shutil.rmtree(info_dir)

            installed_path = validate_and_install(
                tmpdir_path, fab_hash, fab_name, install_dir, skip_prompt
            )

    return installed_path


# pylint: disable=too-many-locals
def validate_and_install(
    project_dir: Path,
    fab_hash: str,
    fab_name: str | None,
    install_dir: Path | None,
    skip_prompt: bool = False,
) -> Path:
    """Validate the TOML file and install the project to the desired directory.

    Parameters
    ----------
    project_dir : Path
        Path to the extracted project directory.
    fab_hash : str
        SHA-256 hash of the FAB file.
    fab_name : str | None
        Name of the FAB file, or None if installing from bytes.
    install_dir : Path | None
        Target installation directory, or None to use default.
    skip_prompt : bool (default: False)
        If True, skip confirmation prompts.

    Returns
    -------
    Path
        Path to the installed application directory.

    Raises
    ------
    click.ClickException
        If configuration is invalid or metadata doesn't match.
    """
    config, _ = load_and_validate(project_dir / "pyproject.toml", check_module=False)

    fab_id, version = get_metadata_from_config(config)
    publisher, project_name = fab_id.split("/")
    config_metadata = (publisher, project_name, version, fab_hash)

    if fab_name:
        _validate_fab_and_config_metadata(fab_name, config_metadata)

    install_base_dir = install_dir.expanduser() if install_dir else get_flwr_home()
    install_path: Path = (
        install_base_dir
        / APP_DIR
        / f"{publisher}.{project_name}.{version}.{fab_hash[:FAB_HASH_TRUNCATION]}"
    )
    if install_path.exists():
        if skip_prompt:
            return install_path
        if not typer.confirm(
            typer.style(
                f"\n💬 {project_name} version {version} is already installed, "
                "do you want to reinstall it?",
                fg=typer.colors.MAGENTA,
                bold=True,
            )
        ):
            return install_path

    install_path.mkdir(parents=True, exist_ok=True)

    # Move contents from source directory
    for item in project_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, install_path / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, install_path / item.name)

    typer.secho(
        f"Successfully installed {project_name} to {install_path}.",
        fg=typer.colors.GREEN,
        bold=True,
    )

    return install_path


def _verify_hashes(list_content: str, tmpdir: Path) -> bool:
    """Verify file hashes based on the CONTENT manifest.

    Parameters
    ----------
    list_content : str
        Content of the CONTENT manifest file with hash information.
    tmpdir : Path
        Temporary directory containing extracted files.

    Returns
    -------
    bool
        True if all file hashes match, False otherwise.
    """
    for line in list_content.strip().split("\n"):
        rel_path, hash_expected, _ = line.split(",")
        file_path = tmpdir / rel_path
        if not file_path.exists() or get_sha256_hash(file_path) != hash_expected:
            return False
    return True


def _validate_fab_and_config_metadata(
    fab_name: str, config_metadata: tuple[str, str, str, str]
) -> None:
    """Validate metadata from the FAB filename and config.

    Parameters
    ----------
    fab_name : str
        The FAB filename (with or without .fab extension).
    config_metadata : tuple[str, str, str, str]
        Tuple of (publisher, project_name, version, fab_hash).

    Raises
    ------
    click.ClickException
        If filename format is incorrect or hash doesn't match.
    """
    publisher, project_name, version, fab_hash = config_metadata

    fab_name = fab_name.removesuffix(".fab")

    fab_publisher, fab_project_name, fab_version, fab_shorthash = fab_name.split(".")
    fab_version = fab_version.replace("-", ".")

    # Check FAB filename format
    if (
        f"{fab_publisher}.{fab_project_name}.{fab_version}"
        != f"{publisher}.{project_name}.{version}"
        or len(fab_shorthash) != FAB_HASH_TRUNCATION  # Verify hash length
    ):
        raise click.ClickException(
            "FAB file has incorrect name. The file name must follow the format "
            "`<publisher>.<project_name>.<version>.<8hexchars>.fab`."
        )

    # Verify hash is a valid hexadecimal
    try:
        _ = int(fab_shorthash, 16)
    except Exception as e:
        raise click.ClickException(
            f"FAB file has an invalid hexadecimal string `{fab_shorthash}`."
        ) from e

    # Verify shorthash matches
    if fab_shorthash != fab_hash[:FAB_HASH_TRUNCATION]:
        raise click.ClickException(
            "The hash in the FAB file name does not match the hash of the FAB."
        )
