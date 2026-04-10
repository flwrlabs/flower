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
"""Flower command line interface `build` command."""


import hashlib
import zipfile
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

import click
import pathspec
import tomli
import tomli_w
import typer

from flwr.common.config import check_pattern_list_value
from flwr.common.constant import (
    FAB_CONFIG_FILE,
    FAB_DATE,
    FAB_EXCLUDE_KEY,
    FAB_EXCLUDE_PATTERNS,
    FAB_HASH_TRUNCATION,
    FAB_INCLUDE_KEY,
    FAB_INCLUDE_PATTERNS,
    FAB_MAX_SIZE,
)
from flwr.supercore.fab_format_version import (
    FabFormatMetadata,
    normalize_and_validate_fab_format,
    validate_fab_files_for_format,
)

from .config_utils import load_and_validate
from .utils import (
    build_pathspec,
    collect_files,
    filter_paths_for_publish,
    validate_project_name,
)


def write_to_zip(
    zipfile_obj: zipfile.ZipFile, filename: str, contents: bytes | str
) -> zipfile.ZipFile:
    """Set a fixed date and write contents to a zip file.

    Parameters
    ----------
    zipfile_obj : zipfile.ZipFile
        The ZipFile object to write to.
    filename : str
        Name of the file within the zip archive.
    contents : bytes | str
        The file contents to write.

    Returns
    -------
    ZipFile
        The modified ZipFile object.
    """
    zip_info = zipfile.ZipInfo(filename)
    zip_info.date_time = FAB_DATE
    zipfile_obj.writestr(zip_info, contents)
    return zipfile_obj


def get_fab_filename(config: dict[str, Any], fab_hash: str) -> str:
    """Get the FAB filename based on the given config and FAB hash.

    Parameters
    ----------
    config : dict[str, Any]
        The Flower App configuration dictionary.
    fab_hash : str
        The SHA-256 hash of the FAB file.

    Returns
    -------
    str
        The formatted FAB filename in the pattern:
        <publisher>.<name>.<version>.<hash_prefix>.fab
    """
    publisher = config["tool"]["flwr"]["app"]["publisher"]
    name = config["project"]["name"]
    version = config["project"]["version"].replace(".", "-")
    fab_hash_truncated = fab_hash[:FAB_HASH_TRUNCATION]
    return f"{publisher}.{name}.{version}.{fab_hash_truncated}.fab"


def _get_project_name(config: dict[str, Any]) -> str:
    """Return the validated project name from pyproject.toml."""
    project = config.get("project")
    if not isinstance(project, dict):
        raise ValueError("Missing [project] section")

    name = project.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError('Property "name" missing in [project]')

    return name.strip()


# pylint: disable=too-many-locals, too-many-statements
def build(
    app: Annotated[
        Path | None,
        typer.Option(help="Path of the Flower App to bundle into a FAB"),
    ] = None,
) -> None:
    """Build a Flower App into a Flower App Bundle (FAB).

    You can run ``flwr build`` without any arguments to bundle the app located in the
    current directory. Alternatively, you can you can specify a path using the ``--app``
    option to bundle an app located at the provided path. For example:

    ``flwr build --app ./apps/flower-hello-world``.
    """
    if app is None:
        app = Path.cwd()

    app = app.expanduser().resolve()
    if not app.is_dir():
        raise click.ClickException(
            f"The path {app} is not a valid path to a Flower app."
        )

    try:
        validate_project_name(app.name, "The Flower App directory name")
    except ValueError as err:
        raise click.ClickException(str(err)) from None

    try:
        config, warnings = load_and_validate(app / "pyproject.toml")
    except ValueError as e:
        raise click.ClickException(str(e)) from None

    if warnings:
        typer.secho(
            "Flower App configuration (pyproject.toml) is missing the following "
            "recommended properties:\n" + "\n".join([f"- {line}" for line in warnings]),
            fg=typer.colors.YELLOW,
            bold=True,
        )

    # Build FAB
    try:
        fab_bytes = build_fab_from_disk(app)
    except ValueError as e:
        raise click.ClickException(str(e)) from None

    # Calculate hash for filename
    fab_hash = hashlib.sha256(fab_bytes).hexdigest()

    # Get the name of the zip file
    fab_filename = get_fab_filename(config, fab_hash)

    # Write the FAB
    Path(fab_filename).write_bytes(fab_bytes)

    typer.secho(
        f"🎊 Successfully built {fab_filename}", fg=typer.colors.GREEN, bold=True
    )


def build_fab_from_disk(app: Path) -> bytes:
    """Build a FAB from files on disk and return the FAB as bytes."""
    return build_fab_from_files(collect_files(app.expanduser().resolve()))[0]


def build_fab_from_files(
    files: Mapping[str, bytes | Path],
) -> tuple[bytes, FabFormatMetadata]:
    r"""Build a FAB from in-memory files and return the FAB plus metadata.

    This is the core FAB building function that works with in-memory data.
    It accepts either bytes or Path objects as file contents, applies filtering
    rules (include/exclude patterns), and builds the FAB.

    Parameters
    ----------
    files : Mapping[str, bytes | Path]
        Mapping of relative POSIX file paths to their contents.
        - Keys: Relative paths (strings)
        - Values: Either bytes (file contents) or Path (will be read)
        Must include "pyproject.toml" and optionally ".gitignore".

    Returns
    -------
    tuple[bytes, FabFormatMetadata]
        The FAB as bytes together with normalized compatibility metadata.
        The metadata is consumed by platform-api during publish to persist
        compatibility fields derived from this shared build validation logic.

    Examples
    --------
    Build a FAB from in-memory files::

        files = {
            "pyproject.toml": b"[project]\nname = 'myapp'\n...",
            ".gitignore": b"*.pyc\n__pycache__/\n",
            "src/client.py": Path("/path/to/client.py"),
            "src/server.py": b"print('hello')",
            "README.md": b"# My App\n",
        }
        fab_bytes, metadata = build_fab_from_files(files)
    """

    def _to_bytes(content: bytes | Path) -> bytes:
        return content.read_bytes() if isinstance(content, Path) else content

    def _add_to_fab(
        fab_file: zipfile.ZipFile,
        path: str,
        content: bytes,
    ) -> str:
        """Write a file to the FAB and return its CONTENT manifest line.

        Parameters
        ----------
        fab_file : zipfile.ZipFile
            The ZipFile object to write to.
        path : str
            The file path within the FAB.
        content : bytes
            The file contents as bytes.

        Returns
        -------
        str
            A CONTENT manifest line: "path,sha256,size_bits"
        """
        write_to_zip(fab_file, path, content)
        sha256_hash = hashlib.sha256(content).hexdigest()
        file_size_bits = len(content) * 8
        return f"{path},{sha256_hash},{file_size_bits}"

    # Apply publish-style rules (.gitignore + publish include/exclude).
    files = filter_paths_for_publish(files)

    # Extract, load, and parse pyproject.toml
    if FAB_CONFIG_FILE not in files:
        raise ValueError(
            f"{FAB_CONFIG_FILE} not found in files. It may have been excluded by "
            ".gitignore."
        )
    pyproject_content = _to_bytes(files[FAB_CONFIG_FILE])
    config = tomli.loads(pyproject_content.decode("utf-8"))
    validate_project_name(_get_project_name(config), "The Flower App [project].name")
    metadata = normalize_and_validate_fab_format(config)

    # Remove the 'federations' field if it exists
    if (
        "tool" in config
        and "flwr" in config["tool"]
        and "federations" in config["tool"]["flwr"]
    ):
        del config["tool"]["flwr"]["federations"]

    # Apply FAB include/exclude rules (user patterns + built-in).
    filtered_paths = get_filtered_fab_paths(files, config)
    filtered_paths.sort()  # Sort for deterministic output
    validate_fab_files_for_format(config, filtered_paths)

    # Build FAB with CONTENT manifest
    fab_buffer = BytesIO()
    with zipfile.ZipFile(fab_buffer, "w", zipfile.ZIP_DEFLATED) as fab_file:
        # Add pyproject.toml and collect manifest entries
        pyproject_bytes = tomli_w.dumps(config).encode("utf-8")
        manifest_lines = [_add_to_fab(fab_file, FAB_CONFIG_FILE, pyproject_bytes)]

        # Add remaining files and collect their manifest entries
        for file_path in filtered_paths:
            file_content = _to_bytes(files[file_path])
            manifest_lines.append(_add_to_fab(fab_file, file_path, file_content))

        # Write CONTENT manifest to the zip file
        write_to_zip(fab_file, ".info/CONTENT", "\n".join(manifest_lines))

    fab_bytes = fab_buffer.getvalue()

    # Validate FAB size
    if len(fab_bytes) > FAB_MAX_SIZE:
        raise ValueError(
            f"FAB size exceeds maximum allowed size of {FAB_MAX_SIZE:,} bytes. "
            f"To reduce package size, narrow `{FAB_INCLUDE_KEY}` or add "
            f"`{FAB_EXCLUDE_KEY}` patterns in [tool.flwr.app]."
        )

    # Returned metadata is consumed by platform during publish.
    return fab_bytes, metadata


def get_user_fab_patterns(
    config: dict[str, Any],
) -> tuple[list[str] | None, list[str] | None]:
    """Return user-defined FAB include/exclude patterns.

    Returns ``None`` for a key that is absent from the config, or the
    non-empty pattern list itself. Raises ``ValueError`` if a key is
    present but set to an empty list.
    """
    app_conf = config.get("tool", {}).get("flwr", {}).get("app", {})
    if not isinstance(app_conf, dict):
        return None, None

    def _get_pattern_list(key: str) -> list[str] | None:
        if key not in app_conf:
            return None
        value: list[str] = app_conf[key]
        error = check_pattern_list_value(value, key)
        if error:
            raise ValueError(error)
        return value

    return _get_pattern_list(FAB_INCLUDE_KEY), _get_pattern_list(FAB_EXCLUDE_KEY)


def get_filtered_fab_paths(
    files: dict[str, bytes | Path],
    config: dict[str, Any],
) -> list[str]:
    """Compute final FAB file list using user patterns and non-overridable defaults."""
    # Build built-in spec
    normalized_paths = list(files.keys())
    built_in_include_spec = build_pathspec(FAB_INCLUDE_PATTERNS)
    built_in_exclude_spec = build_pathspec(FAB_EXCLUDE_PATTERNS)
    user_include_spec = None
    user_exclude_spec = None

    # Load and validate user patterns, and build user specs
    user_include_patterns, user_exclude_patterns = get_user_fab_patterns(config)
    if user_include_patterns is not None:
        _raise_on_unresolved_patterns(
            user_include_patterns, normalized_paths, FAB_INCLUDE_KEY
        )
        user_include_spec = build_pathspec(user_include_patterns)
    if user_exclude_patterns is not None:
        _raise_on_unresolved_patterns(
            user_exclude_patterns, normalized_paths, FAB_EXCLUDE_KEY
        )
        user_exclude_spec = build_pathspec(user_exclude_patterns)
    has_user_rules = bool(user_include_spec or user_exclude_spec)

    # Build the candidate set of files based on user-defined patterns,
    # or all files if no user patterns are defined.
    candidate_paths = normalized_paths
    if user_include_spec:
        candidate_paths = list(
            user_include_spec.match_files(candidate_paths)  # type: ignore
        )
    if user_exclude_spec:
        candidate_paths = list(
            user_exclude_spec.match_files(candidate_paths, negate=True)  # type: ignore
        )

    # Apply built-in constraints and validate against user patterns
    if has_user_rules:
        _raise_on_built_in_pattern_conflicts(candidate_paths, built_in_include_spec)
    final_paths = [
        path
        for path in candidate_paths
        if built_in_include_spec.match_file(path)
        and not built_in_exclude_spec.match_file(path)
    ]
    return final_paths


def _raise_on_unresolved_patterns(
    patterns: list[str], file_paths: list[str], key_name: str
) -> None:
    """Raise ValueError for any user-defined pattern that is invalid or matches
    nothing."""
    for pattern in patterns:
        try:
            pattern_spec = build_pathspec([pattern])
        except Exception as err:  # pylint: disable=broad-except
            raise ValueError(
                f'Invalid pattern in "{key_name}": "{pattern}" ({err})'
            ) from err

        if not any(pattern_spec.match_file(path) for path in file_paths):
            raise ValueError(
                f'Pattern in "{key_name}" did not match any files: "{pattern}". '
                "Correct the pattern or remove it."
            )


def _raise_on_built_in_pattern_conflicts(
    candidate_paths: list[str],
    built_in_include_spec: pathspec.PathSpec,
) -> None:
    """Raise ValueError for user-defined rules and built-in rules conflicts."""
    # Only count files whose type is not supported by built-in include patterns
    # (e.g. .txt files). Files that match built-in includes but are removed by
    # built-in excludes (e.g. .toml inside .venv/, pyproject.toml) are expected
    # removals and should not be flagged.
    removed_files = set(built_in_include_spec.match_files(candidate_paths, negate=True))
    if removed_files:
        files_list = "\n".join(f"- {file}" for file in removed_files)
        raise ValueError(
            f'{len(removed_files)} file(s) matched "{FAB_INCLUDE_KEY}" but were '
            "removed by non-overridable built-in FAB constraints. "
            f'Remove the conflicting patterns from "{FAB_INCLUDE_KEY}".\n\n'
            f"Affected files:\n{files_list}"
        )
