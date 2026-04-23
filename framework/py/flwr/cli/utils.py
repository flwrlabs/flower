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
"""Flower command line interface utils."""


import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any, cast

import click
import grpc
import pathspec
import typer
from rich.console import Console

from flwr.cli.typing import SuperLinkConnection
from flwr.common.constant import (
    ACCESS_TOKEN_KEY,
    AUTHN_TYPE_JSON_KEY,
    FEDERATION_NOT_FOUND_MESSAGE,
    NO_ACCOUNT_AUTH_MESSAGE,
    NO_ARTIFACT_PROVIDER_MESSAGE,
    NODE_NOT_FOUND_MESSAGE,
    PUBLIC_KEY_ALREADY_IN_USE_MESSAGE,
    PUBLIC_KEY_NOT_VALID,
    PULL_UNFINISHED_RUN_MESSAGE,
    REFRESH_TOKEN_KEY,
    RUN_ID_NOT_FOUND_MESSAGE,
    AuthnType,
    CliOutputFormat,
)
from flwr.common.grpc import (
    GRPC_MAX_MESSAGE_LENGTH,
    create_channel,
    on_channel_state_change,
)
from flwr.common.logger import print_json_error, redirect_output, restore_output
from flwr.proto.control_pb2_grpc import ControlStub  # pylint: disable=E0611
from flwr.supercore.constant import (
    APP_PUBLISH_EXCLUDE_PATTERNS,
    APP_PUBLISH_INCLUDE_PATTERNS,
    MAX_DIR_DEPTH,
    MAX_NAME_LENGTH,
)
from flwr.supercore.credential_store import get_credential_store
from flwr.supercore.utils import is_valid_name

from .auth_plugin import CliAuthPlugin, get_cli_plugin_class
from .cli_account_auth_interceptor import CliAccountAuthInterceptor
from .config_utils import load_certificate_in_connection
from .constant import AUTHN_TYPE_STORE_KEY
from .flower_config import read_superlink_connection
from .local_superlink import ensure_local_superlink


def print_json_to_stdout(data: str | Any) -> None:
    """Print JSON data to stdout, bypassing any output redirection.

    Use this function within the `cli_output_handler` context manager to print JSON
    output directly to the terminal, even when stdout is being captured.
    """
    if isinstance(data, str):
        Console(file=sys.__stdout__).print_json(data)
    else:
        Console(file=sys.__stdout__).print_json(data=data)


@contextmanager  # docsig: ignore=SIG503
def cli_output_handler(
    output_format: str = CliOutputFormat.DEFAULT,
) -> Iterator[bool]:
    """Context manager for handling CLI output in different formats.

    This context manager provides consistent output handling for CLI commands by:
    - Redirecting stdout/stderr when JSON format is requested
    - Catching and handling exceptions appropriately based on the output format

    Use the `print_json_to_stdout()` utility function to print JSON output that bypasses
    output redirection.
    """
    is_json = output_format == CliOutputFormat.JSON
    captured_output = StringIO()

    if is_json:
        redirect_output(captured_output)

    try:
        yield is_json
    except Exception as err:  # pylint: disable=broad-except
        if is_json:
            restore_output()
            print_json_error(captured_output.getvalue(), err)
        else:
            if isinstance(err, typer.Exit):
                raise  # Allow typer.Exit to escape normally
            raise click.ClickException(str(err)) from None
    finally:
        if is_json:
            restore_output()
        captured_output.close()


def prompt_text(
    text: str,
    predicate: Callable[[str], bool] = lambda _: True,
    default: str | None = None,
) -> str:
    """Ask user to enter text input.

    Parameters
    ----------
    text : str
        The prompt text to display to the user.
    predicate : Callable[[str], bool] (default: lambda _: True)
        A function to validate the user input. Default accepts all non-empty strings.
    default : str | None (default: None)
        Default value to use if user presses enter without input.

    Returns
    -------
    str
        The validated user input.
    """
    while True:
        result = typer.prompt(
            typer.style(f"\n💬 {text}", fg=typer.colors.MAGENTA, bold=True),
            default=default,
        )
        if predicate(result) and len(result) > 0:
            break
        print(typer.style("❌ Invalid entry", fg=typer.colors.RED, bold=True))

    return cast(str, result)


def prompt_options(text: str, options: list[str]) -> str:
    """Ask user to select one of the given options and return the selected item.

    Parameters
    ----------
    text : str
        The prompt text to display to the user.
    options : list[str]
        List of options to present to the user.

    Returns
    -------
    str
        The selected option from the list.
    """
    # Turn options into a list with index as in " [ 0] quickstart-pytorch"
    options_formatted = [
        " [ "
        + typer.style(index, fg=typer.colors.GREEN, bold=True)
        + "]"
        + f" {typer.style(name, fg=typer.colors.WHITE, bold=True)}"
        for index, name in enumerate(options)
    ]

    while True:
        index = typer.prompt(
            "\n"
            + typer.style(f"💬 {text}", fg=typer.colors.MAGENTA, bold=True)
            + "\n\n"
            + "\n".join(options_formatted)
            + "\n\n\n"
        )
        try:
            options[int(index)]  # pylint: disable=expression-not-assigned
            break
        except IndexError:
            print(typer.style("❌ Index out of range", fg=typer.colors.RED, bold=True))
            continue
        except ValueError:
            print(
                typer.style("❌ Please choose a number", fg=typer.colors.RED, bold=True)
            )
            continue

    result = options[int(index)]
    return result


def validate_project_name(name: str, target: str) -> None:
    """Validate a project-related name and raise ValueError if invalid."""
    valid, _ = is_valid_name(name)
    if not valid:
        raise ValueError(
            f'{target} "{name}" is invalid, '
            "a valid app name must start with a letter, "
            "and can only contain letters, digits, and hyphens. The name "
            f"must also be no longer than {MAX_NAME_LENGTH} characters."
        )


def get_sha256_hash(file_path_or_int: Path | int) -> str:
    """Calculate the SHA-256 hash of a file or integer.

    Parameters
    ----------
    file_path_or_int : Path | int
        Either a path to a file to hash, or an integer to convert to string and hash.

    Returns
    -------
    str
        The SHA-256 hash as a hexadecimal string.
    """
    sha256 = hashlib.sha256()
    if isinstance(file_path_or_int, Path):
        with open(file_path_or_int, "rb") as f:
            while True:
                data = f.read(65536)  # Read in 64kB blocks
                if not data:
                    break
                sha256.update(data)
    elif isinstance(file_path_or_int, int):
        sha256.update(str(file_path_or_int).encode())
    return sha256.hexdigest()


def get_authn_type(host: str) -> str:
    """Retrieve the authentication type for the given host from the credential store.

    `AuthnType.NOOP` is returned if no authentication type is found.
    """
    store = get_credential_store()
    authn_type = store.get(AUTHN_TYPE_STORE_KEY % host)
    if authn_type is None:
        return AuthnType.NOOP
    return authn_type.decode("utf-8")


def load_cli_auth_plugin_from_connection(
    host: str, authn_type: str | None = None
) -> CliAuthPlugin:
    """Load the CLI-side account auth plugin for the given connection.

    Parameters
    ----------
    host : str
        The SuperLink Control API address.
    authn_type : str | None
        Authentication type. If None, will be determined from config.

    Returns
    -------
    CliAuthPlugin
        The loaded authentication plugin instance.

    Raises
    ------
    click.ClickException
        If the authentication type is unknown.
    """
    # Determine the auth type if not provided
    # Only `flwr login` command can provide `authn_type` explicitly, as it can query the
    # SuperLink for the auth type.
    if authn_type is None:
        authn_type = get_authn_type(host)

    # Retrieve auth plugin class and instantiate it
    try:
        auth_plugin_class = get_cli_plugin_class(authn_type)
        return auth_plugin_class(host)
    except ValueError:
        raise click.ClickException(
            f"Unknown account authentication type: {authn_type}"
        ) from None


def get_executed_command() -> str:
    """Get the full command path from the current Click context.

    Traverses up the Click context hierarchy to build the complete command path.

    Returns
    -------
    str
        The full command path including the "flwr" prefix.
    """
    ctx: click.Context | None = click.get_current_context()
    cmd_parts = []
    while ctx is not None:
        if ctx.info_name:
            cmd_parts.append(ctx.info_name)
        ctx = ctx.parent
    cmd_parts.reverse()
    return " ".join(cmd_parts)


def init_channel_from_connection(
    connection: SuperLinkConnection, auth_plugin: CliAuthPlugin | None = None
) -> grpc.Channel:
    """Initialize gRPC channel to the Control API.

    Parameters
    ----------
    connection : SuperLinkConnection
        SuperLink connection configuration.
    auth_plugin : CliAuthPlugin | None (default: None)
        Authentication plugin instance for handling credentials.

    Returns
    -------
    grpc.Channel
        Configured gRPC channel with authentication interceptors.
    """
    connection = ensure_local_superlink(connection)
    address = cast(str, connection.address)

    root_certificates_bytes = load_certificate_in_connection(connection)

    # Load authentication plugin
    if auth_plugin is None:
        auth_plugin = load_cli_auth_plugin_from_connection(address)
    # Load tokens
    auth_plugin.load_tokens()

    # Create the gRPC channel
    channel = create_channel(
        server_address=address,
        insecure=connection.insecure,
        root_certificates=root_certificates_bytes,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
        interceptors=[CliAccountAuthInterceptor(auth_plugin)],
    )
    channel.subscribe(on_channel_state_change)
    return channel


@contextmanager  # docsig: disable=SIG503
def cli_output_control_stub(
    superlink: str | None,
    output_format: str = CliOutputFormat.DEFAULT,
) -> Iterator[tuple[ControlStub, bool]]:
    """Manage CLI output handling and Control API stub lifecycle.

    Parameters
    ----------
    superlink : str | None
        Name of the SuperLink connection.
    output_format : str
        Output format for CLI rendering.

    Yields
    ------
    tuple[ControlStub, bool]
        A tuple of (ControlStub, is_json), where `is_json` indicates JSON output.
    """
    with cli_output_handler(output_format=output_format) as is_json:
        superlink_connection = read_superlink_connection(superlink)
        channel = init_channel_from_connection(superlink_connection)
        try:
            yield ControlStub(channel), is_json
        finally:
            channel.close()


@contextmanager  # docsig: disable=SIG503
def flwr_cli_grpc_exc_handler(  # pylint: disable=too-many-branches
    custom_handler: Callable[[grpc.RpcError], None] | None = None,
) -> Iterator[None]:
    """Context manager to handle specific gRPC errors.

    Catches grpc.RpcError exceptions and translates them into user-friendly messages
    based on the error code and details.

    Parameters
    ----------
    custom_handler : Callable[[grpc.RpcError], None] | None (default: None)
        Optional custom handler called with the caught gRPC error before applying
        default Flower CLI error handling.

    Yields
    ------
    None
        Context manager yields nothing.

    Raises
    ------
    click.ClickException
        For handled gRPC errors, with user-friendly messages. Or raw gRPC error details
        will be shown.
    """
    try:
        yield
    except grpc.RpcError as e:
        if custom_handler is not None:
            custom_handler(e)
        if e.code() == grpc.StatusCode.UNAUTHENTICATED:
            raise click.ClickException(
                "Authentication failed. Please run `flwr login`"
                " to authenticate and try again."
            ) from None
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            if e.details() == NO_ACCOUNT_AUTH_MESSAGE:  # pylint: disable=E1101
                raise click.ClickException(
                    "Account authentication is not enabled on this SuperLink."
                ) from None
            if e.details() == NO_ARTIFACT_PROVIDER_MESSAGE:  # pylint: disable=E1101
                raise click.ClickException(
                    "The SuperLink does not support `flwr pull` command."
                ) from None
            raise click.ClickException(e.details()) from None  # pylint: disable=E1101
        if e.code() == grpc.StatusCode.PERMISSION_DENIED:
            # pylint: disable-next=E1101
            raise click.ClickException(f"Permission denied.\n{e.details()}") from None
        if e.code() == grpc.StatusCode.UNAVAILABLE:
            raise click.ClickException(
                "Connection to the SuperLink is unavailable. Please check your network "
                "connection and 'address' in the SuperLink connection configuration."
            ) from None
        if e.code() == grpc.StatusCode.NOT_FOUND:
            if e.details() == RUN_ID_NOT_FOUND_MESSAGE:  # pylint: disable=E1101
                raise click.ClickException("Run ID not found.") from None
            if e.details() == NODE_NOT_FOUND_MESSAGE:  # pylint: disable=E1101
                raise click.ClickException(
                    "Node ID not found for this account."
                ) from None
        if e.code() == grpc.StatusCode.FAILED_PRECONDITION:
            if e.details() == PULL_UNFINISHED_RUN_MESSAGE:  # pylint: disable=E1101
                raise click.ClickException(
                    "Run is not finished yet. Artifacts can only be pulled after "
                    "the run is finished. You can check the run status with `flwr ls`."
                ) from None
            if (
                e.details() == PUBLIC_KEY_ALREADY_IN_USE_MESSAGE
            ):  # pylint: disable=E1101
                raise click.ClickException(
                    "The provided public key is already in use by another SuperNode."
                ) from None
            if e.details() == PUBLIC_KEY_NOT_VALID:  # pylint: disable=E1101
                raise click.ClickException(
                    "The provided public key is invalid. Please provide a valid "
                    "NIST EC public key."
                ) from None
            patten = re.compile(FEDERATION_NOT_FOUND_MESSAGE.replace("%s", "(.+)"))
            if m := patten.match(e.details()):  # pylint: disable=E1101
                raise click.ClickException(
                    f"Federation '{m.group(1)}' does not exist. "
                    "Please verify the federation name and try again."
                ) from None

        # Log details from grpc error directly
        raise click.ClickException(f"{e.details()}") from None


def build_pathspec(patterns: Iterable[str]) -> pathspec.PathSpec:
    """Build a PathSpec from a list of GitIgnore-style patterns.

    Parameters
    ----------
    patterns : Iterable[str]
        Iterable of GitIgnore-style pattern strings.

    Returns
    -------
    pathspec.PathSpec
        Compiled PathSpec object for pattern matching.
    """
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def load_gitignore_patterns(file: Path | bytes) -> list[str]:
    """Load gitignore patterns from .gitignore file bytes.

    Parameters
    ----------
    file : Path | bytes
        The path to a .gitignore file or its bytes content.

    Returns
    -------
    list[str]
        List of gitignore patterns.
        Returns empty list if content can't be decoded or the file does not exist.
    """
    try:
        if isinstance(file, Path):
            content = file.read_text(encoding="utf-8")
        else:
            content = file.decode("utf-8")
        patterns = [
            line.strip()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return patterns
    except (UnicodeDecodeError, OSError):
        return []


def depth_of(relative_path: Path) -> int:
    """Return the directory depth of a relative path."""
    return max(0, len(relative_path.parts) - 1)


def collect_files(root: Path) -> dict[str, Path]:
    """Collect all files under the root directory and return a mapping of relative POSIX
    paths to absolute Paths.

    Symlinks (both files and directories) are ignored, and only paths that resolve
    within ``root`` are included. The traversal uses ``os.walk`` with
    ``followlinks=False`` so symlinked directories are never entered. The relative
    paths are in POSIX format (using forward slashes) for consistency across platforms.
    """
    resolved_root = root.resolve()
    files: dict[str, Path] = {}
    for dirpath, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            path = Path(dirpath) / filename
            # Skip symlink files
            if path.is_symlink():
                continue
            # Skip paths that resolve outside the root directory
            if not path.resolve().is_relative_to(resolved_root):
                continue
            relative_path = path.relative_to(root).as_posix()
            files[relative_path] = path
    return files


def filter_paths_for_publish(
    files: Mapping[str, Path | bytes],
) -> dict[str, Path | bytes]:
    """Filter paths for app publishing, using publish-specific include/exclude rules.

    Parameters
    ----------
    files : Mapping[str, Path | bytes]
        Mapping of POSIX-style relative paths to file contents (as Path or bytes).

    Returns
    -------
    dict[str, Path | bytes]
        Filtered mapping of paths to contents that match publish include/exclude rules.

    Raises
    ------
    ValueError
        Raised if any path exceeds the maximum directory depth.
    """
    # Load gitignore patterns if exists
    gitignore_patterns = tuple(load_gitignore_patterns(files.get(".gitignore", b"")))
    gitignore_spec = build_pathspec(gitignore_patterns)

    # Build include/exclude pathspecs for app publish
    include_spec = build_pathspec(APP_PUBLISH_INCLUDE_PATTERNS)
    exclude_spec = build_pathspec(APP_PUBLISH_EXCLUDE_PATTERNS)

    # Apply filtering
    filtered_paths = include_spec.match_files(files.keys())
    filtered_paths = exclude_spec.match_files(filtered_paths, negate=True)
    filtered_paths = gitignore_spec.match_files(filtered_paths, negate=True)

    # Collect filtered files and check directory depth
    ret_files = {}
    for rel_pth in cast(Iterable[str], filtered_paths):
        if depth_of(Path(rel_pth)) > MAX_DIR_DEPTH:
            raise ValueError(
                f"'{rel_pth}' in the project exceeds the maximum directory depth "
                f"of {MAX_DIR_DEPTH}. Consider refactoring your project structure to "
                "reduce nesting."
            )
        ret_files[rel_pth] = files[rel_pth]
    return ret_files


def validate_credentials_content(creds_path: Path) -> str:
    """Load and validate the credentials file content.

    Ensures required keys exist:
      - AUTHN_TYPE_JSON_KEY
      - ACCESS_TOKEN_KEY
      - REFRESH_TOKEN_KEY
    """
    try:
        creds: dict[str, str] = json.loads(creds_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        raise click.ClickException(
            f"Invalid credentials file at '{creds_path}': {err}"
        ) from err

    required_keys = [AUTHN_TYPE_JSON_KEY, ACCESS_TOKEN_KEY, REFRESH_TOKEN_KEY]
    missing = [key for key in required_keys if key not in creds]

    if missing:
        raise click.ClickException(
            f"Credentials file '{creds_path}' is missing "
            f"required key(s): {', '.join(missing)}. Please log in again."
        )

    return creds[ACCESS_TOKEN_KEY]


def validate_federation_name(name: str) -> tuple[bool, str]:
    """Validate a federation name based on specific security and formatting rules.

    The same validation rules as project names are applied.

    Parameters
    ----------
    name : str
        The federation name to validate.

    Returns
    -------
        tuple: (bool, str)
               - A boolean indicating if the name is valid (True/False).
               - A string containing a success message or a detailed error message.
    """
    return is_valid_name(name)
