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
"""Test for Flower command line interface `new` command."""


import importlib
import io
import zipfile
from pathlib import Path
from unittest.mock import Mock

import click
import pytest
import typer

from .new import download_remote_app_via_api

new_module = importlib.import_module("flwr.cli.new.new")


@pytest.mark.parametrize(
    "value",
    [
        "user/app==1.2.3",  # missing '@'
        "@accountapp==1.2.3",  # missing slash
        "@account/app==1.2",  # bad version
        "@account/app==1.2.3.4",  # bad version
        "@account*/app==1.2.3",  # bad user id chars
        "@account/app*==1.2.3",  # bad app id chars
    ],
)
def test_download_remote_app_via_api_rejects_invalid_formats(value: str) -> None:
    """For an invalid string, the function should fail fast with
    click.ClickException(code=1)."""
    with pytest.raises(click.ClickException) as exc:
        download_remote_app_via_api(value)

    # Ensure we specifically exited with code 1
    assert exc.value.exit_code == 1


def test_download_remote_app_via_api_rejects_zip_slip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject app ZIP archives containing path traversal entries."""

    def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in entries:
                zf.writestr(name, content)
        return buf.getvalue()

    malicious_zip = _zip_bytes([("../evil.txt", b"x")])
    mock_response = Mock(content=malicious_zip, raise_for_status=lambda: None)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        new_module,
        "request_download_link",
        lambda *_args, **_kwargs: ("https://example.invalid/fake.zip", None, None),
    )
    monkeypatch.setattr(
        new_module.requests,
        "get",
        lambda *_args, **_kwargs: mock_response,
    )

    with pytest.raises(click.ClickException, match="Unsafe path in FAB archive"):
        download_remote_app_via_api("@account/app==1.2.3")


def test_download_remote_app_via_api_surfaces_platform_api_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Print compatibility warnings returned by Platform API."""

    def _zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in entries:
                zf.writestr(name, content)
        return buf.getvalue()

    app_zip = _zip_bytes([("app/README.md", b"hello")])
    mock_response = Mock(content=app_zip, raise_for_status=lambda: None)
    messages: list[tuple[str, bool]] = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        new_module,
        "request_download_link",
        lambda *_args, **_kwargs: (
            "https://example.invalid/fake.zip",
            None,
            "Using app version 1.0.0 because the latest published version "
            "requires a newer Flower version.",
        ),
    )
    monkeypatch.setattr(
        new_module.requests,
        "get",
        lambda *_args, **_kwargs: mock_response,
    )
    monkeypatch.setattr(
        typer,
        "secho",
        lambda message, *args, **kwargs: messages.append(
            (str(message), bool(kwargs.get("err", False)))
        ),
    )
    monkeypatch.setattr(new_module, "print_success_prompt", lambda *_args: None)

    download_remote_app_via_api("@account/app==1.2.3")

    assert (
        "Note: Using app version 1.0.0 because the latest published version "
        "requires a newer Flower version.",
        True,
    ) in messages
