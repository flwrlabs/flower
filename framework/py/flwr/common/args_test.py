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
"""Tests for runtime dependency installation CLI arguments."""


import argparse

import pytest

from flwr.common.args import (
    add_args_flwr_app_common,
    add_args_runtime_dependency_install,
)
from flwr.common.constant import RUNTIME_DEPENDENCY_INSTALL


def test_runtime_dependency_install_args_defaults() -> None:
    """Verify runtime dependency installation args default values."""
    parser = argparse.ArgumentParser()
    add_args_runtime_dependency_install(parser)

    args = parser.parse_args([])

    assert args.runtime_dependency_install is RUNTIME_DEPENDENCY_INSTALL


def test_runtime_dependency_install_args_flags() -> None:
    """Verify runtime dependency installation args parse correctly."""
    parser = argparse.ArgumentParser()
    add_args_runtime_dependency_install(parser)

    args = parser.parse_args(["--allow-runtime-dependency-installation"])

    assert args.runtime_dependency_install is True


def test_flwr_app_common_args_require_token() -> None:
    """App process CLIs should require a token."""
    parser = argparse.ArgumentParser()
    add_args_flwr_app_common(parser)

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_flwr_app_common_args_parse_token() -> None:
    """App process CLIs should parse token and common flags."""
    parser = argparse.ArgumentParser()
    add_args_flwr_app_common(parser)

    args = parser.parse_args(
        [
            "--token",
            "test-token",
            "--insecure",
            "--parent-pid",
            "1234",
            "--allow-runtime-dependency-installation",
        ]
    )

    assert args.token == "test-token"
    assert args.insecure is True
    assert args.parent_pid == 1234
    assert args.runtime_dependency_install is True


def test_flwr_app_common_args_reject_run_once() -> None:
    """The removed deprecated flag should no longer parse."""
    parser = argparse.ArgumentParser()
    add_args_flwr_app_common(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--token", "test-token", "--run-once"])
