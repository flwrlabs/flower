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
"""Test for Flower command line interface `run` command."""


import os
import textwrap
from pathlib import Path
from typing import Any

import click
import pytest

from .config_utils import load, validate_federation_in_project_config


def test_load_pyproject_toml_load_from_cwd(tmp_path: Path) -> None:
    """Test if load_template returns a string."""
    # Prepare
    pyproject_toml_content = """
        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [project]
        name = "fedgpt"
        version = "1.0.0"
        description = ""
        license = {text = "Apache License (2.0)"}
        dependencies = [
            "flwr[simulation]>=1.9.0,<2.0",
            "numpy>=1.21.0",
        ]

        [tool.flwr.app]
        publisher = "flwrlabs"

        [tool.flwr.app.components]
        serverapp = "fedgpt.server:app"
        clientapp = "fedgpt.client:app"
    """
    expected_config = {
        "build-system": {"build-backend": "hatchling.build", "requires": ["hatchling"]},
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": {"text": "Apache License (2.0)"},
            "dependencies": ["flwr[simulation]>=1.9.0,<2.0", "numpy>=1.21.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "components": {
                        "serverapp": "fedgpt.server:app",
                        "clientapp": "fedgpt.client:app",
                    },
                },
            },
        },
    }

    # Current directory
    origin = Path.cwd()

    try:
        # Change into the temporary directory
        os.chdir(tmp_path)
        with open("pyproject.toml", "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(pyproject_toml_content))

        # Execute
        config = load(toml_path=Path.cwd() / "pyproject.toml")

        # Assert
        assert config == expected_config
    finally:
        os.chdir(origin)


def test_load_pyproject_toml_from_path(tmp_path: Path) -> None:
    """Test if load_template returns a string."""
    # Prepare
    pyproject_toml_content = """
        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [project]
        name = "fedgpt"
        version = "1.0.0"
        description = ""
        license = {text = "Apache License (2.0)"}
        dependencies = [
            "flwr[simulation]>=1.9.0,<2.0",
            "numpy>=1.21.0",
        ]

        [tool.flwr.app]
        publisher = "flwrlabs"

        [tool.flwr.app.components]
        serverapp = "fedgpt.server:app"
        clientapp = "fedgpt.client:app"
    """
    expected_config = {
        "build-system": {"build-backend": "hatchling.build", "requires": ["hatchling"]},
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": {"text": "Apache License (2.0)"},
            "dependencies": ["flwr[simulation]>=1.9.0,<2.0", "numpy>=1.21.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "components": {
                        "serverapp": "fedgpt.server:app",
                        "clientapp": "fedgpt.client:app",
                    },
                },
            },
        },
    }

    # Current directory
    origin = Path.cwd()

    try:
        # Change into the temporary directory
        os.chdir(tmp_path)
        with open("pyproject.toml", "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(pyproject_toml_content))

        # Execute
        config = load(toml_path=tmp_path / "pyproject.toml")

        # Assert
        assert config == expected_config
    finally:
        os.chdir(origin)


def test_validate_federation_in_project_config() -> None:
    """Test that validate_federation_in_config succeeds correctly."""
    # Prepare - Test federation is None
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:run",
                    },
                },
                "federations": {
                    "default": "default_federation",
                    "default_federation": {"default_key": "default_val"},
                },
            },
        },
    }
    federation = None

    # Execute
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )

    # Assert
    assert federation == "default_federation"
    assert federation_config == {"default_key": "default_val"}

    federation = "new_federation"
    config["tool"]["flwr"]["federations"]["new_federation"] = {"new_key": "new_val"}

    # Execute
    federation, federation_config = validate_federation_in_project_config(
        federation, config
    )

    # Assert
    assert federation == "new_federation"
    assert federation_config == {"new_key": "new_val"}


def test_validate_federation_in_project_config_with_overrides() -> None:
    """Test that validate_federation_in_config works with overrides."""
    # Prepare - Test federation is None
    federation_config = {"k1": "v1", "k2": True, "grp": {"k3": 42.8}}
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:run",
                    },
                },
                "federations": {
                    "default": "default_federation",
                    "default_federation": federation_config,
                },
            },
        },
    }
    overrides = ["k1=false grp.k3=42.9", "k2='hello, world!'"]
    federation = None

    # Execute
    federation, federation_config = validate_federation_in_project_config(
        federation, config, overrides
    )

    # Assert
    assert federation == "default_federation"
    assert federation_config == {
        "k1": False,
        "k2": "hello, world!",
        "grp": {"k3": 42.9},
    }


def test_validate_federation_in_project_config_fail() -> None:
    """Test that validate_federation_in_config fails correctly."""

    def run_and_assert_exit(federation: str | None, config: dict[str, Any]) -> None:
        """Execute validation and assert exit code is 1."""
        with pytest.raises(click.ClickException) as excinfo:
            validate_federation_in_project_config(federation, config)
        assert excinfo.value.exit_code == 1

    # Prepare
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:run",
                    },
                },
                "federations": {},
            },
        },
    }
    federation = None

    # Test federation is None and no default federation is declared
    # Execute and assert
    run_and_assert_exit(federation, config)

    # Prepare - Test federation name is not in config
    federation = "fed_not_in_config"
    config["tool"]["flwr"]["federations"] = {"fed_in_config": {}}

    # Execute and assert
    run_and_assert_exit(federation, config)
