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
"""Test util functions handling Flower config."""


import os
import tempfile
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from flwr.app.user_config import UserConfig

from .config import (
    flatten_dict,
    fuse_dicts,
    get_project_config,
    get_project_dir,
    parse_config_args,
    unflatten_dict,
    validate_config,
    validate_fields_in_config,
)

# Mock constants
FAB_CONFIG_FILE = "pyproject.toml"


def test_get_project_dir_invalid_fab_id() -> None:
    """Test get_project_dir with an invalid fab_id."""
    with pytest.raises(ValueError):
        get_project_dir(
            "invalid_fab_id",
            "1.0.0",
            "03840e932bf61247c1231f0aec9e8ec5f041ed5516fb23638f24d25f3a007acd",
        )


def test_get_project_dir_valid() -> None:
    """Test get_project_dir with an valid fab_id and version."""
    with patch.dict(os.environ, {"FLWR_HOME": "/test_home"}):
        app_path = get_project_dir(
            "app_name/user",
            "1.0.0",
            "03840e932bf61247c1231f0aec9e8ec5f041ed5516fb23638f24d25f3a007acd",
        )
    assert app_path == Path("/test_home") / "apps" / "app_name.user.1.0.0.03840e93"


def test_get_project_config_file_not_found() -> None:
    """Test get_project_config when the configuration file is not found."""
    with pytest.raises(FileNotFoundError):
        get_project_config("/invalid/dir")


def test_fuse_dicts() -> None:
    """Test that fuse_dicts works correctly."""
    dict1 = {"a": 1, "b": 2, "c": 3, "d": {"e": 4}}
    dict2 = {"b": 4, "c": 5, "d": {"e": 6}}
    expected = {"a": 1, "b": 4, "c": 5, "d": {"e": 6}}
    assert fuse_dicts(dict1, dict2) == expected


def test_fuse_dicts_key_mismatch() -> None:
    """Test that fuse_dicts fails with key mismatch."""
    dict1 = {"a": 1, "b": 2, "c": 3}
    dict2 = {"b": 4, "c": 5, "d": 6}

    with pytest.raises(ValueError):
        fuse_dicts(dict1, dict2)


def test_get_fused_config_valid(tmp_path: Path) -> None:
    """Test get_project_config when the configuration file is not found."""
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

        [tool.flwr.app.config]
        num_server_rounds = 10
        momentum = 0.1
        lr = 0.01
        progress_bar = true
        serverapp.test = "key"

        [tool.flwr.app.config.clientapp]
        test = "key"
    """
    overrides: UserConfig = {
        "num_server_rounds": 5,
        "lr": 0.2,
        "serverapp.test": "overriden",
    }
    expected_config = {
        "num_server_rounds": 5,
        "momentum": 0.1,
        "lr": 0.2,
        "progress_bar": True,
        "serverapp.test": "overriden",
        "clientapp.test": "key",
    }
    # Current directory
    origin = Path.cwd()

    try:
        # Change into the temporary directory
        os.chdir(tmp_path)
        with open(FAB_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(pyproject_toml_content))

        # Execute
        default_config = get_project_config(tmp_path)["tool"]["flwr"]["app"].get(
            "config", {}
        )

        config = fuse_dicts(flatten_dict(default_config), overrides)

        # Assert
        assert config == expected_config
    finally:
        os.chdir(origin)


def test_get_project_config_file_valid(tmp_path: Path) -> None:
    """Test get_project_config when the configuration file is not found."""
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

        [tool.flwr.app.config]
        num_server_rounds = 10
        momentum = 0.1
        progress_bar = true
        lr = "0.01"
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
                    "config": {
                        "num_server_rounds": 10,
                        "momentum": 0.1,
                        "progress_bar": True,
                        "lr": "0.01",
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
        with open(FAB_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(pyproject_toml_content))

        # Execute
        config = get_project_config(tmp_path)

        # Assert
        assert config == expected_config
    finally:
        os.chdir(origin)


def test_flatten_dict() -> None:
    """Test flatten_dict with a nested dictionary."""
    raw_dict = {"a": {"b": {"c": "d"}}, "e": "f"}
    expected = {"a.b.c": "d", "e": "f"}
    assert flatten_dict(raw_dict) == expected


def test_unflatten_dict() -> None:
    """Test unflatten_dict with a flat dictionary."""
    raw_dict = {"a.b.c": "d", "e": "f"}
    expected = {"a": {"b": {"c": "d"}}, "e": "f"}
    assert unflatten_dict(raw_dict) == expected


def test_parse_config_args_none() -> None:
    """Test parse_config_args with None as input."""
    assert not parse_config_args(None)


def test_parse_config_args_overrides() -> None:
    """Test parse_config_args with key-value pairs."""
    assert parse_config_args(
        ["key1='value1' key2='value2'", "key3=1", "key4=2.0 key5=true key6='value6'"]
    ) == {
        "key1": "value1",
        "key2": "value2",
        "key3": 1,
        "key4": 2.0,
        "key5": True,
        "key6": "value6",
    }


def test_parse_config_args_from_toml_file() -> None:
    """Test if a toml passed to --run-config it is loaded and fused correctly."""
    # Will be saved as a temp .toml file
    toml_config = """
        num-server-rounds = 10
        momentum = 0.1
        verbose = true
    """
    # This is the UserConfig that would be extracted from pyproject.toml
    initial_run_config: UserConfig = {
        "num-server-rounds": 5,
        "momentum": 0.2,
        "dataset": "my-fancy-dataset",
        "verbose": False,
    }
    expected_config = {
        "num-server-rounds": 10,
        "momentum": 0.1,
        "dataset": "my-fancy-dataset",
        "verbose": True,
    }

    # Create a temporary directory using a context manager
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary TOML file within that directory
        toml_config_file = os.path.join(temp_dir, "extra_config.toml")

        # Write the data to the TOML file
        with open(toml_config_file, "w", encoding="utf-8") as toml_file:
            toml_file.write(textwrap.dedent(toml_config))

        # Parse config (this mimics what `--run-config path/to/config.toml` does)
        config_from_toml = parse_config_args([toml_config_file])
        # Fuse
        config = fuse_dicts(initial_run_config, config_from_toml)

        # Assert
        assert config == expected_config


def test_parse_config_args_passing_toml_and_key_value() -> None:
    """Test that passing a toml and key-value configs aren't allowed."""
    config = ["my-other-config.toml", "lr=0.1", "epochs=99"]
    with pytest.raises(ValueError):
        parse_config_args(config)


def test_validate_pyproject_toml_fields_empty() -> None:
    """Test that validate_pyproject_toml_fields fails correctly."""
    # Prepare
    config: dict[str, Any] = {}

    # Execute
    is_valid, errors, warnings = validate_fields_in_config(config)

    # Assert
    assert not is_valid
    assert len(errors) == 2
    assert len(warnings) == 0


def test_validate_pyproject_toml_fields_no_flower() -> None:
    """Test that validate_pyproject_toml_fields fails correctly."""
    # Prepare
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        }
    }

    # Execute
    is_valid, errors, warnings = validate_fields_in_config(config)

    # Assert
    assert not is_valid
    assert len(errors) == 1
    assert len(warnings) == 0


def test_validate_pyproject_toml_fields_no_flower_components() -> None:
    """Test that validate_pyproject_toml_fields fails correctly."""
    # Prepare
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        },
        "tool": {"flwr": {"app": {}}},
    }

    # Execute
    is_valid, errors, warnings = validate_fields_in_config(config)

    # Assert
    assert not is_valid
    assert len(errors) == 2
    assert len(warnings) == 0


def test_validate_pyproject_toml_fields_no_server_and_client_app() -> None:
    """Test that validate_pyproject_toml_fields fails correctly."""
    # Prepare
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": "",
            "authors": [],
        },
        "tool": {"flwr": {"app": {"components": {}}}},
    }

    # Execute
    is_valid, errors, warnings = validate_fields_in_config(config)

    # Assert
    assert not is_valid
    assert len(errors) == 3
    assert len(warnings) == 0


def test_validate_pyproject_toml_fields() -> None:
    """Test that validate_pyproject_toml_fields succeeds correctly."""
    # Prepare
    config = {
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
                    "components": {"serverapp": "", "clientapp": ""},
                },
            },
        },
    }

    # Execute
    is_valid, errors, warnings = validate_fields_in_config(config)

    # Assert
    assert is_valid
    assert len(errors) == 0
    assert len(warnings) == 0


@pytest.mark.parametrize("key", ["fab-include", "fab-exclude"])
@pytest.mark.parametrize(
    "value, valid",
    [
        (["**/*.py", "**/*.md"], True),
        (["tests/**"], True),
        ("not-a-list", False),
        (123, False),
        ({"a": "b"}, False),
        (["valid", ""], False),
        (["valid", "  "], False),
        (["valid", 123], False),
        ([None], False),
    ],
)
def test_validate_fab_pattern_field(key: str, value: Any, valid: bool) -> None:
    """Test fab-include/fab-exclude validation for valid and invalid values."""
    # Prepare
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.1",
            "description": "",
            "license": "",
            "authors": [],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    key: value,
                    "components": {"serverapp": "", "clientapp": ""},
                },
            },
        },
    }

    # Execute
    is_valid, errors, _ = validate_fields_in_config(config)

    # Assert
    if valid:
        assert is_valid
        assert not any(key in e for e in errors)
    else:
        assert not is_valid
        assert any(key in e for e in errors)


def test_validate_pyproject_toml() -> None:
    """Test that validate_pyproject_toml succeeds correctly."""
    # Prepare
    config = {
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
            },
        },
    }

    # Execute
    is_valid, errors, warnings = validate_config(config)

    # Assert
    assert is_valid
    assert not errors
    assert not warnings


def test_validate_pyproject_toml_with_fab_format_version_derives_metadata() -> None:
    """Test fab-format-version=1 succeeds without mutating authored metadata."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": {"file": "LICENSE"},
            "dependencies": ["flwr[simulation]>=1.26.0,<=1.28.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "flwr-version-target": "1.27.0",
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:run",
                    },
                },
            },
        },
    }

    is_valid, errors, warnings = validate_config(config)

    assert is_valid
    assert not errors
    assert not warnings


def test_v1_fab_format_requires_flwr_dependency() -> None:
    """Test fab-format-version=1 requires a flwr dependency."""
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": {"file": "LICENSE"},
            "dependencies": ["numpy>=1.26.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "flwr-version-target": "1.27.0",
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:run",
                    },
                },
            },
        },
    }

    is_valid, errors, warnings = validate_config(config)

    assert not is_valid
    assert len(errors) == 1
    assert 'Missing "flwr" dependency' in errors[0]
    assert not warnings


def test_v1_fab_format_rejects_exclusive_lower_bound() -> None:
    """Test fab-format-version=1 rejects exclusive lower bounds."""
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": {"file": "LICENSE"},
            "dependencies": ["flwr>1.26.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "flwr-version-target": "1.27.0",
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:run",
                    },
                },
            },
        },
    }

    is_valid, errors, warnings = validate_config(config)

    assert not is_valid
    assert len(errors) == 1
    assert "inclusive lower bound" in errors[0]
    assert not warnings


def test_v1_fab_format_rejects_target_outside_declared_range() -> None:
    """Test fab-format-version=1 rejects targets outside the flwr specifier."""
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": {"file": "LICENSE"},
            "dependencies": ["flwr>=1.26.0,<1.28.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "flwr-version-target": "2.0.0",
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:run",
                    },
                },
            },
        },
    }

    is_valid, errors, warnings = validate_config(config)

    assert not is_valid
    assert len(errors) == 1
    assert 'must satisfy the declared "flwr" dependency specifier' in errors[0]
    assert not warnings


def test_v1_fab_format_accepts_additional_specifiers_with_lower_bound() -> None:
    """Test fab-format-version=1 accepts extra specifiers when `>=` is present."""
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": {"file": "LICENSE"},
            "dependencies": ["flwr>=1.26.0,==1.27.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "flwr-version-target": "1.27.0",
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:run",
                    },
                },
            },
        },
    }

    is_valid, errors, warnings = validate_config(config)

    assert is_valid
    assert not errors
    assert not warnings


def test_v1_fab_format_requires_target_version() -> None:
    """Test fab-format-version=1 requires flwr-version-target."""
    config = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "description": "",
            "license": {"file": "LICENSE"},
            "dependencies": ["flwr>=1.26.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "components": {
                        "serverapp": "flwr.cli.run:run",
                        "clientapp": "flwr.cli.run:run",
                    },
                },
            },
        },
    }

    is_valid, errors, warnings = validate_config(config)

    assert not is_valid
    assert len(errors) == 1
    assert "flwr-version-target" in errors[0]
    assert not warnings


def test_validate_pyproject_toml_fail() -> None:
    """Test that validate_pyproject_toml fails correctly."""
    # Prepare
    config = {
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
                        "clientapp": "flwr.cli.run:runa",
                    },
                },
            },
        },
    }

    # Execute
    is_valid, errors, warnings = validate_config(config)

    # Assert
    assert not is_valid
    assert len(errors) == 1
    assert len(warnings) == 0
