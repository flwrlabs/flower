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
"""Tests for object refs."""


import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from .object_ref import OBJECT_REF_HELP_STR, load_app, validate


class LoadAppError(Exception):
    """Error raised while loading an app reference."""


@pytest.fixture(autouse=True)
def restore_import_state() -> Iterator[None]:
    """Restore import state after tests that modify it."""
    original_sys_path = list(sys.path)
    original_modules = set(sys.modules)

    yield

    sys.path[:] = original_sys_path
    for name in set(sys.modules) - original_modules:
        if name.startswith("object_ref_"):
            del sys.modules[name]


def _write_module(project_dir: Path, module_name: str, source: str) -> None:
    """Write a test module to project_dir."""
    (project_dir / f"{module_name}.py").write_text(source, encoding="utf-8")


def test_validate_object_reference() -> None:
    """Test that validate succeeds correctly."""
    # Prepare
    ref = "flwr.cli.run:run"

    # Execute
    is_valid, error = validate(ref)

    # Assert
    assert is_valid
    assert error is None


def test_validate_object_reference_from_project_dir(tmp_path: Path) -> None:
    """Test that validate succeeds for a module in a project dir."""
    # Prepare
    _write_module(tmp_path, "object_ref_valid_app", "app = object()\n")

    # Execute
    is_valid, error = validate("object_ref_valid_app:app", project_dir=tmp_path)

    # Assert
    assert is_valid
    assert error is None


def test_validate_object_reference_missing_module() -> None:
    """Test that validate fails for a missing module."""
    # Prepare
    ref = "object_ref_missing_module:app"

    # Execute
    is_valid, error = validate(ref)

    # Assert
    assert not is_valid
    assert (
        error == f"Unable to load module object_ref_missing_module{OBJECT_REF_HELP_STR}"
    )


def test_validate_object_reference_fails() -> None:
    """Test that validate fails correctly."""
    # Prepare
    ref = "flwr.cli.run:runa"

    # Execute
    is_valid, error = validate(ref)

    # Assert
    assert not is_valid
    assert (
        error
        == f"Unable to find attribute runa in module flwr.cli.run{OBJECT_REF_HELP_STR}"
    )


def test_validate_check_module_false() -> None:
    """Test that check_module=False only validates the reference shape."""
    # Execute
    is_valid, error = validate("object_ref_missing_module:app", check_module=False)
    is_invalid, invalid_error = validate("object_ref_missing_module")

    # Assert
    assert is_valid
    assert error is None
    assert not is_invalid
    assert (
        invalid_error == f"Missing attribute in object_ref_missing_module"
        f"{OBJECT_REF_HELP_STR}"
    )


def test_validate_does_not_execute_module(tmp_path: Path) -> None:
    """Test that validate does not execute module code."""
    # Prepare
    marker_path = tmp_path / "marker"
    _write_module(
        tmp_path,
        "object_ref_static_app",
        "from pathlib import Path\n"
        f"Path({str(marker_path)!r}).touch()\n"
        "app = object()\n",
    )

    # Execute
    is_valid, error = validate("object_ref_static_app:app", project_dir=tmp_path)

    # Assert
    assert is_valid
    assert error is None
    assert not marker_path.exists()


def test_load_app(tmp_path: Path) -> None:
    """Test that load_app loads the referenced object."""
    # Prepare
    _write_module(tmp_path, "object_ref_runtime_app", "app = 'loaded'\n")

    # Execute
    app = load_app("object_ref_runtime_app:app", LoadAppError, tmp_path)

    # Assert
    assert app == "loaded"


def test_load_app_dotted_attribute(tmp_path: Path) -> None:
    """Test that load_app supports dotted attributes."""
    # Prepare
    _write_module(
        tmp_path,
        "object_ref_nested_app",
        "class Wrapper:\n    value = 1\nwrapper = Wrapper()\n",
    )

    # Execute
    value = load_app("object_ref_nested_app:wrapper.value", LoadAppError, tmp_path)

    # Assert
    assert value == 1


def test_load_app_raises_error_type(tmp_path: Path) -> None:
    """Test that load_app raises the provided error type."""
    # Prepare
    _write_module(tmp_path, "object_ref_missing_attr_app", "app = object()\n")

    # Execute & assert
    with pytest.raises(LoadAppError) as exc:
        load_app("object_ref_missing_attr_app:missing", LoadAppError, tmp_path)

    assert (
        str(exc.value)
        == "Unable to load attribute missing from module object_ref_missing_attr_app"
        f"{OBJECT_REF_HELP_STR}"
    )


def test_load_app_leaves_project_dir_importable(tmp_path: Path) -> None:
    """Test that load_app leaves the project dir on sys.path."""
    # Prepare
    _write_module(tmp_path, "object_ref_importable_app", "app = object()\n")

    # Execute
    _ = load_app("object_ref_importable_app:app", LoadAppError, tmp_path)

    # Assert
    assert str(tmp_path.absolute()) in sys.path
    assert "object_ref_importable_app" in sys.modules


def test_validate_dotted_attribute(tmp_path: Path) -> None:
    """Test that validate accepts a dotted attribute when the root name exists.

    Static AST analysis cannot verify the full attribute chain without importing
    the module, so only the first segment is checked.
    """
    # Prepare: module defines `wrapper`; `wrapper.app` is only resolvable at runtime
    _write_module(
        tmp_path,
        "object_ref_dotted_valid",
        "class _Wrapper:\n    app = object()\nwrapper = _Wrapper()\n",
    )

    # Execute
    is_valid, error = validate(
        "object_ref_dotted_valid:wrapper.app", project_dir=tmp_path
    )

    # Assert
    assert is_valid
    assert error is None
