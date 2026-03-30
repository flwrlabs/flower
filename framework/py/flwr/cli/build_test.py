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
"""Test for Flower command line interface `build` command."""


from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from .build import build_fab_from_files

_DUMMY_PY = b"print('ok')\n"


def _make_files(
    app_toml: str = "",
    /,
    **extra_files: bytes,
) -> dict[str, bytes | Path]:
    """Build a minimal files dict with the given [tool.flwr.app] fragment."""
    pyproject = f'[project]\nname = "app"\nversion = "1.0.0"\n{app_toml}'
    return {"pyproject.toml": pyproject.encode(), **extra_files}


def _build_entries(files: dict[str, bytes | Path]) -> set[str]:
    """Build a FAB and return the set of archive entry names."""
    fab, _ = build_fab_from_files(files)
    with ZipFile(BytesIO(fab), "r") as zf:
        return set(zf.namelist())


def test_build_fab_from_files_includes_root_license_and_pyproject() -> None:
    """Test root LICENSE and pyproject.toml are included in FAB when present."""
    files = _make_files(**{"client.py": _DUMMY_PY, "LICENSE": b"Apache-2.0\n"})

    fab, _ = build_fab_from_files(files)

    with ZipFile(BytesIO(fab), "r") as zf:
        names = set(zf.namelist())
        content_manifest = zf.read(".info/CONTENT").decode("utf-8")

    assert "LICENSE" in names
    assert "LICENSE," in content_manifest
    assert "pyproject.toml" in names
    assert "pyproject.toml," in content_manifest


def test_build_fab_from_files_missing_pyproject_raises() -> None:
    """Test that missing pyproject.toml raises ValueError."""
    with pytest.raises(ValueError, match="pyproject.toml"):
        build_fab_from_files({"client.py": _DUMMY_PY})


def test_build_fab_from_files_rejects_invalid_project_name() -> None:
    """Test shared FAB build rejects invalid project names."""
    files = {
        "pyproject.toml": (
            b'[project]\nname = "app_numpy33"\nversion = "1.0.0"\n'
            b'[tool.flwr.app]\npublisher = "alice"\n'
        ),
        "client.py": _DUMMY_PY,
    }

    with pytest.raises(ValueError, match="valid app name must start"):
        build_fab_from_files(files)


def test_build_fab_from_files_missing_project_name_raises() -> None:
    """Test shared FAB build rejects missing project names cleanly."""
    files = {
        "pyproject.toml": (
            b'[project]\nversion = "1.0.0"\n[tool.flwr.app]\npublisher = "alice"\n'
        ),
        "client.py": _DUMMY_PY,
    }

    with pytest.raises(ValueError, match='Property "name" missing in \\[project\\]'):
        build_fab_from_files(files)


def test_build_fab_from_files_non_string_project_name_raises() -> None:
    """Test shared FAB build rejects non-string project names cleanly."""
    files = {
        "pyproject.toml": (
            b'[project]\nname = 123\nversion = "1.0.0"\n'
            b'[tool.flwr.app]\npublisher = "alice"\n'
        ),
        "client.py": _DUMMY_PY,
    }

    with pytest.raises(ValueError, match='Property "name" missing in \\[project\\]'):
        build_fab_from_files(files)


def test_build_fab_from_files_defaults_fab_format_version() -> None:
    """Test missing fab-format-version defaults in returned metadata."""
    files = _make_files(**{"client.py": _DUMMY_PY})

    _, metadata = build_fab_from_files(files)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min is None


def test_build_fab_from_files_ignores_target_for_version_zero() -> None:
    """Test fab-format-version=0 ignores flwr-version-target."""
    files = _make_files(
        '\n[tool.flwr.app]\npublisher = "alice"\n'
        'fab-format-version = 0\nflwr-version-target = "1.27.1"\n',
        **{"client.py": _DUMMY_PY},
    )

    _, metadata = build_fab_from_files(files)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min is None
    assert metadata.flwr_version_target is None


def test_build_fab_from_files_derives_flwr_minimum() -> None:
    """Test fab-format-version=1 derives only the lower-bound metadata."""
    files = _make_files(
        'license = { file = "LICENSE" }\n'
        'dependencies = ["flwr[simulation]>=1.26.0,<=1.28.0", "numpy>=1.0.0"]\n'
        '\n[tool.flwr.app]\npublisher = "alice"\n'
        'fab-format-version = 1\nflwr-version-target = "1.27.1"\n',
        **{"client.py": _DUMMY_PY, "LICENSE": b"Apache-2.0\n"},
    )

    _, metadata = build_fab_from_files(files)

    assert metadata.fab_format_version == 1
    assert metadata.flwr_version_min == "1.26.0"
    assert metadata.flwr_version_target == "1.27.1"


def test_build_fab_from_files_rejects_unsupported_fab_format_version() -> None:
    """Test build fails for unsupported fab-format-version values."""
    files = _make_files(
        '\n[tool.flwr.app]\npublisher = "alice"\nfab-format-version = 2\n',
        **{"client.py": _DUMMY_PY},
    )

    with pytest.raises(ValueError, match="Unsupported"):
        build_fab_from_files(files)


def test_build_fab_from_files_skips_unsupported_bounds_for_version_zero() -> None:
    """Test fab-format-version=0 ignores flwr-version-target without derivation."""
    files = _make_files(
        'dependencies = ["flwr>1.26.0"]\n'
        '\n[tool.flwr.app]\npublisher = "alice"\n'
        'fab-format-version = 0\nflwr-version-target = "1.27.1"\n',
        **{"client.py": _DUMMY_PY},
    )

    _, metadata = build_fab_from_files(files)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min is None
    assert metadata.flwr_version_target is None


def test_build_fab_from_files_rejects_unsupported_flwr_specifier() -> None:
    """Test build fails for fab-format-version=1 with an exclusive lower bound."""
    files = _make_files(
        'license = { file = "LICENSE" }\n'
        'dependencies = ["flwr>1.26.0"]\n'
        '\n[tool.flwr.app]\npublisher = "alice"\n'
        'fab-format-version = 1\nflwr-version-target = "1.27.1"\n',
        **{"client.py": _DUMMY_PY, "LICENSE": b"Apache-2.0\n"},
    )

    with pytest.raises(ValueError, match="inclusive lower bound"):
        build_fab_from_files(files)


def test_build_fab_from_files_rejects_target_outside_declared_range() -> None:
    """Test build fails when the target does not satisfy the flwr specifier."""
    files = _make_files(
        'license = { file = "LICENSE" }\n'
        'dependencies = ["flwr>=1.26.0,<1.28.0"]\n'
        '\n[tool.flwr.app]\npublisher = "alice"\n'
        'fab-format-version = 1\nflwr-version-target = "2.0.0"\n',
        **{"client.py": _DUMMY_PY, "LICENSE": b"Apache-2.0\n"},
    )

    with pytest.raises(ValueError, match='must satisfy the declared "flwr"'):
        build_fab_from_files(files)


def test_build_fab_from_files_uses_highest_inclusive_lower_bound() -> None:
    """Test build derives the highest declared `>=` lower bound."""
    files = _make_files(
        'license = { file = "LICENSE" }\n'
        'dependencies = ["flwr>=1.26.0,>=1.27.0,==1.27.0"]\n'
        '\n[tool.flwr.app]\npublisher = "alice"\n'
        'fab-format-version = 1\nflwr-version-target = "1.27.0"\n',
        **{"client.py": _DUMMY_PY, "LICENSE": b"Apache-2.0\n"},
    )

    _, metadata = build_fab_from_files(files)

    assert metadata.fab_format_version == 1
    assert metadata.flwr_version_min == "1.27.0"
    assert metadata.flwr_version_target == "1.27.0"


def test_build_fab_from_files_requires_target_for_version_one() -> None:
    """Test build fails for fab-format-version=1 without flwr-version-target."""
    files = _make_files(
        'license = { file = "LICENSE" }\n'
        'dependencies = ["flwr>=1.26.0"]\n'
        '\n[tool.flwr.app]\npublisher = "alice"\nfab-format-version = 1\n',
        **{"client.py": _DUMMY_PY, "LICENSE": b"Apache-2.0\n"},
    )

    with pytest.raises(ValueError, match="flwr-version-target"):
        build_fab_from_files(files)


def test_build_fab_from_files_rejects_v1_without_license_file_reference() -> None:
    """Test fab-format-version=1 requires [project].license.file."""
    files = _make_files(
        'dependencies = ["flwr>=1.26.0"]\n'
        '\n[tool.flwr.app]\npublisher = "alice"\n'
        'fab-format-version = 1\nflwr-version-target = "1.27.1"\n',
        **{"client.py": _DUMMY_PY},
    )

    with pytest.raises(ValueError, match=r"\[project\]\.license"):
        build_fab_from_files(files)


def test_build_fab_from_files_rejects_v1_when_license_file_missing() -> None:
    """Test fab-format-version=1 requires the declared license file in the FAB."""
    files = _make_files(
        'license = { file = "LICENSE" }\n'
        'dependencies = ["flwr>=1.26.0"]\n'
        '\n[tool.flwr.app]\npublisher = "alice"\n'
        'fab-format-version = 1\nflwr-version-target = "1.27.1"\n',
        **{"client.py": _DUMMY_PY},
    )

    with pytest.raises(ValueError, match="included in the FAB"):
        build_fab_from_files(files)


def test_build_fab_from_files_rejects_v1_when_license_file_is_excluded() -> None:
    """Test fab-format-version=1 fails when fab-exclude removes the license file."""
    files = _make_files(
        'license = { file = "LICENSE" }\n'
        'dependencies = ["flwr>=1.26.0"]\n'
        '\n[tool.flwr.app]\npublisher = "alice"\n'
        'fab-format-version = 1\nflwr-version-target = "1.27.1"\n'
        'fab-exclude = ["LICENSE"]\n',
        **{"client.py": _DUMMY_PY, "LICENSE": b"Apache-2.0\n"},
    )

    with pytest.raises(ValueError, match="included in the FAB"):
        build_fab_from_files(files)


def test_build_fab_from_files_accepts_v1_with_license_md() -> None:
    """Test fab-format-version=1 accepts LICENSE.md as the declared license file."""
    files = _make_files(
        'license = { file = "LICENSE.md" }\n'
        'dependencies = ["flwr>=1.26.0"]\n'
        '\n[tool.flwr.app]\npublisher = "alice"\n'
        'fab-format-version = 1\nflwr-version-target = "1.27.1"\n',
        **{"client.py": _DUMMY_PY, "LICENSE.md": b"# Apache-2.0\n"},
    )

    _, metadata = build_fab_from_files(files)

    assert metadata.fab_format_version == 1
    assert metadata.flwr_version_min == "1.26.0"
    assert metadata.flwr_version_target == "1.27.1"


def test_build_fab_from_files_without_fab_include_uses_all_then_builtin() -> None:
    """Test missing fab-include considers all files before built-in filtering."""
    files = _make_files(
        **{
            "client.py": _DUMMY_PY,
            "README.md": b"# docs\n",
            "data.mock": b"not included",
            "config.json": b'{"a": 1}',
        },
    )
    entries = _build_entries(files)

    assert "client.py" in entries
    assert "README.md" in entries
    assert "data.mock" not in entries
    assert "config.json" in entries


def test_build_fab_from_files_fab_include_is_constrained_by_builtin_include() -> None:
    """Test fab-include that only matches publish-excluded files raises ValueError."""
    files = _make_files(
        '\n[tool.flwr.app]\nfab-include = ["**/*.mock"]\n',
        **{"client.py": _DUMMY_PY, "data.mock": b"not included"},
    )

    with pytest.raises(ValueError, match="did not match any files"):
        build_fab_from_files(files)


def test_build_fab_from_files_fab_include_toml_does_not_raise_for_pyproject() -> None:
    """Test fab-include matching pyproject.toml does not raise a false conflict error.

    pyproject.toml is excluded by built-in constraints but always re-added to the FAB
    via a separate rewrite path, so it must not count toward the built_in_removed total.
    """
    files = _make_files(
        '\n[tool.flwr.app]\nfab-include = ["**/*.toml"]\n',
        **{"client.py": _DUMMY_PY},
    )

    # Should not raise even though pyproject.toml matches fab-include and is in
    # FAB_EXCLUDE_PATTERNS, because it is unconditionally included in the FAB.
    entries = _build_entries(files)
    assert "pyproject.toml" in entries


def test_build_fab_from_files_fab_exclude_only_removes_files() -> None:
    """Test fab-exclude removes files selected by other steps."""
    files = _make_files(
        '\n[tool.flwr.app]\nfab-exclude = ["**/*.md"]\n',
        **{"client.py": _DUMMY_PY, "README.md": b"# docs\n"},
    )
    entries = _build_entries(files)

    assert "client.py" in entries
    assert "README.md" not in entries


@pytest.mark.parametrize("key", ["fab-include", "fab-exclude"])
def test_build_fab_from_files_empty_fab_pattern_raises(key: str) -> None:
    """Test empty fab-include/fab-exclude raises ValueError."""
    files = _make_files(
        f"\n[tool.flwr.app]\n{key} = []\n",
        **{"client.py": _DUMMY_PY},
    )

    with pytest.raises(ValueError, match="must not be an empty list"):
        build_fab_from_files(files)


@pytest.mark.parametrize("key", ["fab-include", "fab-exclude"])
def test_build_fab_from_files_raises_on_unresolved_pattern(key: str) -> None:
    """Test build fails when a FAB pattern matches no files."""
    files = _make_files(
        f'\n[tool.flwr.app]\n{key} = ["no_such_dir/**/*.py"]\n',
        **{"client.py": _DUMMY_PY},
    )

    with pytest.raises(ValueError, match="did not match any files"):
        build_fab_from_files(files)


def test_build_fab_from_files_exclude_prevails_over_include() -> None:
    """Test user-defined exclude prevails when it overlaps with include."""
    files = _make_files(
        '\n[tool.flwr.app]\nfab-include = ["**/*.py"]\nfab-exclude = ["client.py"]\n',
        **{"client.py": _DUMMY_PY, "server.py": _DUMMY_PY},
    )
    entries = _build_entries(files)

    assert "server.py" in entries
    assert "client.py" not in entries
