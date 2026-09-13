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
"""Tests for FAB format version validation and metadata derivation."""


from typing import Any

import pytest

from .fab_format_version import normalize_and_validate_fab_format


def test_normalize_and_validate_fab_format_rejects_unsupported_version() -> None:
    """Test unsupported fab-format-version values fail explicitly."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 2,
                }
            }
        },
    }

    with pytest.raises(ValueError, match="Unsupported"):
        normalize_and_validate_fab_format(config)


def test_normalize_and_validate_fab_format_ignores_target_for_version_zero() -> None:
    """Test flwr-version-target is ignored for fab-format-version=0."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 0,
                    "flwr-version-target": "1.27.0",
                }
            }
        },
    }

    metadata = normalize_and_validate_fab_format(config)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min is None
    assert metadata.flwr_version_target is None


def test_normalize_and_validate_fab_format_derives_min_for_version_zero() -> None:
    """Test fab-format-version=0 derives only the lower bound when usable."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "dependencies": ["flwr>=1.26.0,<=1.28.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 0,
                    "flwr-version-target": "1.27.0",
                }
            }
        },
    }

    metadata = normalize_and_validate_fab_format(config)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min == "1.26.0"
    assert metadata.flwr_version_target is None


def test_v0_fab_format_skips_unsupported_bounds() -> None:
    """Test fab-format-version=0 ignores unrepresentable flwr dependency specifiers."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "dependencies": ["flwr>1.26.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 0,
                    "flwr-version-target": "1.27.0",
                }
            }
        },
    }

    metadata = normalize_and_validate_fab_format(config)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min is None
    assert metadata.flwr_version_target is None


def test_v0_fab_format_ignores_target_even_with_upper_bounds() -> None:
    """Test fab-format-version=0 ignores flwr-version-target even with upper bounds."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "dependencies": ["flwr>=1.26.0,<1.28.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 0,
                    "flwr-version-target": "2.1.0",
                }
            }
        },
    }

    metadata = normalize_and_validate_fab_format(config)

    assert metadata.fab_format_version == 0
    assert metadata.flwr_version_min == "1.26.0"
    assert metadata.flwr_version_target is None


def test_v1_fab_format_uses_highest_inclusive_lower_bound() -> None:
    """Test fab-format-version=1 derives the highest declared `>=` lower bound."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "license": {"file": "LICENSE"},
            "dependencies": ["flwr>=1.26.0,>=1.27.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "flwr-version-target": "1.27.0",
                }
            }
        },
    }

    metadata = normalize_and_validate_fab_format(config)

    assert metadata.fab_format_version == 1
    assert metadata.flwr_version_min == "1.27.0"
    assert metadata.flwr_version_target == "1.27.0"


def test_v1_fab_format_accepts_additional_non_lower_bound_specifiers() -> None:
    """Test fab-format-version=1 accepts extra specifiers beyond the lower bound."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "license": {"file": "LICENSE"},
            "dependencies": ["flwr>=1.26.0,==1.27.0,!=1.27.1"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "flwr-version-target": "1.27.0",
                }
            }
        },
    }

    metadata = normalize_and_validate_fab_format(config)

    assert metadata.fab_format_version == 1
    assert metadata.flwr_version_min == "1.26.0"
    assert metadata.flwr_version_target == "1.27.0"


def test_v1_fab_format_rejects_target_outside_declared_range() -> None:
    """Test fab-format-version=1 rejects targets outside the flwr specifier."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "license": {"file": "LICENSE"},
            "dependencies": ["flwr>=1.26.0,<2.0.0,<=1.28.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "flwr-version-target": "2.1.0",
                }
            }
        },
    }

    with pytest.raises(ValueError, match='must satisfy the declared "flwr"'):
        normalize_and_validate_fab_format(config)


def test_v1_fab_format_rejects_missing_inclusive_lower_bound() -> None:
    """Test fab-format-version=1 rejects lower bounds declared with `>` only."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "license": {"file": "LICENSE"},
            "dependencies": ["flwr>1.26.0,<2.0.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "flwr-version-target": "1.27.0",
                }
            }
        },
    }

    with pytest.raises(ValueError, match="inclusive lower bound"):
        normalize_and_validate_fab_format(config)


def test_v1_fab_format_requires_target_version() -> None:
    """Test fab-format-version=1 requires flwr-version-target."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "license": {"file": "LICENSE"},
            "dependencies": ["flwr>=1.26.0,<=1.28.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                }
            }
        },
    }

    with pytest.raises(ValueError, match="flwr-version-target"):
        normalize_and_validate_fab_format(config)


def test_v1_fab_format_requires_license_file_reference() -> None:
    """Test fab-format-version=1 requires [project].license.file."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "dependencies": ["flwr>=1.26.0,<=2.0.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "flwr-version-target": "1.27.0",
                }
            }
        },
    }

    with pytest.raises(ValueError, match=r"\[project\]\.license"):
        normalize_and_validate_fab_format(config)


def test_v1_fab_format_rejects_inline_license_text() -> None:
    """Test fab-format-version=1 rejects inline license text."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "license": {"text": "Apache-2.0"},
            "dependencies": ["flwr>=1.26.0,<=2.0.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "flwr-version-target": "1.27.0",
                }
            }
        },
    }

    with pytest.raises(ValueError, match="root-level license file"):
        normalize_and_validate_fab_format(config)


def test_v1_fab_format_rejects_invalid_license_file_name() -> None:
    """Test fab-format-version=1 only allows LICENSE or LICENSE.md."""
    config: dict[str, Any] = {
        "project": {
            "name": "fedgpt",
            "version": "1.0.0",
            "license": {"file": "legal/LICENSE.txt"},
            "dependencies": ["flwr>=1.26.0,<=2.0.0"],
        },
        "tool": {
            "flwr": {
                "app": {
                    "publisher": "flwrlabs",
                    "fab-format-version": 1,
                    "flwr-version-target": "1.27.0",
                }
            }
        },
    }

    with pytest.raises(ValueError, match='"LICENSE" or "LICENSE.md"'):
        normalize_and_validate_fab_format(config)
