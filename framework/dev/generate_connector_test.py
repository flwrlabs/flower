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
"""Tests for connector provider scaffolding."""

from pathlib import Path

import pytest

import dev.generate_connector as generate_connector


def test_scaffold_provider_and_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scaffolding should create three provider modules and register them."""
    providers = tmp_path / "providers"
    providers.mkdir()
    registry = providers / "registry_generated.py"
    registry.write_text(generate_connector.render_registry([]), encoding="utf-8")
    monkeypatch.setattr(generate_connector, "_FRAMEWORK_DIR", tmp_path)
    monkeypatch.setattr(generate_connector, "_PROVIDERS_DIR", providers)
    monkeypatch.setattr(generate_connector, "_REGISTRY_PATH", registry)

    generate_connector.scaffold_provider("example", "Example")
    generate_connector.update_registry()

    assert {path.name for path in (providers / "example").iterdir()} == {
        "__init__.py",
        "actions.py",
        "definition.py",
        "executors.py",
    }
    assert "access=ActionAccess.READ" in (providers / "example/actions.py").read_text(
        encoding="utf-8"
    )
    assert "providers.example" in registry.read_text(encoding="utf-8")
    generate_connector.update_registry(check=True)


def test_scaffold_rejects_invalid_provider() -> None:
    """Scaffolding should reject invalid provider package names."""
    with pytest.raises(ValueError, match="lowercase snake-case"):
        generate_connector._validate_identifier("Notion")
