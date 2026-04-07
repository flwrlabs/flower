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
"""Tests for config migration."""


import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import click
import tomli
import typer
from parameterized import parameterized

from .config_migration import (
    CONFIG_MIGRATION_NOTICE,
    _comment_out_legacy_toml_config,
    _is_legacy_usage,
    _migrate_pyproject_toml_to_flower_config,
    migrate,
)
from .constant import LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE
from .flower_config import init_flwr_config, read_superlink_connection

TEST_PYPROJECT_TOML = """[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "demo"
version = "1.0.0"
dependencies = ["flwr[simulation]>=1.19.0"]

[tool.hatch.build.targets.wheel]
packages = ["."]

[tool.flwr.app]
publisher = "pan"

[tool.flwr.app.components]
serverapp = "demo.server_app:app"
clientapp = "demo.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 30

[tool.flwr.federations]
default = "local-poc"

[tool.flwr.federations.local-poc]
address = "127.0.0.1:9093"
insecure = true

[tool.flwr.federations.my-sim]
options.num-supernodes = 2

[tool.flwr.federations.researchgrid]
address = "researchgrid.flower.blue"
root-certificates = "certs/researchgrid.crt"
options.num-supernodes = 2
"""


class TestConfigMigration(unittest.TestCase):
    """Tests for config migration helpers."""

    def setUp(self) -> None:
        """Set up temporary Flower home directory for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=R1732

        # Patch get_flwr_home to point to a temporary Flower home directory
        self.home_path = Path(self.temp_dir.name)
        self.home_path.mkdir(parents=True, exist_ok=True)
        self.get_home_patcher = patch(
            "flwr.cli.flower_config.get_flwr_home", return_value=self.home_path
        )
        self.get_home_patcher.start()
        init_flwr_config()

        # Create a temporary app directory with pyproject.toml
        self.app_path = self.home_path / "test-app"
        self.app_path.mkdir(parents=True, exist_ok=True)
        (self.app_path / "pyproject.toml").write_text(TEST_PYPROJECT_TOML)

    def tearDown(self) -> None:
        """Tear down temporary Flower home directory."""
        self.get_home_patcher.stop()
        self.temp_dir.cleanup()

    @parameterized.expand(  # type: ignore[misc]
        [
            ("local", ["my-federation"], True),  # single extra arg
            ("/abs/path", [], True),  # absolute path
            ("./relative/path", [], True),  # relative path with slash
            (".", [], True),  # dot path
            ("named-conn", [], False),  # normal name
            (None, [], False),  # None superlink; only possible with new usage
        ]
    )
    def test_is_legacy_usage(
        self, superlink: str | None, args: list[str], expected: bool
    ) -> None:
        """Test `_is_legacy_usage` function."""
        assert _is_legacy_usage(superlink, args) is expected

    def test_migrate_pyproject_toml_to_flower_config(self) -> None:
        """Test `_migrate_pyproject_toml_to_flower_config` function."""
        # Execute
        _migrate_pyproject_toml_to_flower_config(self.app_path, None)
        default_conn = read_superlink_connection()

        # Assert default is set from legacy config (local-poc)
        assert default_conn is not None
        assert default_conn.name == "local-poc"

        # Assert migrated connections are readable
        local_poc = read_superlink_connection("local-poc")
        assert local_poc is not None
        assert local_poc.address == "127.0.0.1:9093"
        assert local_poc.insecure is True

        my_sim = read_superlink_connection("my-sim")
        assert my_sim is not None
        assert my_sim.address == LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE
        assert my_sim.options is not None
        assert my_sim.options.num_supernodes == 2

        researchgrid = read_superlink_connection("researchgrid")
        assert researchgrid is not None
        assert researchgrid.address == "researchgrid.flower.blue"
        assert researchgrid.root_certificates == str(
            (self.app_path / "certs/researchgrid.crt").resolve()
        )

    def test_comment_out_legacy_toml_config(self) -> None:
        """Test `_comment_out_legacy_toml_config` function."""
        # Execute
        _comment_out_legacy_toml_config(self.app_path)

        pyproject_content = (self.app_path / "pyproject.toml").read_text()

        # Notice is added
        assert CONFIG_MIGRATION_NOTICE in pyproject_content

        # After commenting, tool.flwr.federations should be absent when parsed
        parsed = tomli.loads(pyproject_content)
        assert "tool" in parsed
        assert "flwr" in parsed["tool"]
        assert "federations" not in parsed["tool"]["flwr"]

        # The app section should remain intact
        assert parsed["tool"]["flwr"]["app"]["publisher"] == "pan"

    @patch("flwr.cli.config_migration.click.get_current_context")
    def test_migrate_success_with_legacy_usage(
        self, mock_get_context: MagicMock
    ) -> None:
        """Test successful migration with legacy usage pattern."""
        # Prepare: Mock context for usage output
        mock_ctx = MagicMock()
        mock_ctx.get_usage.return_value = "Usage: flwr [OPTIONS]"
        mock_get_context.return_value = mock_ctx

        # Execute and expect typer.Exit due to legacy usage
        with self.assertRaises(typer.Exit):
            migrate(str(self.app_path), ["local-poc"])

        # Assert: Verify connection was migrated
        conn = read_superlink_connection("local-poc")
        assert conn is not None
        assert conn.address == "127.0.0.1:9093"

    def test_migrate_failure_legacy_usage_no_federations(self) -> None:
        """Test migration failure when legacy usage but no federations."""
        # Create app without federations section
        app_path = self.home_path / "no-fed-app"
        app_path.mkdir(parents=True, exist_ok=True)
        (app_path / "pyproject.toml").write_text("[project]\nname='test'")

        with self.assertRaises(click.ClickException) as cm:
            migrate(str(app_path), [])
        assert "Cannot migrate configuration" in str(cm.exception)

    def test_migrate_no_action_non_legacy_usage(self) -> None:
        """Test no migration when non-legacy usage detected."""
        migrate("named-conn", [])

        # Should not create this connection
        with self.assertRaises(click.ClickException):
            _ = read_superlink_connection("named-conn")

    @parameterized.expand(  # type: ignore[misc]
        [
            (None, False),
            ("named-conn", False),
            ("/abs/path", True),  # `flwr run` usage
        ]
    )
    @patch("flwr.cli.config_migration.typer.echo")
    @patch("flwr.cli.config_migration.typer.secho")
    def test_migrate_silent_when_not_migratable_and_non_legacy(
        self,
        superlink: str | None,
        ignore_legacy_usage: bool,
        mock_secho: MagicMock,
        mock_echo: MagicMock,
    ) -> None:
        """Test no output when not migratable and not legacy usage."""
        # Execute with a non-path superlink that doesn't exist
        migrate(superlink, [], ignore_legacy_usage)

        # Verify no output was printed
        mock_echo.assert_not_called()
        mock_secho.assert_not_called()

    def test_migrate_raises_on_multiple_args(self) -> None:
        """Test that multiple extra args raise UsageError."""
        with self.assertRaises(click.UsageError):
            migrate(str(self.app_path), ["arg1", "arg2"])

    @patch("flwr.cli.config_migration.click.get_current_context")
    def test_migrate_with_default_federation(self, mock_get_context: MagicMock) -> None:
        """Test migration properly sets default federation."""
        # Mock context for usage output
        mock_ctx = MagicMock()
        mock_ctx.get_usage.return_value = "Usage: flwr [OPTIONS]"
        mock_get_context.return_value = mock_ctx

        # Execute migration without specifying federation (legacy path)
        with self.assertRaises(typer.Exit):
            migrate(str(self.app_path), [])

        # Verify default connection is set from legacy config
        default_conn = read_superlink_connection()
        assert default_conn is not None
        assert default_conn.name == "local-poc"
