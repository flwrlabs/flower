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
"""Test for Flower command line interface configuration utils."""


import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import click
import tomli
from parameterized import parameterized

from flwr.cli.constant import (
    DEFAULT_FLOWER_CONFIG_TOML,
    FLOWER_CONFIG_FILE,
    LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE,
    SuperLinkConnectionTomlKey,
)
from flwr.cli.typing import (
    SimulationBackendConfig,
    SimulationClientResources,
    SuperLinkConnection,
    SuperLinkSimulationOptions,
)
from flwr.common.constant import FLWR_HOME

from .flower_config import (
    init_flwr_config,
    parse_superlink_connection,
    read_superlink_connection,
    write_flower_config,
    write_superlink_connection,
)


class TestInitFlwrConfig(unittest.TestCase):
    """Test `init_flwr_config` function."""

    def test_init_flwr_config_creates_file(self) -> None:
        """Test that init_flwr_config creates the config file if it doesn't exist."""
        # Prepare
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Set FLWR_HOME to the temporary directory
            with patch.dict(os.environ, {FLWR_HOME: tmp_dir}):
                # Execute
                init_flwr_config()

                # Assert
                config_path = Path(tmp_dir) / "config.toml"
                self.assertTrue(config_path.exists())

                self.assertEqual(
                    config_path.read_text(encoding="utf-8"), DEFAULT_FLOWER_CONFIG_TOML
                )

    def test_default_config_matches_constants(self) -> None:
        """Verify that DEFAULT_FLOWER_CONFIG_TOML uses the correct keys."""
        # Parse the default config string
        config = tomli.loads(DEFAULT_FLOWER_CONFIG_TOML)

        # 1. Check top-level [superlink]
        self.assertIn(SuperLinkConnectionTomlKey.SUPERLINK, config)
        superlink = config[SuperLinkConnectionTomlKey.SUPERLINK]

        # 2. Check default = "local"
        self.assertEqual(superlink[SuperLinkConnectionTomlKey.DEFAULT], "local")

        # 3. Check [superlink.supergrid]
        self.assertIn("supergrid", superlink)
        supergrid = superlink["supergrid"]
        self.assertEqual(
            supergrid[SuperLinkConnectionTomlKey.ADDRESS], "supergrid.flower.ai"
        )

        # 4. Check [superlink.local]
        self.assertIn("local", superlink)
        local = superlink["local"]
        self.assertEqual(
            local[SuperLinkConnectionTomlKey.ADDRESS],
            LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE,
        )

    def test_init_flwr_config_does_not_overwrite(self) -> None:
        """Test that init_flwr_config does not overwrite existing config file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup - create existing config
            config_path = Path(tmp_dir) / "config.toml"
            config_path.write_text("existing_content", encoding="utf-8")

            # Mock FLWR_HOME
            with patch.dict(os.environ, {FLWR_HOME: tmp_dir}):
                # Execute
                init_flwr_config()

                # Assert
                self.assertEqual(
                    config_path.read_text(encoding="utf-8"), "existing_content"
                )


class TestSuperLinkConnection(unittest.TestCase):
    """Unit tests for SuperLink connections."""

    def test_parse_superlink_connection_valid(self) -> None:
        """Test parse_superlink_connection with valid input."""
        # Prepare
        conn_dict = {
            SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:8080",
            SuperLinkConnectionTomlKey.ROOT_CERTIFICATES: "/path/to/root_cert.crt",
            SuperLinkConnectionTomlKey.INSECURE: False,
        }
        name = "test_service"

        # Execute
        config = parse_superlink_connection(conn_dict, name)

        # Assert
        self.assertEqual(config.name, name)
        self.assertEqual(config.address, "127.0.0.1:8080")
        self.assertEqual(config.root_certificates, "/path/to/root_cert.crt")
        self.assertFalse(config.insecure)

    def test_parse_superlink_connection_raises_on_relative_path(self) -> None:
        """Test parse_superlink_connection raises on relative path."""
        # Prepare
        conn_dict = {
            SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:8080",
            SuperLinkConnectionTomlKey.ROOT_CERTIFICATES: "certs/ca.crt",
        }
        name = "test_path_res"

        # Execute
        with self.assertRaises(ValueError):
            parse_superlink_connection(conn_dict, name)

    def test_parse_superlink_connection_invalid_type(self) -> None:
        """Test parse_superlink_connection with invalid type."""
        conn_dict = {
            SuperLinkConnectionTomlKey.ADDRESS: 123,  # Invalid type, should be str
        }
        name = "test_service"

        with self.assertRaises(ValueError):
            parse_superlink_connection(conn_dict, name)

    @parameterized.expand(  # type: ignore
        [
            (
                "supergrid",
                {
                    SuperLinkConnectionTomlKey.ADDRESS: "supergrid.flower.ai",
                },
                SuperLinkConnection(
                    name="supergrid",
                    address="supergrid.flower.ai",
                ),
            ),
            (
                "local",
                {
                    SuperLinkConnectionTomlKey.ADDRESS: (
                        LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE
                    ),
                },
                SuperLinkConnection(
                    name="local",
                    address=LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE,
                ),
            ),
            (
                "local-with-options",
                {
                    SuperLinkConnectionTomlKey.ADDRESS: (
                        LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE
                    ),
                    SuperLinkConnectionTomlKey.OPTIONS: {
                        "num-supernodes": 10,
                    },
                },
                SuperLinkConnection(
                    name="local-with-options",
                    address=LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE,
                    options=SuperLinkSimulationOptions(
                        num_supernodes=10,
                    ),
                ),
            ),
            (
                "local-poc",
                {
                    SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:9093",
                    SuperLinkConnectionTomlKey.INSECURE: True,
                },
                SuperLinkConnection(
                    name="local-poc",
                    address="127.0.0.1:9093",
                    insecure=True,
                ),
            ),
            (
                "local-poc-dev",
                {
                    SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:9093",
                    SuperLinkConnectionTomlKey.ROOT_CERTIFICATES: "/app/root_cert.crt",
                },
                SuperLinkConnection(
                    name="local-poc-dev",
                    address="127.0.0.1:9093",
                    root_certificates="/app/root_cert.crt",
                ),
            ),
            (
                "local-poc-dev-sys-cert",
                {
                    SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:9093",
                },
                SuperLinkConnection(
                    name="local-poc-dev-sys-cert",
                    address="127.0.0.1:9093",
                ),
            ),
            (
                "remote-sim",
                {
                    SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:9093",
                    SuperLinkConnectionTomlKey.ROOT_CERTIFICATES: "/app/root_cert.crt",
                    SuperLinkConnectionTomlKey.OPTIONS: {
                        "num-supernodes": 10,
                        "backend": {
                            "client-resources": {"num-cpus": 1},
                        },
                    },
                },
                SuperLinkConnection(
                    name="remote-sim",
                    address="127.0.0.1:9093",
                    root_certificates="/app/root_cert.crt",
                    options=SuperLinkSimulationOptions(
                        num_supernodes=10,
                        backend=SimulationBackendConfig(
                            client_resources=SimulationClientResources(num_cpus=1),
                        ),
                    ),
                ),
            ),
        ]
    )
    def test_parse_superlink_connection_valid_cases(
        self,
        name: str,
        conn_dict: dict[str, Any],
        expected_config: SuperLinkConnection,
    ) -> None:
        """Test various valid connection configurations from valid.toml."""
        config = parse_superlink_connection(conn_dict, name)
        self.assertEqual(config, expected_config)

    def test_parse_superlink_connection_simulation_full(self) -> None:
        """Test parse_superlink_connection with simulation options."""
        # Prepare
        conn_dict = {
            SuperLinkConnectionTomlKey.OPTIONS: {
                "num-supernodes": 10,
                "backend": {
                    "client-resources": {"num-cpus": 1.0, "num-gpus": 0.5},
                    "init-args": {"logging-level": "info", "log-to-driver": True},
                    "name": "custom-backend",
                },
            }
        }
        name = "local-simulation"

        # Execute
        config = parse_superlink_connection(conn_dict, name)

        # Assert
        self.assertEqual(config.name, name)
        self.assertIsNone(config.address)
        self.assertIsNotNone(config.options)
        assert config.options is not None
        self.assertEqual(config.options.num_supernodes, 10)

        # Check Backend
        backend = config.options.backend
        self.assertIsNotNone(backend)
        assert backend is not None
        self.assertIsNotNone(backend.client_resources)
        assert backend.client_resources is not None
        self.assertEqual(backend.client_resources.num_cpus, 1.0)
        self.assertEqual(backend.client_resources.num_gpus, 0.5)

        self.assertIsNotNone(backend.init_args)
        assert backend.init_args is not None
        self.assertEqual(backend.init_args.logging_level, "info")
        self.assertTrue(backend.init_args.log_to_driver)
        self.assertEqual(backend.name, "custom-backend")

    def test_parse_superlink_connection_simulation_invalid_name(self) -> None:
        """Test parse_superlink_connection with invalid name type."""
        # Prepare
        conn_dict = {
            "options": {
                "num-supernodes": 10,
                "backend": {
                    "name": 123,  # Invalid type, should be str
                },
            }
        }
        name = "local-simulation"

        # Execute & Assert
        with self.assertRaisesRegex(ValueError, "backend.name must be a string"):
            parse_superlink_connection(conn_dict, name)

    def test_parse_superlink_connection_simulation_missing_num_supernodes(self) -> None:
        """Test mixed connection missing required simulation field."""
        # Prepare
        conn_dict: dict[str, Any] = {
            SuperLinkConnectionTomlKey.ADDRESS: "127.0.0.1:8080",
            SuperLinkConnectionTomlKey.ROOT_CERTIFICATES: None,
            SuperLinkConnectionTomlKey.INSECURE: False,
            "options": {},  # Missing num-supernodes
        }
        name = "mixed-invalid"

        # Execute & Assert
        with self.assertRaisesRegex(
            ValueError, "Invalid simulation options: num-supernodes must be an integer"
        ):
            parse_superlink_connection(conn_dict, name)

    @patch("flwr.cli.flower_config.get_flwr_home")
    def test_read_superlink_connection_defaults(self, mock_get_flwr_home: Mock) -> None:
        """Test read_superlink_connection uses default when no arg provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Create TOML content
            toml_content = """
            [superlink]
            default = "mock-service-2"

            [superlink.mock-service]
            address = "losthost:1234"
            insecure = false

            [superlink.mock-service-2]
            address = "losthost:9093"
            insecure = true
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute
            config = read_superlink_connection()

            # Assert
            assert config is not None
            self.assertEqual(config.name, "mock-service-2")
            self.assertEqual(config.address, "losthost:9093")

    @patch("flwr.cli.flower_config.get_flwr_home")
    def test_read_superlink_connection_explicit(self, mock_get_flwr_home: Mock) -> None:
        """Test read_superlink_connection with explicit connection name."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Prepare
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Create TOML content
            toml_content = """
            [superlink]
            default = "mock-service-2"

            [superlink.mock-service]
            address = "losthost:1234"
            insecure = false

            [superlink.mock-service-2]
            address = "losthost:9093"
            insecure = true
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute
            config = read_superlink_connection("mock-service")

            # Assert
            assert config is not None
            self.assertEqual(config.name, "mock-service")
            self.assertEqual(config.address, "losthost:1234")

    @patch("flwr.cli.flower_config.get_flwr_home")
    def test_read_superlink_connection_explicit_missing(
        self, mock_get_flwr_home: Mock
    ) -> None:
        """Test read_superlink_connection with explicit but missing connection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Create TOML content
            toml_content = """
            [superlink]
            default = "mock-service"

            [superlink.mock-service]
            address = "losthost:9093"
            insecure = false
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute & Assert
            with self.assertRaises(click.ClickException):
                read_superlink_connection("missing-service")

    @patch("flwr.cli.flower_config.get_flwr_home")
    def test_read_superlink_connection_no_default_failure(
        self, mock_get_flwr_home: Mock
    ) -> None:
        """Test failure when no default is set and no arg provided."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Create TOML content without default
            toml_content = """
            [superlink]
            # No default

            [superlink.mock-service]
            address = "losthost:9093"
            insecure = false
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute & Assert
            with self.assertRaises(click.ClickException):
                read_superlink_connection()

    @patch("flwr.cli.flower_config.get_flwr_home")
    def test_read_superlink_connection_default_missing_connection(
        self, mock_get_flwr_home: Mock
    ) -> None:
        """Test failure when default is set but the connection block is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Default points to "missing-service", which is not defined
            toml_content = """
            [superlink]
            default = "missing-service"

            [superlink.other-service]
            address = "losthost:9093"
            insecure = false
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute & Assert
            with self.assertRaises(click.ClickException):
                read_superlink_connection()

    @patch("flwr.cli.flower_config.get_flwr_home")
    def test_read_superlink_connection_corrupted(
        self, mock_get_flwr_home: Mock
    ) -> None:
        """Test read_superlink_connection when file is corrupted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Write invalid TOML
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("invalid = toml [ [")

            with self.assertRaises(click.ClickException):
                read_superlink_connection()

    @patch("flwr.cli.flower_config.get_flwr_home")
    def test_read_superlink_connection_simulation(
        self, mock_get_flwr_home: Mock
    ) -> None:
        """Test read_superlink_connection with simulation profile."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_get_flwr_home.return_value = Path(temp_dir)
            config_path = Path(temp_dir) / FLOWER_CONFIG_FILE

            # Create TOML content
            toml_content = """
            [superlink]
            default = "local-sim"

            [superlink.local-sim]
            address = ":local:"
            options.num-supernodes = 2
            options.verbose = true
            options.backend.client-resources.num-cpus = 2.0
            options.backend.init-args.num-cpus = 1
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            # Execute
            config = read_superlink_connection()

            # Assert
            assert config is not None
            self.assertEqual(config.name, "local-sim")
            self.assertEqual(config.address, LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE)
            assert config.options is not None
            self.assertEqual(config.options.num_supernodes, 2)
            assert config.options.backend is not None
            self.assertEqual(config.options.verbose, True)
            assert config.options.backend.client_resources is not None
            self.assertEqual(config.options.backend.client_resources.num_cpus, 2.0)
            assert config.options.backend.init_args is not None
            self.assertEqual(config.options.backend.init_args.num_cpus, 1)
            self.assertEqual(config.options.backend.name, "ray")

    @patch("flwr.cli.flower_config.get_flwr_home")
    def test_read_superlink_connection_simulation_unknown_key(
        self, mock_get_flwr_home: Mock
    ) -> None:
        """Test read_superlink_connection with unknown keys (should be ignored)."""
        # Prepare
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_get_flwr_home.return_value = Path(tmp_dir)
            config_path = os.path.join(tmp_dir, "config.toml")
            toml_content = """
            [superlink]
            default = 'local-sim'

            [superlink.local-sim]
            address = ':local:'
            options.num-supernodes = 2
            options.backend.client-resources.num-cpus = 2.0
            options.backend.unknown-key = "unexpected"
            """

            with open(config_path, "w", encoding="utf-8") as f:
                f.write(toml_content)

            name = "local-sim"
            # Expected: a valid config that doesn't include keys
            # that don't align with the scheema
            expected_config = SuperLinkConnection(
                name="local-sim",
                address=LOCAL_SUPERLINK_ADDRESS_MAGIC_VALUE,
                options=SuperLinkSimulationOptions(
                    num_supernodes=2,
                    backend=SimulationBackendConfig(
                        client_resources=SimulationClientResources(num_cpus=2.0),
                    ),
                ),
            )

            # Execute
            config = read_superlink_connection(name)

            # Assert
            self.assertEqual(config, expected_config)

    @patch("flwr.cli.flower_config.get_flwr_home")
    def test_write_superlink_connection(self, mock_get_flwr_home: Mock) -> None:
        """Test write_superlink_connection updates the config file correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            mock_get_flwr_home.return_value = temp_path

            # Prepare
            init_flwr_config()
            config_path = temp_path / FLOWER_CONFIG_FILE
            init_content = config_path.read_text()
            added_content = """
[superlink.new-service]
address = "localhost:9999"
insecure = false
options.num-supernodes = 12
options.backend.name = "ray"
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 0.5
"""
            new_conn = SuperLinkConnection(
                name="new-service",
                address="localhost:9999",
                insecure=False,
                options=SuperLinkSimulationOptions(
                    num_supernodes=12,
                    backend=SimulationBackendConfig(
                        name="ray",
                        client_resources=SimulationClientResources(
                            num_cpus=2, num_gpus=0.5
                        ),
                    ),
                ),
            )

            # Execute
            write_superlink_connection(new_conn)
            read_conn = read_superlink_connection("new-service")
            updated_content = config_path.read_text()

            # Assert
            self.assertEqual(read_conn, new_conn)
            self.assertEqual(updated_content, init_content + added_content)


def test_write_flower_config() -> None:
    """Test that write_flower_config removes quotes from dotted keys."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Prepare
        # Test data with dotted keys
        toml_dict = {
            "superlink": {
                "default": "mock-service",
                "mock-service": {
                    "address": "localhost:9999",
                    "insecure": True,
                    "options.num-supernodes": 10,
                    "options.backend.client-resources.num-gpus": 0,
                },
            }
        }
        expected_content = """[superlink]
default = "mock-service"

[superlink.mock-service]
address = "localhost:9999"
insecure = true
options.num-supernodes = 10
options.backend.client-resources.num-gpus = 0
"""

        # Execute
        # Patch get_flwr_home to return our temporary directory
        with patch("flwr.cli.flower_config.get_flwr_home", return_value=temp_path):
            config_path = write_flower_config(toml_dict)
        content = config_path.read_text()

        # Assert
        assert config_path == temp_path / FLOWER_CONFIG_FILE
        assert content == expected_content
