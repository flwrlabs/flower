# Copyright 2026 Inria (cyrille kenfack & davide frey). All Rights Reserved.
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
"""Unit tests for args, node_config and runtime_node."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DYNAMIC_MODE = MagicMock(name="TopologyMode.dynamic()")
_STATIC_MODE = MagicMock(name="TopologyMode.static()")

# Patch targets used across tests
_PATCH_DYNAMIC = patch(
    "flwr.decentralized.common.node_config.topology_mode_dynamic",
    return_value=_DYNAMIC_MODE,
)
_PATCH_STATIC = patch(
    "flwr.decentralized.common.node_config.topology_mode_static",
    return_value=_STATIC_MODE,
)
_PATCH_GENERATE = patch(
    "flwr.decentralized.common.node_config.generate_deploy_topology_yaml",
    return_value=None,
)


# ---------------------------------------------------------------------------
# RuntimeNode
# ---------------------------------------------------------------------------


class TestRuntimeNode:
    """Tests for :class:`~flwr.decentralized.common.runtime_node.RuntimeNode`."""

    def _make(self, **kwargs: Any):
        from flwr.decentralized.common.runtime_node import RuntimeNode

        defaults = {
            "context": "test",
            "address": "0.0.0.0",
            "port": 9100,
            "topology_mode": _DYNAMIC_MODE,
        }
        defaults.update(kwargs)
        return RuntimeNode(**defaults)

    def test_defaults(self):
        node = self._make()
        assert node.tcp is True
        assert node.udp is False
        assert node.sampling_conf is None
        assert node.network_settings is None
        assert node.bootnodes is None

    def test_to_dnode_kwargs_keys(self):
        node = self._make()
        kwargs = node.to_dnode_kwargs()
        expected_keys = {
            "context",
            "address",
            "port",
            "topology_mode",
            "sampling_conf",
            "tcp",
            "udp",
            "network_settings",
            "bootnodes",
        }
        assert set(kwargs.keys()) == expected_keys

    def test_to_dnode_kwargs_values(self):
        node = self._make(context="cls", port=8000, tcp=False, udp=True)
        kw = node.to_dnode_kwargs()
        assert kw["context"] == "cls"
        assert kw["port"] == 8000
        assert kw["tcp"] is False
        assert kw["udp"] is True


# ---------------------------------------------------------------------------
# _parse_args_nodes  (argument parser structure)
# ---------------------------------------------------------------------------


class TestParseArgsNodes:
    """Tests for :func:`~flwr.decentralized.common.args._parse_args_nodes`."""

    def _parser(self):
        from flwr.decentralized.common.args import _parse_args_nodes
        import argparse

        parser = argparse.ArgumentParser()
        _parse_args_nodes(parser)

        return parser

    def test_returns_argument_parser(self):
        import argparse

        assert isinstance(self._parser(), argparse.ArgumentParser)

    def test_all_expected_args_present(self):
        parser = self._parser()
        option_strings = {
            a.option_strings[0] for a in parser._actions if a.option_strings
        }
        for flag in (
            "--config",
            "--context",
            "--address",
            "--port",
            "--tcp",
            "--udp",
            "--bootnodes",
            "--topology-mode",
            "--topology-file",
            "--node-name",
        ):
            assert flag in option_strings, f"Missing flag: {flag}"

    def test_topology_mode_choices(self):
        parser = self._parser()
        action = next(
            a for a in parser._actions if "--topology-mode" in a.option_strings
        )
        assert set(action.choices) == {"dynamic", "static"}

    def test_topology_mode_default_dynamic(self):
        from flwr.decentralized.common.args import _parse_args_nodes
        import argparse

        parser = argparse.ArgumentParser()

        _parse_args_nodes(parser)
        args = parser.parse_args([])  # no topology-mode flag
        assert args.topology_mode == "dynamic"

    def test_parse_minimal_args(self):
        from flwr.decentralized.common.args import _parse_args_nodes
        import argparse

        parser = argparse.ArgumentParser()
        _parse_args_nodes(parser)
        args = parser.parse_args(["--context", "cls", "--port", "9100"])
        assert args.context == "cls"
        assert args.port == 9100


# ---------------------------------------------------------------------------
# validate_topology_args
# ---------------------------------------------------------------------------


class TestValidateTopologyArgs:
    """Tests for :func:`~flwr.decentralized.common.args.validate_topology_args`."""

    def _ns(self, **kwargs):
        import argparse

        base = {"topology_mode": "dynamic", "topology_file": None}
        base.update(kwargs)
        return argparse.Namespace(**base)

    def _parser(self):
        import argparse

        return argparse.ArgumentParser()

    def test_dynamic_without_file_ok(self):
        from flwr.decentralized.common.args import validate_topology_args

        validate_topology_args(self._ns(), self._parser())  # no exception

    def test_static_without_file_raises(self):
        from flwr.decentralized.common.args import validate_topology_args

        with pytest.raises(SystemExit):
            validate_topology_args(
                self._ns(topology_mode="static", topology_file=None),
                self._parser(),
            )

    def test_static_with_file_ok(self, tmp_path):
        from flwr.decentralized.common.args import validate_topology_args

        f = tmp_path / "topo.yaml"
        f.touch()
        validate_topology_args(
            self._ns(topology_mode="static", topology_file=f),
            self._parser(),
        )  # no exception


# ---------------------------------------------------------------------------
# validate_node_args
# ---------------------------------------------------------------------------


class TestValidateNodeArgs:
    """Tests for :func:`~flwr.decentralized.common.args.validate_node_args`."""

    def _ns(self, **kwargs):
        import argparse

        base = {
            "config": None,
            "context": "cls",
            "topology_mode": "dynamic",
            "topology_file": None,
            "node_name": None,
        }
        base.update(kwargs)
        return argparse.Namespace(**base)

    def _parser(self):
        import argparse

        return argparse.ArgumentParser()

    def test_valid_dynamic(self):
        from flwr.decentralized.common.args import validate_node_args

        validate_node_args(self._ns(), self._parser())  # no exception

    def test_missing_context_without_config_raises(self):
        from flwr.decentralized.common.args import validate_node_args

        with pytest.raises(SystemExit):
            validate_node_args(self._ns(context=None), self._parser())

    def test_static_missing_node_name_raises(self):
        from flwr.decentralized.common.args import validate_node_args

        with pytest.raises(SystemExit):
            validate_node_args(
                self._ns(
                    topology_mode="static",
                    topology_file=Path("/some/topo.yaml"),
                    node_name=None,
                ),
                self._parser(),
            )

    def test_static_with_node_name_ok(self, tmp_path):
        from flwr.decentralized.common.args import validate_node_args

        f = tmp_path / "topo.yaml"
        f.touch()
        validate_node_args(
            self._ns(
                topology_mode="static",
                topology_file=f,
                node_name="node_0",
            ),
            self._parser(),
        )  # no exception

    def test_context_not_required_when_config_provided(self, tmp_path):
        from flwr.decentralized.common.args import validate_node_args

        cfg = tmp_path / "node.yaml"
        cfg.touch()
        validate_node_args(
            self._ns(config=cfg, context=None),
            self._parser(),
        )  # no exception


# ---------------------------------------------------------------------------
# get_args_nodes (CLI integration)
# ---------------------------------------------------------------------------


class TestGetArgsNodes:
    """Integration tests for :func:`~flwr.decentralized.common.args.get_args_nodes`."""

    @_PATCH_DYNAMIC
    def test_minimal_dynamic_cli(self, _mock_dyn):
        from flwr.decentralized.common.args import get_args_nodes

        node = get_args_nodes(["--context", "cls", "--port", "9100"])
        assert node.context == "cls"
        assert node.port == 9100
        assert node.topology_mode is _DYNAMIC_MODE

    @_PATCH_DYNAMIC
    def test_address_and_transport_cli(self, _mock_dyn):
        from flwr.decentralized.common.args import get_args_nodes

        node = get_args_nodes(
            [
                "--context",
                "fl",
                "--address",
                "192.168.1.1",
                "--port",
                "8888",
                "--tcp",
                "--no-udp",
            ]
        )
        assert node.address == "192.168.1.1"
        assert node.tcp is True
        assert node.udp is False

    @_PATCH_DYNAMIC
    def test_bootnodes_parsed_as_list(self, _mock_dyn):
        from flwr.decentralized.common.args import get_args_nodes

        node = get_args_nodes(
            [
                "--context",
                "cls",
                "--bootnodes",
                "127.0.0.1:9001",
                "127.0.0.1:9002",
            ]
        )
        assert node.bootnodes == ["127.0.0.1:9001", "127.0.0.1:9002"]

    def test_missing_context_exits(self):
        from flwr.decentralized.common.args import get_args_nodes

        with pytest.raises(SystemExit):
            get_args_nodes(["--port", "9100"])

    def test_static_missing_topology_file_exits(self):
        from flwr.decentralized.common.args import get_args_nodes

        with pytest.raises(SystemExit):
            get_args_nodes(["--context", "cls", "--topology-mode", "static"])

    def test_static_missing_node_name_exits(self, tmp_path):
        from flwr.decentralized.common.args import get_args_nodes

        f = tmp_path / "topo.yaml"
        f.touch()
        with pytest.raises(SystemExit):
            get_args_nodes(
                [
                    "--context",
                    "cls",
                    "--topology-mode",
                    "static",
                    "--topology-file",
                    str(f),
                    # --node-name missing
                ]
            )

    @_PATCH_STATIC
    def test_static_with_all_required_args(self, _mock_static, tmp_path):
        from flwr.decentralized.common.args import get_args_nodes

        f = tmp_path / "topo.yaml"
        f.write_text("---")
        node = get_args_nodes(
            [
                "--context",
                "cls",
                "--topology-mode",
                "static",
                "--topology-file",
                str(f),
                "--node-name",
                "node_0",
            ]
        )
        assert node.context == "cls"
        assert node.topology_mode is _STATIC_MODE

    def test_unsupported_config_extension_exits(self, tmp_path):
        from flwr.decentralized.common.args import get_args_nodes

        cfg = tmp_path / "node.json"
        cfg.write_text("{}")
        with pytest.raises(SystemExit):
            get_args_nodes(["--config", str(cfg), "--context", "cls"])


# ---------------------------------------------------------------------------
# load_node_config_yaml
# ---------------------------------------------------------------------------


class TestLoadNodeConfigYaml:
    """Tests for :func:`~flwr.decentralized.common.node_config.load_node_config_yaml`."""

    def _write(self, tmp_path, content: str) -> Path:
        p = tmp_path / "node.yaml"
        p.write_text(textwrap.dedent(content))
        return p

    @_PATCH_DYNAMIC
    def test_minimal_yaml(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_yaml

        p = self._write(
            tmp_path,
            """\
            context: classification
            address: 0.0.0.0
            port: 9100
        """,
        )
        node = load_node_config_yaml(p)
        assert node.context == "classification"
        assert node.address == "0.0.0.0"
        assert node.port == 9100
        assert node.topology_mode is _DYNAMIC_MODE

    @_PATCH_DYNAMIC
    def test_cli_override_port(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_yaml

        p = self._write(tmp_path, "context: cls\nport: 9100\n")
        node = load_node_config_yaml(p, overrides={"port": 9999})
        assert node.port == 9999

    @_PATCH_DYNAMIC
    def test_transport_flags(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_yaml

        p = self._write(tmp_path, "context: cls\ntcp: false\nudp: true\n")
        node = load_node_config_yaml(p)
        assert node.tcp is False
        assert node.udp is True

    @_PATCH_DYNAMIC
    def test_bootnodes(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_yaml

        p = self._write(
            tmp_path,
            textwrap.dedent("""\
            context: cls
            bootnodes:
              - "127.0.0.1:9001"
              - "127.0.0.1:9002"
        """),
        )
        node = load_node_config_yaml(p)
        assert node.bootnodes == ["127.0.0.1:9001", "127.0.0.1:9002"]

    @_PATCH_DYNAMIC
    def test_network_section_builds_network_settings(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_yaml
        from flwr.decentralized.common.network import NetworkSettings

        p = self._write(
            tmp_path,
            textwrap.dedent("""\
            context: cls
            network:
              idle_connection_timeout_secs: 30
              enable_mdns: false
        """),
        )
        node = load_node_config_yaml(p)
        assert isinstance(node.network_settings, NetworkSettings)
        assert node.network_settings.idle_connection_timeout_secs == 30
        assert node.network_settings.enable_mdns is False

    @_PATCH_DYNAMIC
    def test_sampling_gbps(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_yaml
        from flwr.decentralized.common.sampling import Configuration

        p = self._write(
            tmp_path,
            textwrap.dedent("""\
            context: cls
            sampling:
              algorithm: gbps
              config_file: /tmp/test_sampling.json
              params:
                view_size: 10
                heal: 2
                swap: 3
                selection_policy: rand
                propagation_policy: pushpull
                delay: 5
                age: 1
        """),
        )
        node = load_node_config_yaml(p)
        assert isinstance(node.sampling_conf, Configuration)

    @_PATCH_DYNAMIC
    def test_sampling_brahams(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_yaml
        from flwr.decentralized.common.sampling import Configuration

        p = self._write(
            tmp_path,
            textwrap.dedent("""\
            context: cls
            sampling:
              algorithm: brahams
              params:
                view_size: 8
                sampler_size: 5
                alpha: 0.45
                beta: 0.45
                delay: 3
        """),
        )
        node = load_node_config_yaml(p)
        assert isinstance(node.sampling_conf, Configuration)

    @_PATCH_DYNAMIC
    def test_sampling_basalt(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_yaml
        from flwr.decentralized.common.sampling import Configuration

        p = self._write(
            tmp_path,
            textwrap.dedent("""\
            context: cls
            sampling:
              algorithm: basalt
              params:
                view_size: 6
                refresh: 3
                delay: 4
        """),
        )
        node = load_node_config_yaml(p)
        assert isinstance(node.sampling_conf, Configuration)

    @_PATCH_DYNAMIC
    def test_unknown_sampling_algorithm_raises(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_yaml

        p = self._write(
            tmp_path,
            textwrap.dedent("""\
            context: cls
            sampling:
              algorithm: unknown_algo
              params:
                view_size: 5
                delay: 1
        """),
        )
        with pytest.raises(ValueError, match="Unknown sampling algorithm"):
            load_node_config_yaml(p)

    @_PATCH_STATIC
    def test_static_topology_with_file(self, _mock_static, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_yaml

        topo = tmp_path / "topo.yaml"
        topo.write_text("---")
        p = self._write(
            tmp_path,
            textwrap.dedent(f"""\
            context: cls
            topology:
              mode: static
              node_name: node_0
              file: {topo}
        """),
        )
        node = load_node_config_yaml(p)
        assert node.topology_mode is _STATIC_MODE

    @_PATCH_STATIC
    @_PATCH_GENERATE
    def test_static_topology_auto_generate(self, _mock_gen, _mock_static, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_yaml

        out = tmp_path / "gen_topo.yaml"
        p = self._write(
            tmp_path,
            textwrap.dedent(f"""\
            context: cls
            topology:
              mode: static
              node_name: node_0
              generate:
                node_count: 4
                kind: ring
                output_path: {out}
        """),
        )
        node = load_node_config_yaml(p)
        assert node.topology_mode is _STATIC_MODE
        _mock_gen.assert_called_once()

    def test_file_not_found_raises(self):
        from flwr.decentralized.common.node_config import load_node_config_yaml

        with pytest.raises(FileNotFoundError):
            load_node_config_yaml("/nonexistent/path/node.yaml")


# ---------------------------------------------------------------------------
# load_node_config_toml
# ---------------------------------------------------------------------------


class TestLoadNodeConfigToml:
    """Tests for :func:`~flwr.decentralized.common.node_config.load_node_config_toml`."""

    @_PATCH_DYNAMIC
    def test_minimal_toml(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_toml

        p = tmp_path / "node.toml"
        p.write_text(textwrap.dedent("""\
            context = "classification"
            address = "0.0.0.0"
            port = 9100
        """))
        node = load_node_config_toml(p)
        assert node.context == "classification"
        assert node.port == 9100

    @_PATCH_DYNAMIC
    def test_cli_override_context(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_toml

        p = tmp_path / "node.toml"
        p.write_text('context = "original"\nport = 9100\n')
        node = load_node_config_toml(p, overrides={"context": "overridden"})
        assert node.context == "overridden"

    @_PATCH_DYNAMIC
    def test_network_section_toml(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.node_config import load_node_config_toml
        from flwr.decentralized.common.network import NetworkSettings

        p = tmp_path / "node.toml"
        p.write_text(textwrap.dedent("""\
            context = "cls"
            [network]
            idle_connection_timeout_secs = 120
            enable_kad = false
        """))
        node = load_node_config_toml(p)
        assert isinstance(node.network_settings, NetworkSettings)
        assert node.network_settings.idle_connection_timeout_secs == 120
        assert node.network_settings.enable_kad is False

    def test_file_not_found_raises(self):
        from flwr.decentralized.common.node_config import load_node_config_toml

        with pytest.raises(FileNotFoundError):
            load_node_config_toml("/nonexistent/path/node.toml")


# ---------------------------------------------------------------------------
# get_args_nodes with --config file
# ---------------------------------------------------------------------------


class TestGetArgsNodesWithConfig:
    """Integration tests combining --config + CLI overrides."""

    @_PATCH_DYNAMIC
    def test_yaml_config_loaded(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.args import get_args_nodes

        cfg = tmp_path / "node.yaml"
        cfg.write_text("context: loaded\nport: 7000\n")
        node = get_args_nodes(["--config", str(cfg)])
        assert node.context == "loaded"
        assert node.port == 7000

    @_PATCH_DYNAMIC
    def test_cli_port_overrides_yaml(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.args import get_args_nodes

        cfg = tmp_path / "node.yaml"
        cfg.write_text("context: cls\nport: 7000\n")
        node = get_args_nodes(["--config", str(cfg), "--port", "9999"])
        assert node.port == 9999

    @_PATCH_DYNAMIC
    def test_toml_config_loaded(self, _mock_dyn, tmp_path):
        from flwr.decentralized.common.args import get_args_nodes

        cfg = tmp_path / "node.toml"
        cfg.write_text('context = "toml_ctx"\nport = 8100\n')
        node = get_args_nodes(["--config", str(cfg)])
        assert node.context == "toml_ctx"
        assert node.port == 8100
