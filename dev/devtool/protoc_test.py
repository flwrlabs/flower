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
"""Tests for `devtool.protoc`."""


from __future__ import annotations

from pathlib import Path

import grpc_tools
import pytest
from grpc_tools import protoc as grpc_protoc

from devtool import protoc


def _write_pyproject(project_dir: Path) -> None:
    (project_dir / "pyproject.toml").write_text(
        """
[tool.devtool.protoc]
proto_root = "proto"
include_globs = ["flwr/**/*.proto", "google/**/*.proto"]
include_paths = ["proto", "third_party/proto"]

[tool.devtool.protoc.outputs]
python = "py"
grpc_python = "py"
mypy = "py"
mypy_grpc = "py"
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_load_protoc_config_requires_tool_section(tmp_path: Path) -> None:
    """Missing `[tool.devtool.protoc]` should fail fast."""
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'test'\n",
        encoding="utf-8",
    )

    with pytest.raises(protoc.ProtocConfigError, match="tool.devtool.protoc"):
        protoc.load_protoc_config(tmp_path)


def test_load_protoc_config_resolves_paths_relative_to_project_dir(
    tmp_path: Path,
) -> None:
    """Configured paths should resolve relative to the project directory."""
    project_dir = tmp_path / "project"
    (project_dir / "proto").mkdir(parents=True)
    (project_dir / "third_party" / "proto").mkdir(parents=True)
    (project_dir / "py").mkdir()
    _write_pyproject(project_dir)

    config = protoc.load_protoc_config(project_dir)

    assert isinstance(config, protoc.ProtocConfig)
    assert config.proto_root == (project_dir / "proto").resolve()
    assert config.include_paths == (
        (project_dir / "proto").resolve(),
        (project_dir / "third_party" / "proto").resolve(),
    )
    assert config.outputs.python == (project_dir / "py").resolve()


def test_discover_proto_files_matches_globs(tmp_path: Path) -> None:
    """Proto discovery should union and sort matches from all configured globs."""
    proto_root = tmp_path / "proto"
    (proto_root / "flwr" / "nested").mkdir(parents=True)
    (proto_root / "google").mkdir()
    fleet_proto = proto_root / "flwr" / "fleet.proto"
    nested_proto = proto_root / "flwr" / "nested" / "node.proto"
    google_proto = proto_root / "google" / "timestamp.proto"
    fleet_proto.write_text("", encoding="utf-8")
    nested_proto.write_text("", encoding="utf-8")
    google_proto.write_text("", encoding="utf-8")

    discovered = protoc.discover_proto_files(
        proto_root,
        ("flwr/**/*.proto", "google/**/*.proto"),
    )

    assert discovered == [
        fleet_proto.resolve(),
        nested_proto.resolve(),
        google_proto.resolve(),
    ]


def test_build_protoc_command_includes_expected_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Command construction should include include paths, outputs, and files."""
    proto_root = tmp_path / "proto"
    include_path = tmp_path / "third_party" / "proto"
    output_root = tmp_path / "py"
    proto_root.mkdir()
    include_path.mkdir(parents=True)
    output_root.mkdir()
    proto_file = proto_root / "service.proto"
    proto_file.write_text("", encoding="utf-8")
    config = protoc.ProtocConfig(
        project_dir=tmp_path,
        proto_root=proto_root,
        include_globs=("*.proto",),
        include_paths=(include_path,),
        outputs=protoc.ProtocOutputs(
            python=output_root,
            grpc_python=output_root,
            mypy=output_root,
            mypy_grpc=output_root,
        ),
    )

    monkeypatch.setattr(grpc_tools, "__path__", [str(tmp_path / "grpc")])

    command = protoc.build_protoc_command(config, [proto_file])

    assert command[0] == "grpc_tools.protoc"
    assert any(flag.startswith("--proto_path=") for flag in command[1:4])
    assert f"--proto_path={tmp_path / 'grpc' / '_proto'}" in command
    assert f"--proto_path={proto_root}" in command
    assert f"--proto_path={include_path}" in command
    assert f"--python_out={output_root}" in command
    assert f"--grpc_python_out={output_root}" in command
    assert f"--mypy_out={output_root}" in command
    assert f"--mypy_grpc_out={output_root}" in command
    assert command[-1] == str(proto_file)


def test_compile_project_uses_loaded_protoc(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Project compilation should invoke the bundled grpc compiler."""
    project_dir = tmp_path / "project"
    (project_dir / "proto" / "flwr").mkdir(parents=True)
    (project_dir / "third_party" / "proto").mkdir(parents=True)
    (project_dir / "py").mkdir()
    _write_pyproject(project_dir)
    proto_file = project_dir / "proto" / "flwr" / "node.proto"
    proto_file.write_text("", encoding="utf-8")
    calls: list[list[str]] = []

    def dummy_main(command: list[str]) -> int:
        """Return success while recording the invocation."""
        calls.append(command)
        return 0

    monkeypatch.setattr(grpc_tools, "__path__", [str(tmp_path / "grpc")])
    monkeypatch.setattr(grpc_protoc, "main", dummy_main)

    protoc.compile_project(project_dir)

    assert len(calls) == 1
    assert calls[0][0] == "grpc_tools.protoc"
    assert f"--proto_path={tmp_path / 'grpc' / '_proto'}" in calls[0]
    assert str(proto_file.resolve()) in calls[0]
