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
"""Compile protobufs for a configured project."""


from __future__ import annotations

import argparse
import tomllib
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import grpc_tools
from google.protobuf.descriptor_pb2 import (  # pylint: disable=no-name-in-module
    DescriptorProto,
    FileDescriptorProto,
    FileDescriptorSet,
)
from grpc_tools import protoc


class ProtocConfigError(ValueError):
    """Raised when the protoc configuration is invalid."""


class ProtocExecutionError(RuntimeError):
    """Raised when `grpc_tools.protoc` fails."""


@dataclass(frozen=True)
class ProtocOutputs:
    """Output directories for generated artifacts."""

    python: Path
    grpc_python: Path
    mypy: Path
    mypy_grpc: Path


@dataclass(frozen=True)
class ProtocConfig:
    """Resolved protoc configuration."""

    project_dir: Path
    proto_root: Path
    include_globs: tuple[str, ...]
    include_paths: tuple[Path, ...]
    outputs: ProtocOutputs
    documented_messages: tuple[str, ...] = ()


def _load_pyproject(project_dir: Path) -> dict[str, Any]:
    pyproject_path = project_dir / "pyproject.toml"
    if not pyproject_path.is_file():
        raise ProtocConfigError(f"Missing pyproject.toml: {pyproject_path}")
    with pyproject_path.open("rb") as pyproject_file:
        return tomllib.load(pyproject_file)


def _resolve_dir(project_dir: Path, value: str, *, field_name: str) -> Path:
    resolved = (project_dir / value).resolve()
    if not resolved.is_dir():
        raise ProtocConfigError(
            f"Configured `{field_name}` directory does not exist: {resolved}"
        )
    return resolved


def _require_string(config: dict[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ProtocConfigError(f"Expected `{key}` to be a non-empty string")
    return value


def _require_string_list(config: dict[str, Any], key: str) -> tuple[str, ...]:
    value = config.get(key)
    if not isinstance(value, list) or not value:
        raise ProtocConfigError(f"Expected `{key}` to be a non-empty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ProtocConfigError(f"Expected `{key}` to contain only non-empty strings")
    return tuple(value)


def _load_outputs(project_dir: Path, config: dict[str, Any]) -> ProtocOutputs:
    outputs = config.get("outputs")
    if not isinstance(outputs, dict):
        raise ProtocConfigError("Missing `[tool.devtool.protoc.outputs]` table")

    return ProtocOutputs(
        python=_resolve_dir(
            project_dir,
            _require_string(outputs, "python"),
            field_name="outputs.python",
        ),
        grpc_python=_resolve_dir(
            project_dir,
            _require_string(outputs, "grpc_python"),
            field_name="outputs.grpc_python",
        ),
        mypy=_resolve_dir(
            project_dir,
            _require_string(outputs, "mypy"),
            field_name="outputs.mypy",
        ),
        mypy_grpc=_resolve_dir(
            project_dir,
            _require_string(outputs, "mypy_grpc"),
            field_name="outputs.mypy_grpc",
        ),
    )


def load_protoc_config(project_dir: Path) -> ProtocConfig:
    """Load and resolve the protoc config from `pyproject.toml`."""
    pyproject = _load_pyproject(project_dir.resolve())
    tool = pyproject.get("tool")
    if not isinstance(tool, dict):
        raise ProtocConfigError("Missing `[tool.devtool.protoc]` configuration")

    devtool_config = tool.get("devtool")
    if not isinstance(devtool_config, dict):
        raise ProtocConfigError("Missing `[tool.devtool.protoc]` configuration")

    protoc_config = devtool_config.get("protoc")
    if not isinstance(protoc_config, dict):
        raise ProtocConfigError("Missing `[tool.devtool.protoc]` configuration")

    resolved_project_dir = project_dir.resolve()
    proto_root = _resolve_dir(
        resolved_project_dir,
        _require_string(protoc_config, "proto_root"),
        field_name="proto_root",
    )
    include_globs = _require_string_list(protoc_config, "include_globs")
    include_paths = tuple(
        _resolve_dir(
            resolved_project_dir,
            include_path,
            field_name="include_paths",
        )
        for include_path in _require_string_list(protoc_config, "include_paths")
    )

    return ProtocConfig(
        project_dir=resolved_project_dir,
        proto_root=proto_root,
        include_globs=include_globs,
        include_paths=include_paths,
        outputs=_load_outputs(resolved_project_dir, protoc_config),
        documented_messages=(
            _require_string_list(protoc_config, "documented_messages")
            if "documented_messages" in protoc_config
            else ()
        ),
    )


def discover_proto_files(
    proto_root: Path, include_globs: tuple[str, ...]
) -> list[Path]:
    """Return the sorted proto files matched by the configured globs."""
    proto_files = {
        proto_file.resolve()
        for include_glob in include_globs
        for proto_file in proto_root.glob(include_glob)
        if proto_file.is_file()
    }
    discovered = sorted(proto_files, key=lambda proto_file: proto_file.as_posix())
    if not discovered:
        raise ProtocConfigError(
            "No `.proto` files matched the configured include globs under "
            f"{proto_root}"
        )
    return discovered


def build_protoc_command(config: ProtocConfig, proto_files: list[Path]) -> list[str]:
    """Build the `grpc_tools.protoc` invocation for the resolved config."""
    bundled_proto_dir = Path(grpc_tools.__path__[0]) / "_proto"
    include_dirs: list[Path] = []
    for include_dir in (bundled_proto_dir, config.proto_root, *config.include_paths):
        if include_dir not in include_dirs:
            include_dirs.append(include_dir)

    command = ["grpc_tools.protoc"]
    command.extend(f"--proto_path={include_dir}" for include_dir in include_dirs)
    command.extend(
        [
            f"--python_out={config.outputs.python}",
            f"--grpc_python_out={config.outputs.grpc_python}",
            f"--mypy_out={config.outputs.mypy}",
            f"--mypy_grpc_out={config.outputs.mypy_grpc}",
        ]
    )
    command.extend(str(proto_file) for proto_file in proto_files)
    return command


def _render_python_docs(
    proto: FileDescriptorProto, names: dict[tuple[int, ...], str]
) -> str:
    lines = [
        "# Generated by devtool.protoc. DO NOT EDIT.",
        f'"""Protobuf descriptions from {proto.name}."""',
        "",
        "PROTOBUF_DESCRIPTIONS: dict[str, str] = {",
    ]
    for location in proto.source_code_info.location:
        name = names.get(tuple(location.path))
        if name is None:
            continue
        comments = f"{location.leading_comments} {location.trailing_comments}"
        if description := " ".join(comments.split()):
            lines.append(f"    {name!r}: {description!r},")
    lines.extend(("}", ""))
    return "\n".join(lines)


def _write_python_docs(descriptor_set: FileDescriptorSet, config: ProtocConfig) -> None:
    pending = set(config.documented_messages)
    generated_files: dict[Path, str] = {}
    for proto in descriptor_set.file:
        names: dict[tuple[int, ...], str] = {}
        for message_index, message in enumerate(proto.message_type):
            full_name = (
                f"{proto.package}.{message.name}" if proto.package else message.name
            )
            if full_name not in pending:
                continue
            pending.remove(full_name)
            path = (FileDescriptorProto.MESSAGE_TYPE_FIELD_NUMBER, message_index)
            names[path] = full_name
            for field_index, field in enumerate(message.field):
                names[(*path, DescriptorProto.FIELD_FIELD_NUMBER, field_index)] = (
                    f"{full_name}.{field.name}"
                )
        if not names:
            continue

        source = Path(proto.name)
        output_path = config.outputs.python / source.with_name(
            f"{source.stem}_pb2_docs.py"
        )
        generated_files[output_path] = _render_python_docs(proto, names)

    if pending:
        raise ProtocConfigError(
            f"Unknown documented messages: {', '.join(sorted(pending))}"
        )
    for output_path, contents in generated_files.items():
        output_path.write_text(contents, encoding="utf-8")


def compile_project(project_dir: Path) -> None:
    """Compile protobufs for the configured project."""
    config = load_protoc_config(project_dir)
    proto_files = discover_proto_files(config.proto_root, config.include_globs)
    command = build_protoc_command(config, proto_files)
    with TemporaryDirectory() as temp_dir:
        descriptors = Path(temp_dir) / "descriptors.pb"
        if config.documented_messages:
            command[1:1] = [
                f"--descriptor_set_out={descriptors}",
                "--include_source_info",
            ]
        exit_code = protoc.main(command)
        if exit_code != 0:
            raise ProtocExecutionError(
                f"`grpc_tools.protoc` failed with exit code {exit_code}"
            )
        if config.documented_messages:
            descriptor_set = FileDescriptorSet.FromString(descriptors.read_bytes())
            _write_python_docs(descriptor_set, config)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Compile protobufs for a project.")
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory containing the pyproject.toml config",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    compile_project(Path(args.project_dir))


if __name__ == "__main__":
    main()
