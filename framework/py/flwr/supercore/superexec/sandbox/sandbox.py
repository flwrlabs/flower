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
"""Helpers for wrapping SuperExec app launches in an app sandbox."""


import json
import os
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

SandboxMode = Literal["disabled", "nsjail"]

_DEFAULT_NSJAIL_BINARY = "nsjail"
_DEFAULT_NSJAIL_CONFIG = "nsjail-flower-gpu-container.cfg"
_SANDBOX_CONFIG_PACKAGE = "flwr.supercore.superexec.sandbox.config"
_SANDBOX_RESOURCES_ENV = "FLWR_SUPEREXEC_SANDBOX_RESOURCES_JSON"
_SANDBOX_RESOURCES_FILE_ENV = "FLWR_SUPEREXEC_SANDBOX_RESOURCES_FILE"
_SANDBOX_RESOURCE_KEYS_ENV = "FLWR_SUPEREXEC_SANDBOX_RESOURCE_KEYS"
_NODE_CONFIG_ENV = "NODE_CONFIG"
_DEFAULT_SANDBOX_RESOURCE_KEYS = (
    "sandbox.resources",
    "sandbox_resources",
    "sandbox-resource",
    "sandbox_resource",
    "sandbox-resources",
    "sandbox_resources",
    # NeuroFL compatibility aliases. These remain edge-level semantics; the
    # sandbox implementation treats the selected values as generic resources.
    "dataset-profile",
    "dataset_profile",
    "dataset",
    "dataset-profiles",
    "dataset_profiles",
    "datasets",
    "run_config.dataset-profile",
    "run_config.dataset_profile",
)


@dataclass(frozen=True)
class SandboxResource:
    """Operator-approved resource that can be exposed to an app sandbox."""

    name: str
    source: Path
    target: Path
    mode: Literal["ro", "rw"] = "ro"
    root: Path | None = None


@dataclass(frozen=True)
class SandboxConfig:
    """Resolved SuperExec app sandbox configuration."""

    mode: SandboxMode = "disabled"
    nsjail_binary: str = _DEFAULT_NSJAIL_BINARY
    nsjail_config_path: str | None = None

    @property
    def enabled(self) -> bool:
        """Return whether app launches should be sandboxed."""
        return self.mode != "disabled"

    @property
    def include_parent_pid(self) -> bool:
        """Return whether app commands should include parent PID monitoring."""
        return not self.enabled

    def wrap_command(self, command: list[str], run: Any | None = None) -> list[str]:
        """Wrap an app launch command in the configured sandbox."""
        if self.mode == "disabled":
            return command
        if self.mode == "nsjail" and self.nsjail_config_path is not None:
            resolved_command = _resolve_app_command(command)
            resource_args = _resolve_resource_args(run)
            return [
                self.nsjail_binary,
                "--config",
                self.nsjail_config_path,
                *resource_args,
                "--",
                *resolved_command,
            ]
        raise ValueError(f"Unsupported SuperExec sandbox mode: {self.mode}")


def resolve_sandbox_config(
    mode: str | None = None,
    nsjail_config_path: str | None = None,
    nsjail_binary: str | None = None,
) -> SandboxConfig:
    """Resolve and validate SuperExec sandbox settings.

    Explicit nsjail requests fail closed. Disabled mode ignores unused nsjail
    settings to preserve existing SuperExec behavior.
    """
    resolved_mode = mode or os.getenv("FLWR_SUPEREXEC_SANDBOX", "disabled")
    resolved_config = nsjail_config_path or os.getenv("FLWR_SUPEREXEC_SANDBOX_CONFIG")
    resolved_binary = (
        nsjail_binary
        or os.getenv("FLWR_SUPEREXEC_NSJAIL_BINARY")
        or _DEFAULT_NSJAIL_BINARY
    )

    if resolved_mode not in {"disabled", "nsjail"}:
        raise ValueError(
            "Unsupported SuperExec sandbox mode "
            f"'{resolved_mode}'. Expected one of: disabled, nsjail."
        )

    if resolved_mode == "disabled":
        return SandboxConfig(mode="disabled")

    if not resolved_binary.strip():
        raise ValueError("SuperExec nsjail binary must not be empty.")

    executable = _resolve_executable(resolved_binary)
    config_path = _resolve_config_path(resolved_config)
    return SandboxConfig(
        mode="nsjail",
        nsjail_binary=executable,
        nsjail_config_path=config_path,
    )


def _resolve_executable(binary: str) -> str:
    """Resolve an nsjail binary name or path to an executable."""
    expanded = Path(binary).expanduser()
    if expanded.is_absolute() or os.sep in binary:
        if not expanded.is_file():
            raise ValueError(f"SuperExec nsjail binary not found: {expanded}")
        if not os.access(expanded, os.X_OK):
            raise ValueError(f"SuperExec nsjail binary is not executable: {expanded}")
        return str(expanded)

    resolved = shutil.which(binary)
    if resolved is None:
        raise ValueError(f"SuperExec nsjail binary not found in PATH: {binary}")
    return resolved


def _resolve_config_path(config_path: str | None) -> str:
    """Resolve an explicit or packaged nsjail config path."""
    if config_path:
        resolved = Path(config_path).expanduser()
    else:
        resolved = Path(
            str(files(_SANDBOX_CONFIG_PACKAGE).joinpath(_DEFAULT_NSJAIL_CONFIG))
        )

    if not resolved.is_file():
        raise ValueError(f"SuperExec nsjail config not found: {resolved}")
    if not os.access(resolved, os.R_OK):
        raise ValueError(f"SuperExec nsjail config is not readable: {resolved}")
    return str(resolved)


def _resolve_app_command(command: list[str]) -> list[str]:
    """Resolve app executable to an absolute path for nsjail execve."""
    if not command:
        raise ValueError("SuperExec app command must not be empty.")

    executable = command[0]
    expanded = Path(executable).expanduser()
    if expanded.is_absolute() or os.sep in executable:
        return [str(expanded), *command[1:]]

    resolved = shutil.which(executable)
    if resolved is None:
        raise ValueError(
            f"SuperExec app executable not found in PATH for nsjail: {executable}"
        )
    return [resolved, *command[1:]]


def _resolve_resource_args(run: Any | None) -> list[str]:
    """Return nsjail args that expose only selected sandbox resources.

    A sandbox resource catalog is an operator-approved map of logical resource
    names to mount actions. Runs select resources by logical name through run
    config (for example ``sandbox.resources=ondri``), never by raw filesystem
    path. The sandbox hides each configured resource root, then re-binds only
    the selected resources.
    """
    resources = _load_sandbox_resources()
    if not resources:
        return []

    selected_names = _resolve_selected_resource_names(run, resources)

    mount_args: list[str] = []
    hidden_roots: set[Path] = set()
    selected_resources = [resources[name] for name in selected_names]
    for resource in selected_resources:
        root = resource.root
        if root is not None and root not in hidden_roots:
            mount_args.extend(["--tmpfsmount", str(root)])
            hidden_roots.add(root)

    for resource in selected_resources:
        mount_flag = "--bindmount_ro" if resource.mode == "ro" else "--bindmount"
        mount_args.extend(
            [mount_flag, f"{resource.source}:{resource.target}"]
        )
    return mount_args


def _load_sandbox_resources() -> dict[str, SandboxResource]:
    raw_json = os.getenv(_SANDBOX_RESOURCES_ENV, "").strip()
    raw_file = os.getenv(_SANDBOX_RESOURCES_FILE_ENV, "").strip()
    if not raw_json and not raw_file:
        return {}

    raw_maps: list[tuple[str, Any]] = []
    if raw_json:
        raw_maps.append(
            (
                _SANDBOX_RESOURCES_ENV,
                _parse_sandbox_resources_json(raw_json, _SANDBOX_RESOURCES_ENV),
            )
        )
    if raw_file:
        raw_maps.append(
            (
                _SANDBOX_RESOURCES_FILE_ENV,
                _read_sandbox_resources_file(raw_file),
            )
        )

    parsed: dict[str, Any] = {}
    for _, raw_map in raw_maps:
        parsed.update(raw_map)

    source = _sandbox_resources_source_name(raw_json, raw_file)
    return _validate_sandbox_resources(parsed, source)


def _parse_sandbox_resources_json(raw: str, source: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"{source} must be a JSON object.")
    if "resources" in parsed:
        resources = parsed["resources"]
        if not isinstance(resources, dict):
            raise ValueError(f"{source} field 'resources' must be a JSON object.")
        return resources
    return parsed


def _read_sandbox_resources_file(raw_path: str) -> dict[str, Any]:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ValueError(
            f"{_SANDBOX_RESOURCES_FILE_ENV} must be absolute: {raw_path}"
        )
    if not path.is_file():
        raise ValueError(f"{_SANDBOX_RESOURCES_FILE_ENV} does not exist: {path}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"{_SANDBOX_RESOURCES_FILE_ENV} is not readable: {path}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"{_SANDBOX_RESOURCES_FILE_ENV} cannot be read: {path}"
        ) from exc

    return _parse_sandbox_resources_json(raw, _SANDBOX_RESOURCES_FILE_ENV)


def _sandbox_resources_source_name(raw_json: str, raw_file: str) -> str:
    if raw_json and raw_file:
        return f"{_SANDBOX_RESOURCES_ENV}/{_SANDBOX_RESOURCES_FILE_ENV}"
    if raw_file:
        return _SANDBOX_RESOURCES_FILE_ENV
    return _SANDBOX_RESOURCES_ENV


def _validate_sandbox_resources(
    parsed: dict[str, Any], source: str
) -> dict[str, SandboxResource]:
    resources: dict[str, SandboxResource] = {}
    for name, definition in parsed.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{source} contains an invalid profile key.")
        resources[name] = _validate_sandbox_resource(name, definition, source)
    return resources


def _validate_sandbox_resource(
    name: str, definition: Any, source: str
) -> SandboxResource:
    if isinstance(definition, str):
        definition = {"type": "mount", "source": definition}
    if not isinstance(definition, dict):
        raise ValueError(f"{source} entry for '{name}' must be a string or object.")

    resource_type = definition.get("type", "mount")
    if resource_type != "mount":
        raise ValueError(
            f"{source} entry for '{name}' has unsupported type: {resource_type}"
        )

    source_path = _resolve_absolute_path(
        definition.get("source") or definition.get("path"),
        f"{source} entry for '{name}' source",
    )
    target_path = _resolve_absolute_path(
        definition.get("target") or definition.get("destination") or source_path,
        f"{source} entry for '{name}' target",
    )
    root_value = definition.get("root")
    root_path = (
        _resolve_absolute_path(root_value, f"{source} entry for '{name}' root")
        if root_value
        else source_path.parent
    )
    mode = definition.get("mode", "ro")
    if mode not in {"ro", "rw"}:
        raise ValueError(f"{source} entry for '{name}' mode must be 'ro' or 'rw'.")
    if not _is_relative_to(source_path, root_path):
        raise ValueError(
            f"{source} entry for '{name}' source must be under {root_path}: "
            f"{source_path}"
        )
    if not source_path.exists():
        raise ValueError(f"{source} entry for '{name}' source does not exist.")

    return SandboxResource(
        name=name,
        source=source_path,
        target=target_path,
        mode=mode,
        root=root_path,
    )


def _resolve_absolute_path(raw_path: Any, field_name: str) -> Path:
    if isinstance(raw_path, Path):
        return raw_path
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{field_name} must be a non-empty path.")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{field_name} must be absolute: {raw_path}")
    return path


def _resolve_selected_resource_names(
    run: Any | None, resources: dict[str, SandboxResource]
) -> list[str]:
    names = _resource_names_from_run_config(run)
    if not names:
        names = _resource_names_from_node_config()
    if not names and len(resources) == 1:
        names = [next(iter(resources))]
    if not names:
        raise ValueError(
            "The sandbox resource catalog defines multiple resources, but no "
            "sandbox resource was selected in run config or NODE_CONFIG. Set "
            "a run config key such as 'sandbox.resources'."
        )

    missing = [name for name in names if name not in resources]
    if missing:
        raise ValueError(
            "Selected sandbox resource(s) are not present in the resource "
            f"catalog: {', '.join(missing)}"
        )
    return names


def _resource_names_from_run_config(run: Any | None) -> list[str]:
    if run is None:
        return []
    run_config = getattr(run, "override_config", None)
    if not isinstance(run_config, dict):
        return []

    keys = _sandbox_resource_keys()
    for key in keys:
        if key in run_config:
            return _coerce_resource_names(run_config[key])
    return []


def _sandbox_resource_keys() -> list[str]:
    raw_keys = os.getenv(_SANDBOX_RESOURCE_KEYS_ENV, "").strip()
    if not raw_keys:
        return list(_DEFAULT_SANDBOX_RESOURCE_KEYS)
    return [key.strip() for key in raw_keys.split(",") if key.strip()]


def _resource_names_from_node_config() -> list[str]:
    node_config = os.getenv(_NODE_CONFIG_ENV, "")
    for key in _sandbox_resource_keys():
        for part in node_config.split():
            if not part.startswith(f"{key}="):
                continue
            value = part.split("=", maxsplit=1)[1].strip()
            return _coerce_resource_names(value.strip("\"'"))
    return []


def _coerce_resource_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    "Sandbox resource lists must contain non-empty strings."
                )
            names.append(item.strip())
        return names
    raise ValueError(
        "Sandbox resource config must be a string, comma-separated string, or list."
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
