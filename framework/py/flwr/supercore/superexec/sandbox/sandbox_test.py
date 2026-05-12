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
"""Tests for SuperExec sandbox configuration."""


import os
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from .sandbox import resolve_sandbox_config


def _make_executable(path: Path) -> str:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return str(path)


def test_disabled_sandbox_preserves_command() -> None:
    """Disabled sandbox mode should not wrap commands."""
    sandbox = resolve_sandbox_config(mode="disabled")
    command = ["flwr-clientapp", "--token", "abc"]

    assert sandbox.wrap_command(command) == command
    assert sandbox.include_parent_pid


def test_nsjail_sandbox_wraps_command_with_user_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nsjail mode should wrap commands with the resolved binary and config."""
    nsjail = _make_executable(tmp_path / "nsjail")
    serverapp = _make_executable(tmp_path / "flwr-serverapp")
    config = tmp_path / "nsjail.cfg"
    config.write_text("mode: ONCE\n", encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path))

    sandbox = resolve_sandbox_config(
        mode="nsjail",
        nsjail_config_path=str(config),
        nsjail_binary=nsjail,
    )

    assert not sandbox.include_parent_pid
    assert sandbox.wrap_command(["flwr-serverapp"]) == [
        nsjail,
        "--config",
        str(config),
        "--",
        serverapp,
    ]


def test_nsjail_sandbox_scopes_single_configured_resource(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single configured resource should be the only path mounted into nsjail."""
    nsjail = _make_executable(tmp_path / "nsjail")
    clientapp = _make_executable(tmp_path / "flwr-clientapp")
    config = tmp_path / "nsjail.cfg"
    config.write_text("mode: ONCE\n", encoding="utf-8")
    data_root = tmp_path / "data"
    ondri_path = data_root / "ondri"
    other_path = data_root / "other"
    ondri_path.mkdir(parents=True)
    other_path.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("FLWR_SUPEREXEC_SANDBOX_RESOURCE_ROOT", str(data_root))
    monkeypatch.setenv(
        "FLWR_SUPEREXEC_SANDBOX_RESOURCES_JSON", json.dumps({"ondri": str(ondri_path)})
    )

    sandbox = resolve_sandbox_config(
        mode="nsjail",
        nsjail_config_path=str(config),
        nsjail_binary=nsjail,
    )

    assert sandbox.wrap_command(["flwr-clientapp"]) == [
        nsjail,
        "--config",
        str(config),
        "--tmpfsmount",
        str(data_root),
        "--bindmount_ro",
        f"{ondri_path}:{ondri_path}",
        "--",
        clientapp,
    ]


def test_nsjail_sandbox_uses_run_resource_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run config should select the resource exposed to the app sandbox."""
    nsjail = _make_executable(tmp_path / "nsjail")
    clientapp = _make_executable(tmp_path / "flwr-clientapp")
    config = tmp_path / "nsjail.cfg"
    config.write_text("mode: ONCE\n", encoding="utf-8")
    data_root = tmp_path / "data"
    ondri_path = data_root / "ondri"
    adni_path = data_root / "adni"
    ondri_path.mkdir(parents=True)
    adni_path.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("FLWR_SUPEREXEC_SANDBOX_RESOURCE_ROOT", str(data_root))
    monkeypatch.setenv(
        "FLWR_SUPEREXEC_SANDBOX_RESOURCES_JSON",
        json.dumps({"ondri": str(ondri_path), "adni": str(adni_path)}),
    )
    run = SimpleNamespace(override_config={"sandbox.resources": "adni"})

    sandbox = resolve_sandbox_config(
        mode="nsjail",
        nsjail_config_path=str(config),
        nsjail_binary=nsjail,
    )

    wrapped = sandbox.wrap_command(["flwr-clientapp"], run=run)

    assert "--bindmount_ro" in wrapped
    assert f"{adni_path}:{adni_path}" in wrapped
    assert f"{ondri_path}:{ondri_path}" not in wrapped


def test_nsjail_sandbox_rereads_resource_file_per_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resource catalog file should be re-read for each ClientApp launch."""
    nsjail = _make_executable(tmp_path / "nsjail")
    clientapp = _make_executable(tmp_path / "flwr-clientapp")
    config = tmp_path / "nsjail.cfg"
    config.write_text("mode: ONCE\n", encoding="utf-8")
    data_root = tmp_path / "data"
    ondri_path = data_root / "ondri"
    adni_path = data_root / "adni"
    ondri_path.mkdir(parents=True)
    adni_path.mkdir()
    dataset_map = tmp_path / "sandbox_resources.json"
    dataset_map.write_text(json.dumps({"ondri": str(ondri_path)}), encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("FLWR_SUPEREXEC_SANDBOX_RESOURCE_ROOT", str(data_root))
    monkeypatch.setenv("FLWR_SUPEREXEC_SANDBOX_RESOURCES_FILE", str(dataset_map))

    sandbox = resolve_sandbox_config(
        mode="nsjail",
        nsjail_config_path=str(config),
        nsjail_binary=nsjail,
    )

    first_wrapped = sandbox.wrap_command(
        ["flwr-clientapp"],
        run=SimpleNamespace(override_config={"sandbox.resources": "ondri"}),
    )
    dataset_map.write_text(json.dumps({"adni": str(adni_path)}), encoding="utf-8")
    second_wrapped = sandbox.wrap_command(
        ["flwr-clientapp"],
        run=SimpleNamespace(override_config={"sandbox.resources": "adni"}),
    )

    assert f"{ondri_path}:{ondri_path}" in first_wrapped
    assert f"{adni_path}:{adni_path}" not in first_wrapped
    assert f"{adni_path}:{adni_path}" in second_wrapped
    assert f"{ondri_path}:{ondri_path}" not in second_wrapped


def test_nsjail_sandbox_uses_node_resource_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NODE_CONFIG should provide a fallback resource selection for SuperNode sites."""
    nsjail = _make_executable(tmp_path / "nsjail")
    clientapp = _make_executable(tmp_path / "flwr-clientapp")
    config = tmp_path / "nsjail.cfg"
    config.write_text("mode: ONCE\n", encoding="utf-8")
    data_root = tmp_path / "data"
    ondri_path = data_root / "ondri"
    adni_path = data_root / "adni"
    ondri_path.mkdir(parents=True)
    adni_path.mkdir()
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("FLWR_SUPEREXEC_SANDBOX_RESOURCE_ROOT", str(data_root))
    monkeypatch.setenv("NODE_CONFIG", 'site="site-a" sandbox.resources="ondri"')
    monkeypatch.setenv(
        "FLWR_SUPEREXEC_SANDBOX_RESOURCES_JSON",
        json.dumps({"ondri": str(ondri_path), "adni": str(adni_path)}),
    )

    sandbox = resolve_sandbox_config(
        mode="nsjail",
        nsjail_config_path=str(config),
        nsjail_binary=nsjail,
    )

    wrapped = sandbox.wrap_command(["flwr-clientapp"])

    assert f"{ondri_path}:{ondri_path}" in wrapped
    assert f"{adni_path}:{adni_path}" not in wrapped


def test_nsjail_sandbox_fails_closed_when_resource_selection_is_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Multiple configured resources require an explicit run or node selection."""
    nsjail = _make_executable(tmp_path / "nsjail")
    config = tmp_path / "nsjail.cfg"
    config.write_text("mode: ONCE\n", encoding="utf-8")
    data_root = tmp_path / "data"
    ondri_path = data_root / "ondri"
    adni_path = data_root / "adni"
    ondri_path.mkdir(parents=True)
    adni_path.mkdir()
    monkeypatch.setenv("FLWR_SUPEREXEC_SANDBOX_RESOURCE_ROOT", str(data_root))
    monkeypatch.setenv(
        "FLWR_SUPEREXEC_SANDBOX_RESOURCES_JSON",
        json.dumps({"ondri": str(ondri_path), "adni": str(adni_path)}),
    )

    sandbox = resolve_sandbox_config(
        mode="nsjail",
        nsjail_config_path=str(config),
        nsjail_binary=nsjail,
    )

    with pytest.raises(ValueError, match="multiple resources"):
        sandbox.wrap_command(["flwr-clientapp"])


def test_nsjail_sandbox_fails_when_app_executable_is_missing(
    tmp_path: Path,
) -> None:
    """Nsjail mode should fail before launch if the app binary is unresolved."""
    nsjail = _make_executable(tmp_path / "nsjail")
    config = tmp_path / "nsjail.cfg"
    config.write_text("mode: ONCE\n", encoding="utf-8")

    sandbox = resolve_sandbox_config(
        mode="nsjail",
        nsjail_config_path=str(config),
        nsjail_binary=nsjail,
    )

    with pytest.raises(ValueError, match="app executable not found"):
        sandbox.wrap_command(["missing-flwr-app"])


def test_missing_nsjail_binary_fails_closed(tmp_path: Path) -> None:
    """Explicit nsjail mode should fail if nsjail cannot be found."""
    config = tmp_path / "nsjail.cfg"
    config.write_text("mode: ONCE\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not found"):
        resolve_sandbox_config(
            mode="nsjail",
            nsjail_config_path=str(config),
            nsjail_binary=str(tmp_path / "missing-nsjail"),
        )


def test_missing_config_fails_closed(tmp_path: Path) -> None:
    """Explicit nsjail mode should fail if the config cannot be found."""
    nsjail = _make_executable(tmp_path / "nsjail")

    with pytest.raises(ValueError, match="config not found"):
        resolve_sandbox_config(
            mode="nsjail",
            nsjail_config_path=str(tmp_path / "missing.cfg"),
            nsjail_binary=nsjail,
        )


def test_unsupported_sandbox_mode_fails_closed() -> None:
    """Unsupported sandbox modes should fail before any app launch."""
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_sandbox_config(mode="bwrap")


def test_env_vars_are_used_for_nsjail_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Environment variables should configure nsjail mode."""
    nsjail = _make_executable(tmp_path / "nsjail")
    config = tmp_path / "nsjail.cfg"
    config.write_text("mode: ONCE\n", encoding="utf-8")
    monkeypatch.setenv("FLWR_SUPEREXEC_SANDBOX", "nsjail")
    monkeypatch.setenv("FLWR_SUPEREXEC_SANDBOX_CONFIG", str(config))
    monkeypatch.setenv("FLWR_SUPEREXEC_NSJAIL_BINARY", nsjail)

    sandbox = resolve_sandbox_config()

    assert sandbox.mode == "nsjail"
    assert sandbox.nsjail_binary == nsjail
    assert sandbox.nsjail_config_path == str(config)


def test_packaged_gpu_config_exists_and_matches_poc_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The packaged PoC config should be AppIO and GPU-container compatible."""
    nsjail = _make_executable(tmp_path / "nsjail")
    # Avoid depending on host nsjail while exercising the packaged config.
    monkeypatch.setenv("FLWR_SUPEREXEC_NSJAIL_BINARY", nsjail)

    sandbox = resolve_sandbox_config(mode="nsjail")

    assert sandbox.nsjail_config_path is not None
    config_text = Path(sandbox.nsjail_config_path).read_text(encoding="utf-8")
    assert os.path.basename(sandbox.nsjail_config_path) == (
        "nsjail-flower-gpu-container.cfg"
    )
    assert "clone_newpid: true" in config_text
    assert "clone_newnet: false" in config_text
    assert "clone_newuts: false" in config_text
    assert "rlimit_fsize: 2048" in config_text
    assert "rlimit_cpu: 3600" in config_text
    assert 'dst: "/root/.flwr"' in config_text
    assert 'dst: "/root/.cache"' in config_text
    assert 'dst: "/root/.config"' in config_text
    assert 'dst: "/proc"' in config_text
    assert 'fstype: "tmpfs"' in config_text
    assert 'dst: "/dev/shm"' in config_text
    assert "/dev/nvidia" in config_text
    assert 'seccomp_string: "DEFAULT ALLOW"' in config_text
