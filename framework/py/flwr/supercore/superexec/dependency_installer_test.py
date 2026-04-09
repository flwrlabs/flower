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
"""Tests for runtime dependency installation."""


import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from flwr.supercore.superexec import dependency_installer

# pylint: disable=protected-access


def _make_site_packages(runtime_env_dir: Path) -> None:
    site_packages = (
        runtime_env_dir
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True, exist_ok=True)


def _fake_run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    **_kwargs: object,
) -> None:
    _ = cwd
    if cmd[:4] == [sys.executable, "-m", "uv", "sync"] and env is not None:
        runtime_env_dir = Path(env["UV_PROJECT_ENVIRONMENT"])
        _make_site_packages(runtime_env_dir)


def test_exclude_flwr_dependencies_handles_extras_and_markers() -> None:
    """Ensure only the Flower package itself gets filtered out."""
    dependencies = [
        "flwr[simulation]>=1.0.0; python_version >= '3.10'",
        "numpy>=1.26.0",
        "flwr @ file:///tmp/flwr.whl",
        "flwr-datasets>=0.5.0",
        "FlWr == 1.27.0",
    ]

    filtered = dependency_installer._exclude_flwr_dependencies(dependencies)

    assert filtered == ["numpy>=1.26.0", "flwr-datasets>=0.5.0"]


def test_create_runtime_env_dir_uses_run_id_when_provided(tmp_path: Path) -> None:
    """Ensure runtime environment directory is run-ID scoped in deployment mode."""
    with patch.dict(os.environ, {"FLWR_HOME": str(tmp_path)}, clear=False):
        runtime_env_dir = dependency_installer._create_runtime_env_dir(
            project_dir=tmp_path,
            launch_id="token-a",
            run_id=123,
        )

    assert runtime_env_dir == tmp_path / "runtime-envs" / "123"


def test_install_app_dependencies_uses_resolved_index_url(tmp_path: Path) -> None:
    """Ensure resolver output is forwarded to uv sync and uv bootstrap."""
    resolved_index_url = "http://127.0.0.1:3141/root/pypi/+simple/"
    index_context: dependency_installer.RuntimeDependencyIndexContext = {
        "component": "serverapp",
        "project_dir": str(tmp_path),
        "run_id": 321,
        "launch_id": "token-a",
        "fab_id": "publisher/app",
        "fab_version": "1.0.0",
        "fab_hash": "fab-hash",
    }

    with (
        patch.dict(os.environ, {"FLWR_HOME": str(tmp_path)}, clear=False),
        patch.object(dependency_installer, "_ensure_uv_available") as ensure_uv,
        patch.object(
            dependency_installer,
            "_resolve_runtime_dependency_index_url",
            return_value=resolved_index_url,
        ) as resolve_index,
        patch.object(
            dependency_installer,
            "_get_project_dependencies",
            return_value=["numpy>=1.26.0"],
        ),
        patch.object(
            dependency_installer, "_run_cmd", side_effect=_fake_run_cmd
        ) as run_cmd,
    ):
        runtime_env = dependency_installer.install_app_dependencies(
            project_dir=tmp_path,
            launch_id="token-a",
            run_id=321,
            index_context=index_context,
        )

    resolve_index.assert_called_once_with(index_context)
    ensure_uv.assert_called_once_with(resolved_index_url)
    sync_cmd = run_cmd.call_args.args[0]
    sync_env = run_cmd.call_args.kwargs["env"]
    assert sync_cmd == [
        sys.executable,
        "-m",
        "uv",
        "sync",
        "--python",
        sys.executable,
        "--no-install-project",
        "--no-install-package",
        "flwr",
        "--inexact",
        "--index-url",
        resolved_index_url,
    ]
    assert sync_env["UV_PROJECT_ENVIRONMENT"] == str(runtime_env)
    dependency_installer.cleanup_app_runtime_environment(runtime_env)


def test_same_host_superlink_and_supernode_share_run_scoped_env(tmp_path: Path) -> None:
    """Ensure same-run app processes on one host resolve to the same env path."""

    def fake_run_cmd_with_delay(
        cmd: list[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        **_kwargs: object,
    ) -> None:
        _ = cwd
        if cmd[:4] == [sys.executable, "-m", "uv", "sync"] and env is not None:
            runtime_env_dir = Path(env["UV_PROJECT_ENVIRONMENT"])
            _make_site_packages(runtime_env_dir)
            time.sleep(0.1)

    results: list[Path] = []
    failures: list[Exception] = []

    def worker(launch_id: str) -> None:
        try:
            env_path = dependency_installer.install_app_dependencies(
                project_dir=tmp_path,
                launch_id=launch_id,
                run_id=999,
            )
            results.append(env_path)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failures.append(exc)

    with (
        patch.dict(os.environ, {"FLWR_HOME": str(tmp_path)}, clear=False),
        patch.object(dependency_installer, "_ensure_uv_available"),
        patch.object(
            dependency_installer,
            "_get_project_dependencies",
            return_value=["numpy>=1.26.0"],
        ),
        patch.object(
            dependency_installer, "_run_cmd", side_effect=fake_run_cmd_with_delay
        ),
    ):
        t1 = threading.Thread(target=worker, args=("sl-token",))
        t2 = threading.Thread(target=worker, args=("sn-token",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

    assert not failures
    assert len(results) == 2
    assert results[0] == results[1] == tmp_path / "runtime-envs" / "999"
    dependency_installer.cleanup_app_runtime_environment(results[0])


def test_install_app_dependencies_propagates_resolver_error(tmp_path: Path) -> None:
    """Ensure resolver errors are surfaced instead of silently ignored."""
    index_context: dependency_installer.RuntimeDependencyIndexContext = {
        "component": "clientapp",
        "project_dir": str(tmp_path),
        "run_id": 777,
        "launch_id": "token-x",
    }
    with patch.object(
        dependency_installer,
        "_resolve_runtime_dependency_index_url",
        side_effect=RuntimeError("resolver failed"),
    ):
        with pytest.raises(RuntimeError, match="resolver failed"):
            dependency_installer.install_app_dependencies(
                project_dir=tmp_path,
                launch_id="token-x",
                run_id=777,
                index_context=index_context,
            )


def test_cleanup_app_runtime_environment_removes_directory(tmp_path: Path) -> None:
    """Ensure cleanup removes the selected runtime environment."""
    runtime_env_dir = tmp_path / "runtime-envs" / "456"
    runtime_env_dir.mkdir(parents=True, exist_ok=True)
    (runtime_env_dir / "marker.txt").write_text("ok", encoding="utf-8")

    dependency_installer.cleanup_app_runtime_environment(runtime_env_dir)

    assert not runtime_env_dir.exists()


def test_ensure_uv_available_raises_when_uv_is_missing() -> None:
    """Ensure missing uv fails fast without bootstrap installation."""
    with patch.object(dependency_installer, "_run_cmd", return_value="uv missing"):
        with pytest.raises(
            RuntimeError,
            match="`uv` is not available in the current environment",
        ):
            dependency_installer._ensure_uv_available(
                dependency_index_url="http://127.0.0.1:3141/root/pypi/+simple/"
            )
