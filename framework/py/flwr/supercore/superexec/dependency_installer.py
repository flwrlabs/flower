# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Utility for installing app dependencies via uv."""


import atexit
import hashlib
import os
import re
import shutil
import subprocess
import sys
import uuid
from logging import DEBUG, ERROR, INFO, WARNING
from pathlib import Path

from flwr.common.config import get_project_config
from flwr.common.logger import log
from flwr.supercore.utils import get_flwr_home

_RUNTIME_ENV_DIR = "runtime-envs"
RuntimeDependencyIndexContext = dict[str, str | int | None]

try:
    from flwr.ee import (
        resolve_runtime_dependency_index_url as _resolve_index_url_from_ee,
    )
except ImportError:

    def _resolve_index_url_from_ee(
        _context: RuntimeDependencyIndexContext,
    ) -> str | None:
        """Fallback when `flwr.ee` resolver hook is unavailable."""
        return None


def _resolve_runtime_dependency_index_url(
    context: RuntimeDependencyIndexContext,
) -> str | None:
    resolved = _resolve_index_url_from_ee(context)
    if resolved is not None and not isinstance(resolved, str):
        raise TypeError(
            "Runtime dependency index resolver must return `str | None`, "
            f"got `{type(resolved).__name__}`."
        )
    return resolved


def install_app_dependencies(
    project_dir: str | Path,
    launch_id: str | None = None,
    run_id: int | None = None,
    index_context: RuntimeDependencyIndexContext | None = None,
) -> Path:
    """Install app dependencies from a project's pyproject.toml using ``uv sync``.

    Runs ``uv sync`` in the app's project directory and installs dependencies into
    a launch-specific runtime environment to avoid pollution between app runs.
    On success, this runtime environment is activated for the current process.

    Parameters
    ----------
    project_dir : Union[str, Path]
        Path to the installed app directory containing a ``pyproject.toml``.
    launch_id : Optional[str]
        Identifier for the app launch (for example, the execution token). Used
        when deriving a unique runtime environment directory name.
    run_id : Optional[int]
        Run identifier used to include run context in the runtime environment
        directory name.
    index_context : Optional[RuntimeDependencyIndexContext]
        Optional context passed to the EE runtime dependency index resolver.
        If EE provides an index URL, it is passed to uv as ``--index-url``.

    Returns
    -------
    Path
        Path to the runtime environment used for this app launch.
    """
    project_dir = Path(project_dir)

    dependency_index_url = (
        _resolve_runtime_dependency_index_url(index_context)
        if index_context is not None
        else None
    )

    _ensure_uv_available(dependency_index_url)

    dependencies = _get_project_dependencies(project_dir)
    installable_dependencies = _exclude_flwr_dependencies(dependencies)

    if len(installable_dependencies) != len(dependencies):
        log(
            WARNING,
            "Skipping `flwr` dependency from app requirements to avoid replacing "
            "the running Flower installation.",
        )

    if not installable_dependencies:
        log(
            INFO,
            "No non-`flwr` dependencies found in pyproject.toml. "
            "Proceeding with isolated runtime environment creation.",
        )

    runtime_env_dir = _create_runtime_env_dir(project_dir, launch_id, run_id)
    runtime_env_dir.parent.mkdir(parents=True, exist_ok=True)
    log(INFO, "Created env for run in: %s", runtime_env_dir)

    log(INFO, "Installing application dependencies...")

    sync_cmd: list[str] = [
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
    ]

    if dependency_index_url is not None:
        sync_cmd += ["--index-url", dependency_index_url]

    sync_env = os.environ.copy()
    sync_env["UV_PROJECT_ENVIRONMENT"] = str(runtime_env_dir)
    log(DEBUG, "Using UV_PROJECT_ENVIRONMENT=%s", sync_env["UV_PROJECT_ENVIRONMENT"])

    installed_packages: set[str] = set()
    sync_error = _run_cmd(
        sync_cmd,
        cwd=project_dir,
        env=sync_env,
        log_output_level=DEBUG,
        installed_packages=installed_packages,
    )
    if sync_error is not None:
        raise RuntimeError(f"uv sync failed: {sync_error}")

    if installed_packages:
        log(INFO, "Installed: [%s]", ", ".join(sorted(installed_packages)))
    else:
        log(INFO, "No additional application dependencies needed installation.")

    _activate_runtime_env(runtime_env_dir)
    if run_id is not None:
        _register_runtime_env_cleanup(runtime_env_dir)
    log(INFO, "App dependencies installed successfully via uv sync.")
    return runtime_env_dir


def _get_project_dependencies(project_dir: Path) -> list[str]:
    """Read project dependencies from pyproject.toml."""
    config = get_project_config(project_dir)
    deps = config.get("project", {}).get("dependencies", [])
    if not isinstance(deps, list):
        raise RuntimeError(
            "Invalid pyproject.toml: [project].dependencies is not a list"
        )
    return [str(dep) for dep in deps]


def _exclude_flwr_dependencies(dependencies: list[str]) -> list[str]:
    """Exclude flwr requirements to avoid replacing the running installation."""

    def _dep_name(dep: str) -> str:
        # Extract package name from PEP 508-like requirement strings.
        name = re.split(r"[\s\[<>=!~;@]", dep, maxsplit=1)[0]
        return name.strip().lower().replace("_", "-")

    return [dep for dep in dependencies if _dep_name(dep) != "flwr"]


def _create_runtime_env_dir(
    project_dir: Path,
    launch_id: str | None,
    run_id: int | None,
) -> Path:
    """Return a runtime environment path for this app launch."""
    if run_id is not None:
        env_name = str(run_id)
    else:
        project_hash = hashlib.sha256(
            str(project_dir.resolve()).encode("utf-8")
        ).hexdigest()
        launch_source = launch_id if launch_id is not None else uuid.uuid4().hex
        launch_hash = hashlib.sha256(launch_source.encode("utf-8")).hexdigest()
        nonce = uuid.uuid4().hex[:8]
        env_name = f"{project_hash[:12]}-{launch_hash[:12]}-{nonce}"
    return get_flwr_home() / _RUNTIME_ENV_DIR / env_name


def _activate_runtime_env(runtime_env_dir: Path) -> None:
    """Activate runtime environment for the current process."""
    site_packages_dirs = _find_site_packages_dirs(runtime_env_dir)
    if not site_packages_dirs:
        raise RuntimeError(
            "Unable to locate site-packages in runtime environment "
            f"{runtime_env_dir}."
        )

    for site_packages_dir in reversed(site_packages_dirs):
        site_packages = str(site_packages_dir)
        if site_packages in sys.path:
            sys.path.remove(site_packages)
        sys.path.insert(0, site_packages)

    scripts_dir = runtime_env_dir / ("Scripts" if os.name == "nt" else "bin")
    if scripts_dir.is_dir():
        os.environ["PATH"] = f"{scripts_dir}{os.pathsep}{os.environ.get('PATH', '')}"

    os.environ["VIRTUAL_ENV"] = str(runtime_env_dir)


def _find_site_packages_dirs(runtime_env_dir: Path) -> list[Path]:
    """Locate site-packages directories for a virtual environment."""
    candidates = [
        runtime_env_dir / "Lib" / "site-packages",
        runtime_env_dir
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages",
    ]
    return [path for path in candidates if path.is_dir()]


def _register_runtime_env_cleanup(runtime_env_dir: Path) -> None:
    """Register best-effort cleanup for a launch-specific runtime environment."""
    atexit.register(cleanup_app_runtime_environment, runtime_env_dir)


def cleanup_app_runtime_environment(runtime_env_dir: Path | None) -> None:
    """Best-effort cleanup for a runtime environment directory."""
    if runtime_env_dir is None:
        return

    if runtime_env_dir.exists():
        shutil.rmtree(runtime_env_dir, ignore_errors=True)
        log(DEBUG, "Cleaned up runtime environment %s", runtime_env_dir)


def _ensure_uv_available(dependency_index_url: str | None) -> None:
    """Ensure `uv` is available in the current runtime environment."""
    _ = dependency_index_url
    uv_version_cmd = [sys.executable, "-m", "uv", "--version"]
    uv_check_error = _run_cmd(uv_version_cmd, log_output_level=DEBUG)
    if uv_check_error is None:
        return

    raise RuntimeError(
        "`uv` is not available in the current environment. "
        "Please install `uv` and retry."
    )


def _run_cmd(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    log_output_level: int = INFO,
    installed_packages: set[str] | None = None,
) -> str | None:
    """Run command and return error details, or None on success."""
    log(DEBUG, "Running: %s", " ".join(cmd))
    try:
        # Stream output to Flower logs to avoid long silent periods.
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd) if cwd else None,
            env=env,
        ) as proc:
            output_lines: list[str] = []
            if proc.stdout is not None:
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        output_lines.append(line)
                        log(log_output_level, "%s", line)
                        trimmed = line.lstrip()
                        if installed_packages is not None and trimmed.startswith("+ "):
                            package_spec = trimmed[2:].strip()
                            if package_spec:
                                installed_packages.add(package_spec)

            returncode = proc.wait()
        if returncode != 0:
            details = "\n".join(output_lines[-20:]) if output_lines else ""
            return f"exit code {returncode}: {details}"

        return None
    except FileNotFoundError as exc:
        return str(exc)
    except subprocess.SubprocessError as exc:
        log(ERROR, "Command failed: %s", " ".join(cmd))
        return str(exc)
