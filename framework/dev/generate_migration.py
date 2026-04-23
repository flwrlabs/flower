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
"""Generate Alembic migrations against the selected state schema branch."""


from __future__ import annotations

import argparse
import configparser
from pathlib import Path
import subprocess
import sys

from alembic.config import Config
from alembic.script import ScriptDirectory

from flwr.supercore.state.alembic.utils import FLWR_STATE_LATEST_REVISIONS

FRAMEWORK_DIR = Path(__file__).resolve().parent.parent
ALEMBIC_INI = FRAMEWORK_DIR / "alembic.ini"
STATE_DB = FRAMEWORK_DIR / "state.db"


def _infer_module_name(version_path: Path) -> str | None:
    """Infer the state module name from a version path."""
    if "supercore" in version_path.parts:
        return "supercore"
    if "ee" in version_path.parts:
        return "ee"
    return None


def _load_version_paths() -> dict[str, Path]:
    """Load known Alembic version paths from alembic.ini."""
    parser = configparser.ConfigParser(defaults={"here": str(FRAMEWORK_DIR)})
    parser.read(ALEMBIC_INI)
    raw_locations = parser.get("alembic", "version_locations", fallback="")

    version_paths: dict[str, Path] = {}
    for location in raw_locations.splitlines():
        candidate = location.strip()
        if not candidate:
            continue
        path = Path(candidate).resolve()
        module = _infer_module_name(path)
        if module is None or not path.exists():
            continue
        version_paths[module] = path

    return version_paths


VERSION_PATHS = _load_version_paths()
DEFAULT_MODULE = "supercore" if "supercore" in VERSION_PATHS else next(
    iter(VERSION_PATHS), None
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    if not VERSION_PATHS:
        raise RuntimeError(f"No valid Alembic version paths found in {ALEMBIC_INI}.")

    parser = argparse.ArgumentParser(
        description="Generate an Alembic migration for the selected state module."
    )
    parser.add_argument("message", help="Migration message")
    parser.add_argument(
        "--module",
        choices=sorted(VERSION_PATHS),
        default=DEFAULT_MODULE,
        help=f"Migration branch to extend (default: {DEFAULT_MODULE})",
    )
    return parser.parse_args()


def build_alembic_config() -> Config:
    """Build the Alembic config used by the generation script."""
    return Config(str(ALEMBIC_INI))


def resolve_head_for_module(module: str, config: Config) -> str:
    """Return the current Alembic head for the selected module."""
    version_path = VERSION_PATHS[module].resolve()
    script = ScriptDirectory.from_config(config)
    matching_heads: list[str] = []

    for head in script.get_heads():
        revision = script.get_revision(head)
        if revision is None or revision.path is None:
            continue
        if Path(revision.path).resolve().parent == version_path:
            matching_heads.append(head)

    if not matching_heads:
        raise RuntimeError(
            f"No Alembic head found for module '{module}' in '{version_path}'."
        )
    if len(matching_heads) > 1:
        raise RuntimeError(
            f"Multiple Alembic heads found for module '{module}': {matching_heads}. "
            "Merge the branch heads before generating a new migration."
        )
    return matching_heads[0]


def run_alembic(args: list[str]) -> None:
    """Run an Alembic command from the framework root."""
    subprocess.run(
        ["alembic", "-c", str(ALEMBIC_INI), *args],
        check=True,
        cwd=FRAMEWORK_DIR,
        capture_output=False,
    )


def main() -> None:
    """Parse arguments and generate a migration revision."""
    args = parse_args()
    config = build_alembic_config()
    target_head = resolve_head_for_module(args.module, config)
    target_version_path = VERSION_PATHS[args.module]

    if not target_version_path.exists():
        print(
            f"Version path for module '{args.module}' does not exist: "
            f"{target_version_path}"
        )
        sys.exit(1)

    try:
        print("Upgrading temporary database to all head revisions...")
        run_alembic(["upgrade", FLWR_STATE_LATEST_REVISIONS])

        print(f"Generating {args.module} migration: {args.message}")
        run_alembic(
            [
                "revision",
                "--autogenerate",
                "--head",
                target_head,
                "--version-path",
                str(target_version_path),
                "-m",
                args.message,
            ]
        )

        print("Migration generated successfully!")
    finally:
        if STATE_DB.exists():
            try:
                STATE_DB.unlink()
                print("Cleaned up temporary database file.")
            except OSError as err:
                print(f"Warning: Failed to clean up temporary database file: {err}")


if __name__ == "__main__":
    main()
