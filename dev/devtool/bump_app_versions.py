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

"""Utility used to bump the version of the apps.

For example:

major: 1.2.3 -> 2.0.0
minor: 1.2.3 -> 1.3.0
patch: 1.2.3 -> 1.2.4

"""

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

from tomlkit.toml_file import TOMLFile

ROOT_DIR = Path(__file__).parents[2]
APPS_DIR = ROOT_DIR / "hub" / "apps"


def bump_version(version: str, level: str) -> str:
    """Bump the version string according to the specified level."""
    parts = version.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Invalid version format: {version!r}")

    major, minor, patch = map(int, parts)

    if level == "major":
        major += 1
        minor = 0
        patch = 0
    elif level == "minor":
        minor += 1
        patch = 0
    elif level == "patch":
        patch += 1
    else:
        raise ValueError(f"Unsupported level: {level!r}")

    return f"{major}.{minor}.{patch}"


def update_pyproject_version(
    pyproject_path: Path, level: str
) -> tuple[str, str] | None:
    """Update the version in the given pyproject.toml file."""
    toml_file = TOMLFile(pyproject_path)
    doc = toml_file.read()

    project = doc.get("project")
    if project is None:
        return None

    version = project.get("version")
    if version is None:
        return None

    version_str = str(version)
    new_version = bump_version(version_str, level)

    project["version"] = new_version
    toml_file.write(doc)

    return version_str, new_version


def parse_apps_arg(apps_arg: str | None) -> list[str] | None:
    """Parse the --apps argument, which can be a comma-separated list or a file path."""
    if not apps_arg:
        return None

    path = Path(apps_arg)
    if path.is_file():
        apps = []
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                apps.append(line)
        return apps or None

    apps = [a.strip() for a in apps_arg.split(",") if a.strip()]
    return apps or None


def iter_target_app_dirs(
    apps_dir: Path, selected_apps: list[str] | None
) -> Iterator[Path]:
    """Iterate over the target app directories."""
    if not apps_dir.is_dir():
        raise FileNotFoundError(f"Apps directory does not exist: {apps_dir}")

    if selected_apps is None:
        for path in sorted(apps_dir.iterdir()):
            if path.is_dir():
                yield path
        return

    for app_name in selected_apps:
        yield apps_dir / app_name


def main() -> int:
    """Main function to bump app versions."""
    parser = argparse.ArgumentParser(
        description="Bump project.version in hub/apps/*/pyproject.toml"
    )
    parser.add_argument(
        "level",
        nargs="?",
        choices=("major", "minor", "patch"),
        default="patch",
        help='Version part to bump (default: "patch")',
    )
    parser.add_argument(
        "--apps",
        help="Comma-separated app names OR a file containing app names (one per line). "
        "If not provided, all apps will be updated.",
    )
    args = parser.parse_args()

    selected_apps = parse_apps_arg(args.apps)

    updated = 0
    skipped = 0
    failed = 0

    for app_dir in iter_target_app_dirs(APPS_DIR, selected_apps):
        if not app_dir.is_dir():
            print(f"SKIP  {app_dir.name}: app directory does not exist")
            skipped += 1
            continue

        pyproject_path = app_dir / "pyproject.toml"
        if not pyproject_path.is_file():
            print(f"SKIP  {app_dir.name}: missing pyproject.toml")
            skipped += 1
            continue

        try:
            result = update_pyproject_version(pyproject_path, args.level)
            if result is None:
                print(f"SKIP  {app_dir.name}: no [project].version found")
                skipped += 1
                continue

            old_version, new_version = result
            print(f"OK    {app_dir.name}: {old_version} -> {new_version}")
            updated += 1

        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"ERROR {app_dir.name}: {exc}", file=sys.stderr)
            failed += 1

    print(f"\nDone. updated={updated}, skipped={skipped}, failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
