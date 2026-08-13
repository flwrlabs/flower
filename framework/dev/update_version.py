"""Update version references after a Framework minor release."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


ROOT_DIR = Path(__file__).parents[2]
STABLE_DOCS = (
    "baselines/docs/source/conf.py",
    "examples/docs/source/conf.py",
)
COMPOSE_FILES = (
    "framework/docker/complete/compose.yml",
    "framework/docker/distributed/client/compose.yml",
    "framework/docker/distributed/server/compose.yml",
)
DOCKER_READMES = {
    "framework/docker/base/README.md": "base",
    "framework/docker/superexec/README.md": "superexec",
    "framework/docker/superlink/README.md": "superlink",
    "framework/docker/supernode/README.md": "supernode",
}
VERSION_RE = r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"


def _versions(value: str) -> tuple[str, str]:
    """Validate X.Y.0 and return the released and next minor versions."""
    match = re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.0", value)
    if match is None:
        raise argparse.ArgumentTypeError("released version must use X.Y.0 format")
    return value, f"{int(match.group(1))}.{int(match.group(2)) + 1}.0"


def _date(value: str) -> str:
    """Validate the date used in the concrete nightly tag."""
    try:
        parsed = datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD format") from exc
    if parsed.strftime("%Y%m%d") != value:
        raise argparse.ArgumentTypeError("date must use YYYYMMDD format")
    return value


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _replace(
    content: str,
    pattern: str,
    replacement: str,
    description: str,
    count: int | None = 1,
) -> str:
    """Replace a required pattern and fail if its shape changed unexpectedly."""
    updated, matches = re.subn(pattern, replacement, content, flags=re.MULTILINE)
    valid = matches > 0 if count is None else matches == count
    if not valid:
        expected = "at least one" if count is None else str(count)
        raise ValueError(f"Expected {expected} {description}, found {matches}")
    return updated


def _set_toml_string(content: str, key: str, value: str, path: Path) -> str:
    return _replace(
        content,
        rf'^{re.escape(key)}\s*=\s*"[^"]+"$',
        f'{key} = "{value}"',
        f"{key} in {path}",
    )


def _update_example(content: str, path: Path, released: str) -> str:
    """Update FAB v1 metadata and bump the App patch version once."""
    config = tomllib.loads(content)
    app = config.get("tool", {}).get("flwr", {}).get("app", {})
    if app.get("fab-format-version") != 1:
        return content

    target = app.get("flwr-version-target")
    if not isinstance(target, str):
        raise ValueError(f"FAB v1 example {path} has no flwr-version-target")
    if target == released:
        return content

    app_version = config.get("project", {}).get("version")
    match = re.fullmatch(VERSION_RE, app_version or "")
    if match is None:
        raise ValueError(f"FAB v1 example {path} has invalid project version")
    major, minor, patch = map(int, app_version.split("."))

    content = _set_toml_string(content, "flwr-version-target", released, path)
    return _set_toml_string(content, "version", f"{major}.{minor}.{patch + 1}", path)


def _docker_matrix(root: Path, released: str) -> dict[str, Any]:
    """Load stable tags from the existing Docker matrix generator."""
    command = [
        sys.executable,
        str(root / "framework/dev/build-docker-image-matrix.py"),
        "--flwr-version",
        released,
        "--matrix",
        "stable",
    ]
    try:
        output = subprocess.run(
            command, cwd=root, check=True, capture_output=True, text=True
        ).stdout
        return json.loads(output)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ValueError("Failed to generate the stable Docker matrix") from exc


def _docker_tag_groups(
    matrix: dict[str, Any], repository: str, released: str
) -> list[list[str]]:
    """Return README tag groups in display order."""
    images = (
        matrix["base"]["images"] if repository == "base" else matrix["binary"]["images"]
    )
    if repository != "base":
        images = [
            image
            for image in images
            if image["namespace_repository"] == f"flwr/{repository}"
        ]

    groups = []
    for image in reversed(images):
        tags = image["tags_encoded"].splitlines()
        tags.sort(key=lambda tag: (tag != released, tag == "latest"))
        groups.append(tags)
    if not groups:
        raise ValueError(f"No Docker tags found for flwr/{repository}")
    return groups


def _update_docker_readme(
    content: str,
    repository: str,
    released: str,
    next_version: str,
    nightly_date: str,
    matrix: dict[str, Any],
) -> str:
    """Add the release tags and move latest away from the previous release."""
    groups = _docker_tag_groups(matrix, repository, released)

    if repository != "base":
        latest = next(group for group in groups if "latest" in group)
        points_to = " and ".join(f"`{tag}`" for tag in latest if tag != "latest")
        content = _replace(
            content,
            r"^(?P<prefix>- `latest`\r?\n  - points to ).+$",
            rf"\g<prefix>{points_to}",
            f"latest description in flwr/{repository} README",
        )

    content = _replace(
        content,
        r"^(?P<prefix>- `nightly`, `<version>\.dev<YYYYMMDD>` e\.g\. `)[^`]+`$",
        rf"\g<prefix>{next_version}.dev{nightly_date}`",
        f"nightly tag in flwr/{repository} README",
    )

    # Remove `latest` from historical release tag lines.
    lines = content.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if re.match(r"^- `\d+\.\d+\.\d+", line) and "`latest`" in line:
            if f"`{released}" not in line:
                lines[index] = line.replace(", `latest`", "")
    content = "".join(lines)

    tag_block = "".join(
        f"- {', '.join(f'`{tag}`' for tag in group)}\n" for group in groups
    )
    if re.search(rf"^- `{re.escape(released)}(?:`|-)", content, re.MULTILINE) is None:
        nightly = re.search(
            r"^- `nightly`[^\r\n]+\r?\n  - uses Python [^\r\n]+\r?\n",
            content,
            re.MULTILINE,
        )
        if nightly is None:
            raise ValueError(f"Nightly block not found in flwr/{repository} README")
        content = content[: nightly.end()] + tag_block + content[nightly.end() :]

    latest_lines = [
        line
        for line in content.splitlines()
        if re.match(r"^- `\d+\.\d+\.\d+", line) and "`latest`" in line
    ]
    expected = 0 if repository == "base" else 1
    if len(latest_lines) != expected:
        raise ValueError(f"Unexpected latest tags in flwr/{repository} README")
    return content


def _collect_updates(
    root: Path, released: str, next_version: str, nightly_date: str
) -> dict[Path, str]:
    """Compute all updates before writing any files."""
    updates = {}

    pyproject = root / "framework/pyproject.toml"
    current = tomllib.loads(_read(pyproject))["project"]["version"]
    if current not in {released, next_version}:
        raise ValueError(f"Unexpected Framework version: {current}")
    updates[pyproject] = _set_toml_string(
        _read(pyproject), "version", next_version, pyproject
    )

    lock = root / "framework/uv.lock"
    updates[lock] = _replace(
        _read(lock),
        rf'(^\[\[package]]\r?\nname = "flwr"\r?\n)version = "{VERSION_RE}"'
        r'(\r?\nsource = \{ editable = "\." \})',
        rf'\g<1>version = "{next_version}"\g<2>',
        "editable flwr version in framework/uv.lock",
    )

    framework_docs = root / "framework/docs/source/conf.py"
    content = _replace(
        _read(framework_docs),
        rf'^release = "{VERSION_RE}"$',
        f'release = "{next_version}"',
        "Framework docs release",
    )
    updates[framework_docs] = _replace(
        content,
        rf"^\.\. \|stable_flwr_version\| replace:: {VERSION_RE}$",
        f".. |stable_flwr_version| replace:: {next_version}",
        "Framework docs version substitution",
    )

    for relative in STABLE_DOCS:
        path = root / relative
        updates[path] = _replace(
            _read(path),
            rf'^release = "{VERSION_RE}"$',
            f'release = "{released}"',
            f"release in {relative}",
        )

    for relative in COMPOSE_FILES:
        path = root / relative
        updates[path] = _replace(
            _read(path),
            rf"FLWR_VERSION:-{VERSION_RE}",
            f"FLWR_VERSION:-{next_version}",
            f"version in {relative}",
            count=None,
        )

    examples = sorted((root / "examples").glob("**/pyproject.toml"))
    for path in examples:
        updates[path] = _update_example(_read(path), path, released)

    matrix = _docker_matrix(root, released)
    for relative, repository in DOCKER_READMES.items():
        path = root / relative
        updates[path] = _update_docker_readme(
            _read(path), repository, released, next_version, nightly_date, matrix
        )
    return updates


def main(argv: Sequence[str] | None = None) -> int:
    """Run the version update."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--released-version", required=True, type=_versions)
    parser.add_argument("--date", type=_date, default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    released, next_version = args.released_version

    try:
        updates = _collect_updates(ROOT_DIR, released, next_version, args.date)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    changed = [path for path, content in updates.items() if _read(path) != content]
    for path in changed:
        print(f"{'Would update' if args.check else 'Updated'} {path}")
    if args.check:
        return int(bool(changed))
    for path in changed:
        path.write_text(updates[path], encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
