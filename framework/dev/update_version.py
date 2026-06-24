"""Utility used to bump the version of the package."""

import argparse
import datetime
import re
import sys
from pathlib import Path

REPLACE_NEXT_VERSION = {
    "framework/pyproject.toml": ['version = "{version}"'],
    "framework/uv.lock": ['name = "flwr"\nversion = "{version}"'],
    "framework/docs/source/conf.py": [
        'release = "{version}"',
        ".. |stable_flwr_version| replace:: {version}",
    ],
    "examples/docs/source/conf.py": ['release = "{version}"'],
    "baselines/docs/source/conf.py": ['release = "{version}"'],
    "framework/docker/complete/compose.yml": ["FLWR_VERSION:-{version}"],
    "framework/docker/distributed/client/compose.yml": ["FLWR_VERSION:-{version}"],
    "framework/docker/distributed/server/compose.yml": ["FLWR_VERSION:-{version}"],
}

ROOT_DIR = Path(__file__).parents[2]
VERSION_PATTERN = r"\d+\.\d+\.\d+"


def _read_current_version():
    """Read the current Flower version from framework/pyproject.toml."""
    pyproject = ROOT_DIR / "framework/pyproject.toml"
    match = re.search(
        rf'^version = "({VERSION_PATTERN})"$',
        pyproject.read_text(),
        flags=re.MULTILINE,
    )
    if not match:
        raise ValueError("Version not found in framework/pyproject.toml")
    return match.group(1)


def _get_next_version(curr_version, increment):
    """Calculate the next version based on the type of release."""
    major, minor, patch_version = map(int, curr_version.split("."))
    if increment == "patch":
        patch_version += 1
    elif increment == "minor":
        minor += 1
        patch_version = 0
    elif increment == "major":
        major += 1
        minor = 0
        patch_version = 0
    else:
        raise ValueError(
            "Invalid increment type. Must be 'major', 'minor', or 'patch'."
        )
    return f"{major}.{minor}.{patch_version}"


def _bump_patch_version(version):
    """Increment the patch part of a version string."""
    major, minor, patch_version = map(int, version.split("."))
    return f"{major}.{minor}.{patch_version + 1}"


def _write_if_changed(file_path, original_content, content, check):
    """Write a file if changed, or report it in check mode."""
    if content == original_content:
        return False

    if check:
        print(f"{file_path} would be updated")
    else:
        file_path.write_text(content)
        print(f"Updated {file_path}")
    return True


def _update_versions(file_pattern, replace_strings, new_version, check):
    """Update the version strings in the specified files."""
    changed = False
    for file_path in sorted(ROOT_DIR.glob(file_pattern)):
        if not file_path.is_file():
            continue

        content = file_path.read_text()
        original_content = content
        for s in replace_strings:
            pattern = re.escape(s).replace(r"\{version\}", f"({VERSION_PATTERN})")
            content = re.sub(pattern, s.format(version=new_version), content)

        changed |= _write_if_changed(file_path, original_content, content, check)

    return changed


def _update_example_versions(current_version, check):
    """Update app target versions and bump app patch versions."""
    changed = False
    target_pattern = re.compile(
        rf'^(flwr-version-target\s*=\s*")({VERSION_PATTERN})(")$',
        flags=re.MULTILINE,
    )
    project_version_pattern = re.compile(
        rf'(?m)^(version\s*=\s*")({VERSION_PATTERN})(")$'
    )

    for file_path in sorted((ROOT_DIR / "examples").glob("**/pyproject.toml")):
        content = file_path.read_text()
        match = target_pattern.search(content)
        if not match or match.group(2) == current_version:
            continue

        original_content = content
        content = target_pattern.sub(
            rf"\g<1>{current_version}\g<3>",
            content,
            count=1,
        )
        content = project_version_pattern.sub(
            lambda m: f"{m.group(1)}{_bump_patch_version(m.group(2))}{m.group(3)}",
            content,
            count=1,
        )
        changed |= _write_if_changed(file_path, original_content, content, check)

    return changed


def _docker_tag_lines(image_dir, version):
    """Return the stable Docker tag lines for the README."""
    if image_dir.name == "base":
        return [
            f"- `{version}-py3.13-alpine3.22`",
            f"- `{version}-py3.13-ubuntu24.04`",
            f"- `{version}-py3.12-ubuntu24.04`",
            f"- `{version}-py3.11-ubuntu24.04`",
        ]

    if image_dir.name == "superlink":
        return [
            f"- `{version}`, `{version}-py3.13-alpine3.22`",
            f"- `{version}-py3.13-ubuntu24.04`, `latest`",
        ]

    if image_dir.name == "supernode":
        return [
            f"- `{version}`, `{version}-py3.13-alpine3.22`",
            f"- `{version}-py3.13-ubuntu24.04`, `latest`",
            f"- `{version}-py3.12-ubuntu24.04`",
            f"- `{version}-py3.11-ubuntu24.04`",
        ]

    return [
        f"- `{version}`, `{version}-py3.13-ubuntu24.04`, `latest`",
        f"- `{version}-py3.12-ubuntu24.04`",
        f"- `{version}-py3.11-ubuntu24.04`",
    ]


def _update_docker_readmes(current_version, check):
    """Update Docker README supported tags."""
    changed = False
    today = datetime.date.today().strftime("%Y%m%d")

    for image_name in ("base", "superexec", "superlink", "supernode"):
        file_path = ROOT_DIR / "framework/docker" / image_name / "README.md"
        content = file_path.read_text()
        original_content = content
        image_dir = file_path.parent

        content = re.sub(
            rf"(`nightly`, `<version>\.dev<YYYYMMDD>` e\.g\. `)"
            rf"{VERSION_PATTERN}\.dev\d{{8}}(`)",
            rf"\g<1>{current_version}.dev{today}\g<2>",
            content,
        )
        content = re.sub(
            rf"(`({VERSION_PATTERN})-py3\.13-ubuntu24\.04`), `latest`",
            lambda m: m.group(0) if m.group(2) == current_version else m.group(1),
            content,
        )

        if image_dir.name == "superexec":
            content = re.sub(
                rf"^  - points to `{VERSION_PATTERN}` and "
                rf"`{VERSION_PATTERN}-py3\.13-ubuntu24\.04`$",
                f"  - points to `{current_version}` and "
                f"`{current_version}-py3.13-ubuntu24.04`",
                content,
                count=1,
                flags=re.MULTILINE,
            )
        elif image_dir.name != "base":
            content = re.sub(
                rf"^  - points to `{VERSION_PATTERN}-py3\.13-ubuntu24\.04`$",
                f"  - points to `{current_version}-py3.13-ubuntu24.04`",
                content,
                count=1,
                flags=re.MULTILINE,
            )

        tag_lines = _docker_tag_lines(image_dir, current_version)
        if not all(line in content for line in tag_lines):
            content = re.sub(
                rf"^- `{VERSION_PATTERN}.*$",
                lambda m: "\n".join(tag_lines) + "\n" + m.group(0),
                content,
                count=1,
                flags=re.MULTILINE,
            )

        changed |= _write_if_changed(file_path, original_content, content, check)

    return changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility used to bump the version of the package."
    )
    parser.add_argument(
        "--check", action="store_true", help="Fails if any file would be modified."
    )
    parser.add_argument(
        "--no_examples",
        action="store_true",
        help="Skip example app version and target updates.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--patch", action="store_true", help="Increment the patch version."
    )
    group.add_argument(
        "--major", action="store_true", help="Increment the major version."
    )
    args = parser.parse_args()

    # Determine the type of version increment
    if args.major:
        increment = "major"
    elif args.patch:
        increment = "patch"
    else:
        increment = "minor"

    curr_version = _read_current_version()
    next_version = _get_next_version(curr_version, increment)

    changed = False

    # Update files with next version
    for file_pattern, strings in REPLACE_NEXT_VERSION.items():
        changed |= _update_versions(file_pattern, strings, next_version, args.check)

    if not args.no_examples:
        changed |= _update_example_versions(curr_version, args.check)

    changed |= _update_docker_readmes(curr_version, args.check)

    if changed and args.check:
        sys.exit(1)
