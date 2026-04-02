"""Utility used to bump the version of the package."""

import argparse
import re
import sys
from pathlib import Path

REPLACE_CURR_VERSION = {}

REPLACE_NEXT_VERSION = {
    "framework/pyproject.toml": ['version = "{version}"'],
    "framework/docs/source/conf.py": [
        'release = "{version}"',
        ".. |stable_flwr_version| replace:: {version}",
    ],
    "examples/docs/source/conf.py": ['release = "{version}"'],
    "baselines/docs/source/conf.py": ['release = "{version}"'],
    "framework/docker/complete/compose.yml": ["FLWR_VERSION:-{version}"],
    "framework/docker/distributed/client/compose.yml": ["FLWR_VERSION:-{version}"],
    "framework/docker/distributed/server/compose.yml": ["FLWR_VERSION:-{version}"],
    "framework/py/flwr/cli/new/templates/app/pyproject.*.toml.tpl": [
        "flwr[simulation]>={version}",
    ],
}

EXAMPLES = {
    "hub/apps/**/pyproject.toml": [
        "flwr[simulation]>={version}",
        "flwr[simulation]=={version}",
        "flwr>={version}",
        "flwr=={version}",
    ],
}

ROOT_DIR = Path(__file__).parents[2]


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


def _update_versions(file_patterns, replace_strings, new_version, check):
    """Update the version strings in the specified files."""
    wrong = False
    for pattern in file_patterns:
        files = list(ROOT_DIR.glob(pattern))
        for file_path in files:
            if not file_path.is_file():
                continue
            content = file_path.read_text()
            original_content = content
            for s in replace_strings:
                # Construct regex pattern to match any version number in the string
                escaped_s = re.escape(s).replace(r"\{version\}", r"(\d+\.\d+\.\d+)")
                regex_pattern = re.compile(escaped_s)
                content = regex_pattern.sub(s.format(version=new_version), content)
            if content != original_content:
                wrong = True
                if check:
                    print(f"{file_path} would be updated")
                else:
                    file_path.write_text(content)
                    print(f"Updated {file_path}")

    return wrong


if __name__ == "__main__":
    # Search for the latest stable release version in the CHANGELOG
    changelog_path = ROOT_DIR / "framework/docs/source/ref-changelog.md"
    with changelog_path.open("r") as f:
        for line in f:
            if match := re.match(r"^## v(\d+\.\d+\.\d+).+", line):
                break

    parser = argparse.ArgumentParser(
        description="Utility used to bump the version of the package."
    )
    parser.add_argument(
        "--old_version",
        help="Current (non-updated) version of the package, soon to be the old version.",
        default=match.group(1) if match else None,
    )
    parser.add_argument(
        "--check", action="store_true", help="Fails if any file would be modified."
    )
    parser.add_argument(
        "--no_examples",
        action="store_true",
        help="Also modify flwr version in examples.",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--patch", action="store_true", help="Increment the patch version."
    )
    group.add_argument(
        "--major", action="store_true", help="Increment the major version."
    )
    args = parser.parse_args()

    if not args.old_version:
        raise ValueError("Version not found in conf.py, please provide current version")

    # Determine the type of version increment
    if args.major:
        increment = "major"
    elif args.patch:
        increment = "patch"
    else:
        increment = "minor"

    curr_version = _get_next_version(args.old_version, increment)
    next_version = _get_next_version(curr_version, "minor")

    wrong = False

    # Update files with next version
    for file_pattern, strings in REPLACE_NEXT_VERSION.items():
        if not _update_versions([file_pattern], strings, next_version, args.check):
            wrong = True

    # Update files with current version
    for file_pattern, strings in REPLACE_CURR_VERSION.items():
        if not _update_versions([file_pattern], strings, curr_version, args.check):
            wrong = True

    if not args.no_examples:
        for file_pattern, strings in EXAMPLES.items():
            if not _update_versions([file_pattern], strings, curr_version, args.check):
                wrong = True

    if wrong and args.check:
        sys.exit("Some version haven't been updated.")
