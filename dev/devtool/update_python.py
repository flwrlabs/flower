"""Script to update Python versions in the codebase."""

import argparse
import re
from pathlib import Path
from typing import Callable

Replacement = str | Callable[[re.Match[str]], str]
ReplacementsByPattern = dict[str, list[tuple[str, Replacement]]]


def _compute_old_version(new_version: str) -> str:
    """Compute the old version as the immediate previous minor version."""
    major_str, minor_str = new_version.split(".")
    major = int(major_str)
    minor = int(minor_str)

    if minor > 0:
        old_minor = minor - 1
        old_version = f"{major}.{old_minor}"
    else:
        raise ValueError("Minor version is 0, can't infer previous version.")
    return old_version


def _update_python_versions(
    new_full_version: str,
    patch_only: bool = False,
    dry_run: bool = False,
) -> None:
    """Update Python version strings in the specified files."""
    new_major_minor = ".".join(new_full_version.split(".")[:2])
    replacements: ReplacementsByPattern

    if patch_only:
        print(f"Updating patch version for {new_major_minor} to {new_full_version}")

        # Define the version pattern to match any full version with the same major.minor
        version_pattern = re.escape(new_major_minor) + r"\.\d+"

        # Define the file patterns and corresponding replacement patterns
        replacements = {
            # Shell scripts
            "dev/*.sh": [
                # Update version in scripts
                (
                    r"(version=\$\{1:-)" + version_pattern + r"(\})",
                    r"\g<1>" + new_full_version + r"\g<2>",
                ),
                # Update pyenv uninstall commands
                (
                    r"(pyenv uninstall -f flower-)" + version_pattern,
                    r"\g<1>" + new_full_version,
                ),
            ],
            # Python files
            "**/*.py": [
                # Update version assignments
                (
                    r'(["\'])' + version_pattern + r'(["\'])',
                    r"\g<1>" + new_full_version + r"\g<2>",
                ),
            ],
            # Documentation files
            "framework/docs/source/conf.py": [
                # Update Python full version in conf.py
                (
                    r"(\.\.\s*\|python_full_version\|\s*replace::\s*)"
                    + version_pattern,
                    r"\g<1>" + new_full_version,
                ),
            ],
        }
    else:
        # Compute old_version as immediate previous minor version
        old_version = _compute_old_version(new_major_minor)

        print(f"Determined old version: {old_version}")
        print(
            f"Updating to new version: {new_major_minor} "
            f"(full version: {new_full_version})"
        )

        # Define the file patterns and corresponding replacement patterns
        replacements = {
            # action.yml files
            ".github/actions/bootstrap/action.yml": [
                # Update default Python version
                (
                    r"^(\s*default:\s*)" + re.escape(old_version) + r"(\s*)$",
                    r"\g<1>" + new_major_minor + r"\g<2>",
                ),
            ],
            # YAML workflow files
            ".github/workflows/*.yml": [
                # Update specific python-version entries
                (
                    r"^(\s*python-version:\s*)" + re.escape(old_version) + r"(\s*)$",
                    r"\g<1>" + new_major_minor + r"\g<2>",
                ),
                (
                    r"(['\"]?)" + re.escape(old_version) + r"(['\"]?,?\s*)",
                    lambda m: (
                        "" if m.group(2).strip() == "," else ""
                    ),  # Handle the case where a comma follows
                ),
            ],
            # Shell scripts
            "dev/*.sh": [
                # Update version in scripts
                (
                    r"(version=\$\{1:-)" + re.escape(old_version) + r"(\.\d+)?(\})",
                    r"\g<1>" + new_full_version + r"\g<3>",
                ),
                # Update pyenv uninstall commands
                (
                    r"(pyenv uninstall -f flower-)"
                    + re.escape(old_version)
                    + r"(\.\d+)?",
                    r"\g<1>" + new_full_version,
                ),
            ],
            # pyproject.toml files
            "**/pyproject.toml": [
                # Update python version constraints
                (
                    r'(python\s*=\s*">=)'
                    + re.escape(old_version)
                    + r'(,\s*<\d+\.\d+")',
                    r"\g<1>" + new_major_minor + r"\g<2>",
                ),
            ],
            "dev/devtool/*.py": [
                # Update version assignments
                (
                    r'(["\'])' + re.escape(old_version) + r'(\.\d+)?(["\'],?)\s*\n?',
                    lambda m: (
                        "" if m.group(3) == "," else ""
                    ),  # Remove version and handle comma if present
                ),
            ],
            # Python files
            "**/*.py": [
                # Update version assignments
                (
                    r'(["\'])' + re.escape(old_version) + r'(\.\d+)?(["\'])',
                    r"\g<1>" + new_full_version + r"\g<3>",
                ),
            ],
            # Documentation files
            "framework/docs/source/conf.py": [
                # Update Python version in conf.py
                (
                    r"(\.\.\s*\|python_version\|\s*replace::\s*)"
                    + re.escape(old_version),
                    r"\g<1>" + new_major_minor,
                ),
                # Update Python full version in conf.py
                (
                    r"(\.\.\s*\|python_full_version\|\s*replace::\s*)"
                    + re.escape(old_version)
                    + r"\.\d+",
                    r"\g<1>" + new_full_version,
                ),
            ],
            # ReStructuredText files
            "framework/docs/source/*.rst": [
                # Update Python version in rst files
                (
                    r"(`Python\s*"
                    + re.escape(old_version)
                    + r"\s*<https://docs.python.org/"
                    + re.escape(old_version)
                    + r"/>`_)",
                    r"`Python "
                    + new_major_minor
                    + " <https://docs.python.org/"
                    + new_major_minor
                    + "/>`_",
                ),
            ],
            # PO files for localization
            "framework/docs/locales/*/LC_MESSAGES/framework-docs.po": [
                # Update Python version in localization files
                (
                    r"(`Python\s*"
                    + re.escape(old_version)
                    + r"\s*<https://docs.python.org/"
                    + re.escape(old_version)
                    + r"/>`_)",
                    r"`Python "
                    + new_major_minor
                    + " <https://docs.python.org/"
                    + new_major_minor
                    + "/>`_",
                ),
            ],
        }

    # Process each file pattern
    for file_pattern, patterns in replacements.items():
        for file_path in Path().rglob(file_pattern):
            if not file_path.is_file():
                continue
            content = file_path.read_text()
            original_content = content
            for pattern, repl in patterns:
                if callable(repl):
                    content = re.sub(pattern, repl, content, flags=re.MULTILINE)
                else:
                    content = re.sub(pattern, repl, content, flags=re.MULTILINE)
            if content != original_content:
                if dry_run:
                    print(f"Would update {file_path}")
                else:
                    file_path.write_text(content)
                    print(f"Updated {file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to update Python versions in the codebase."
    )
    parser.add_argument(
        "new_full_version", help="New full Python version to use (e.g., 3.9.22)"
    )
    parser.add_argument(
        "--patch-only",
        action="store_true",
        help="Update only the patch version for matching major.minor versions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show changes without modifying files.",
    )
    args = parser.parse_args()

    _update_python_versions(
        new_full_version=args.new_full_version,
        patch_only=args.patch_only,
        dry_run=args.dry_run,
    )
