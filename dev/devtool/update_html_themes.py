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
"""Update HTML themes in Flower docs."""


import json
import re
from pathlib import Path
from typing import Optional, Union

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_DIR = REPO_ROOT

# Define new fields to be added to the `html_theme_options` dictionary in `conf.py`.
# If no fields are needed, set to an empty dictionary.
NEW_FIELDS: dict[str, Optional[Union[dict[str, str], str]]] = {
    "light_css_variables": {
        "color-announcement-background": "#292f36",
        "color-announcement-text": "#ffffff",
    },
    "dark_css_variables": {
        "color-announcement-background": "#292f36",
        "color-announcement-text": "#ffffff",
    },
}

with (REPO_ROOT / "dev" / "docs-ui-config.yml").open(encoding="utf-8") as f:
    announcement = yaml.safe_load(f)["announcement"]
    if announcement["enabled"]:
        NEW_FIELDS["announcement"] = announcement["html"]


def dict_to_fields_str(fields: dict[str, Optional[Union[dict[str, str], str]]]) -> str:
    """
    Convert a dictionary to a formatted string suitable for insertion
    into a Python dictionary literal (without the outer braces).
    """
    if not fields:
        return ""
    # Use json.dumps for a clean, indented format with double quotes.
    s = json.dumps(fields, indent=4, ensure_ascii=False)
    s_lines = s.splitlines()
    # Remove the first and last lines (the outer braces).
    if len(s_lines) >= 2 and s_lines[0].strip() == "{" and s_lines[-1].strip() == "}":
        s = "\n".join(s_lines[1:-1])
    return s


def find_conf_files(root_dir: Path) -> list[Path]:
    """Recursively find all conf.py files under the given directory."""
    return list(root_dir.rglob("conf.py"))


def update_conf_file(file_path: Path, new_fields_str: str) -> None:
    """
    Insert new_fields_str into the html_theme_options block of file_path.
    The new fields are inserted just before the closing brace.
    """
    if not new_fields_str.strip():
        print(f"Skipping {file_path} (no new fields to insert)")
        return

    content = file_path.read_text(encoding="utf-8").splitlines()
    updated_content = []
    inside_options = False
    modified = False

    for line in content:
        updated_content.append(line)
        # Look for the start of html_theme_options.
        if re.match(r"^\s*html_theme_options\s*=\s*{", line):
            inside_options = True
        # When inside html_theme_options, insert new fields before the closing brace.
        if inside_options and re.match(r"^\s*}", line):
            # Determine the indentation from the closing brace line.
            indent_match = re.match(r"^(\s*)}", line)
            indent = indent_match.group(1) if indent_match else ""
            # Indent each line of the new fields.
            new_fields_indented = "\n".join(
                indent + "    " + l for l in new_fields_str.splitlines()
            )
            # Insert before the closing brace.
            updated_content.insert(-1, new_fields_indented + ",")
            inside_options = False
            modified = True

    if modified:
        file_path.write_text("\n".join(updated_content) + "\n", encoding="utf-8")
        print(f"Updated: {file_path}")
    else:
        print(f"No html_theme_options block found in: {file_path}")


def main() -> None:
    """."""
    new_fields_str = dict_to_fields_str(NEW_FIELDS)
    conf_files = find_conf_files(ROOT_DIR)
    if not conf_files:
        print("No conf.py files found.")
        return

    for conf_file in conf_files:
        if "framework/docs/source/conf.py" in str(conf_file):
            continue  # Skip updating conf.py for framework docs
        update_conf_file(conf_file, new_fields_str)


if __name__ == "__main__":
    main()
