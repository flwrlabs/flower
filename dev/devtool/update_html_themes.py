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
MERGE_THEME_VARIABLE_FIELDS = {"light_css_variables", "dark_css_variables"}

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
        s = "\n".join(
            line[4:] if line.startswith("    ") else line for line in s_lines[1:-1]
        )
    return s


def find_conf_files(root_dir: Path) -> list[Path]:
    """Recursively find all conf.py files under the given directory."""
    return list(root_dir.rglob("conf.py"))


def _dict_entry_str(key: str, value: str) -> str:
    """Convert one dictionary entry to a formatted string."""
    return f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)},"


def _brace_delta(line: str) -> int:
    """Return the dictionary brace depth change for a line."""
    return line.count("{") - line.count("}")


def _copy_fields(
    fields: dict[str, Optional[Union[dict[str, str], str]]],
) -> dict[str, Optional[Union[dict[str, str], str]]]:
    """Copy generated fields so per-file updates do not mutate NEW_FIELDS."""
    return {
        key: value.copy() if isinstance(value, dict) else value
        for key, value in fields.items()
    }


def _merge_fields(
    fields: dict[str, Optional[Union[dict[str, str], str]]],
) -> dict[str, dict[str, str]]:
    """Return generated theme variable dictionaries that can be merged."""
    return {
        key: value.copy()
        for key, value in fields.items()
        if key in MERGE_THEME_VARIABLE_FIELDS and isinstance(value, dict)
    }


def _append_indented_fields(
    updated_content: list[str],
    line: str,
    fields: dict[str, Optional[Union[dict[str, str], str]]],
) -> bool:
    """Insert generated fields before the html_theme_options closing brace."""
    indent_match = re.match(r"^(\s*)}", line)
    indent = indent_match.group(1) if indent_match else ""
    new_fields_str = dict_to_fields_str(fields)
    new_fields_indented = "\n".join(
        indent + "    " + new_field_line
        for new_field_line in new_fields_str.splitlines()
    )
    if new_fields_indented:
        updated_content.insert(-1, new_fields_indented + ",")
    return bool(new_fields_indented)


def _split_inline_comment(line: str) -> tuple[str, str]:
    """Split off a Python comment while ignoring # inside strings."""
    in_string = False
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
        elif char in {"'", '"'}:
            in_string = True
            quote = char
        elif char == "#":
            return line[:index], line[index:]
    return line, ""


def _ensure_previous_entry_has_comma(updated_content: list[str]) -> None:
    """Ensure new merged dictionary entries do not create invalid Python."""
    for index in range(len(updated_content) - 1, -1, -1):
        stripped = updated_content[index].strip()
        if not stripped or stripped.startswith("#"):
            continue
        code, comment = _split_inline_comment(updated_content[index])
        if code.rstrip().endswith((",", "{")):
            return
        updated_content[index] = code.rstrip() + (f", {comment}" if comment else ",")
        return


def _process_merge_field_line(
    updated_content: list[str],
    line: str,
    brace_depth: int,
    merge_field: str,
    merge_fields: dict[str, dict[str, str]],
) -> tuple[int, Optional[str], bool]:
    """Update one line while scanning an existing theme variable dictionary."""
    next_merge_field: Optional[str] = merge_field
    variable_match = re.match(r'^(\s*)"([^"]+)"\s*:', line)
    merge_variables = merge_fields[merge_field]
    if variable_match:
        variable_name = variable_match.group(2)
        if variable_name in merge_variables:
            updated_content.append(line)
            del merge_variables[variable_name]
            return brace_depth, next_merge_field, False

    brace_depth += _brace_delta(line)
    if brace_depth == 1:
        indent_match = re.match(r"^(\s*)}", line)
        indent = indent_match.group(1) if indent_match else ""
        if merge_variables:
            _ensure_previous_entry_has_comma(updated_content)
        for key, value in merge_variables.items():
            updated_content.append(indent + "    " + _dict_entry_str(key, value))
        next_merge_field = None

    updated_content.append(line)
    return brace_depth, next_merge_field, bool(merge_variables)


def update_conf_file(
    file_path: Path, new_fields: dict[str, Optional[Union[dict[str, str], str]]]
) -> None:
    """
    Insert new_fields into the html_theme_options block of file_path.
    Theme variable dictionaries are merged when they already exist.
    """
    if not dict_to_fields_str(new_fields).strip():
        print(f"Skipping {file_path} (no new fields to insert)")
        return

    updated_content: list[str] = []
    fields_to_append = _copy_fields(new_fields)
    merge_fields = _merge_fields(new_fields)
    inside_options = False
    brace_depth = 0
    merge_field: Optional[str] = None
    found_options = False
    modified = False

    for line in file_path.read_text(encoding="utf-8").splitlines():
        if merge_field:
            brace_depth, merge_field, line_modified = _process_merge_field_line(
                updated_content, line, brace_depth, merge_field, merge_fields
            )
            modified |= line_modified
            continue

        updated_content.append(line)
        # Look for the start of html_theme_options.
        if re.match(r"^\s*html_theme_options\s*=\s*{", line):
            inside_options = True
            found_options = True
        top_level_match = re.match(r'^\s*"(?P<key>[^"]+)"\s*:', line)
        if inside_options and brace_depth == 1 and top_level_match:
            theme_variable_key = top_level_match.group("key")
            fields_to_append.pop(theme_variable_key, None)
            if theme_variable_key in merge_fields:
                variable_match = re.match(
                    r'^\s*"(?P<key>light_css_variables|dark_css_variables)"\s*:\s*{',
                    line,
                )
                if variable_match:
                    merge_field = theme_variable_key
        if inside_options:
            brace_depth += _brace_delta(line)
        # When inside html_theme_options, insert new fields before the closing brace.
        if inside_options and brace_depth == 0:
            modified |= _append_indented_fields(updated_content, line, fields_to_append)
            inside_options = False

    if modified:
        file_path.write_text("\n".join(updated_content) + "\n", encoding="utf-8")
        print(f"Updated: {file_path}")
    elif found_options:
        print(f"No changes needed in: {file_path}")
    else:
        print(f"No html_theme_options block found in: {file_path}")


def main() -> None:
    """."""
    conf_files = find_conf_files(ROOT_DIR)
    if not conf_files:
        print("No conf.py files found.")
        return

    for conf_file in conf_files:
        if "framework/docs/source/conf.py" in str(conf_file):
            continue  # Skip updating conf.py for framework docs
        update_conf_file(conf_file, NEW_FIELDS.copy())


if __name__ == "__main__":
    main()
