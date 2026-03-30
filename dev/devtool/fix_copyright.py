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
"""Fix copyright notices in all Python files of a given directory.

Example:
    python -m devtool.fix_copyright framework/py/flwr
"""


import os
import sys
from pathlib import Path

from devtool.check_copyright import COPYRIGHT_FORMAT, _get_file_creation_year
from devtool.init_py_check import get_init_dir_list_and_warnings


def _insert_or_edit_copyright(py_file: Path) -> None:
    contents = py_file.read_text()
    lines = contents.splitlines()
    creation_year = _get_file_creation_year(str(py_file.absolute()))
    expected_copyright = COPYRIGHT_FORMAT.format(creation_year)

    if expected_copyright not in contents:
        if "Copyright" in lines[0]:
            end_index = 0
            for idx, line in enumerate(lines):
                if (
                    line.strip()
                    == COPYRIGHT_FORMAT.rsplit("\n", maxsplit=1)[-1].strip()
                ):
                    end_index = idx + 1
                    break
            lines = lines[end_index:]

        lines.insert(0, expected_copyright)
        py_file.write_text("\n".join(lines) + "\n")


def _fix_copyright(dir_list: list[str]) -> None:
    for valid_dir in dir_list:
        if "proto" in valid_dir:
            continue

        dir_path = Path(valid_dir)
        for py_file in dir_path.glob("*.py"):
            _insert_or_edit_copyright(py_file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide at least one directory path relative "
            "to your current working directory."
        )
    for relative_path in sys.argv[1:]:
        abs_path = os.path.abspath(os.path.join(os.getcwd(), relative_path))
        _, init_dirs = get_init_dir_list_and_warnings(abs_path)
        _fix_copyright(init_dirs)
