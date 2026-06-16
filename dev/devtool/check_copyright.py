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
"""Check if copyright notices are present in all Python files.

Example:
    python -m devtool.check_copyright framework/py/flwr
"""


import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from devtool.init_py_check import get_init_dir_list_and_warnings

COPYRIGHT_FORMAT = """# Copyright {} Flower Labs GmbH. All Rights Reserved.
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
# =============================================================================="""

# Regex pattern to extract copyright year from file
COPYRIGHT_PATTERN = re.compile(r"# Copyright (\d{4}) Flower Labs GmbH")


def _get_file_creation_year(filepath: str) -> int:
    result = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%ai", "--", filepath],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )

    if not result.stdout:
        # Since the file is not in Git history, use the current year
        return datetime.now().year

    date_str = result.stdout.splitlines()[-1]  # Get the first commit date
    creation_year_str = date_str.split("-")[0]  # Extract the year
    return int(creation_year_str)


def _check_copyright(dir_list: list[str]) -> None:
    warning_list = []
    for valid_dir in dir_list:
        if "proto" in valid_dir:
            continue

        dir_path = Path(valid_dir)
        for py_file in dir_path.glob("*.py"):
            creation_year = _get_file_creation_year(str(py_file.absolute()))

            # Extract copyright year from file content using regex
            file_content = py_file.read_text()
            match = COPYRIGHT_PATTERN.search(file_content)
            copyright_year = int(match.group(1)) if match else None

            # Allow copyright year to be creation year or one year before
            if copyright_year in [creation_year, creation_year - 1]:
                continue

            # Warn if no copyright notice found or incorrect year
            warning_message = "- " + str(py_file)
            warning_list.append(warning_message)

    if len(warning_list) > 0:
        print("Missing or incorrect copyright notice in the following files:")
        for warning in warning_list:
            print(warning)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide at least one directory path relative "
            "to your current working directory."
        )
    for relative_path in sys.argv[1:]:
        abs_path = os.path.abspath(os.path.join(os.getcwd(), relative_path))
        _, init_dirs = get_init_dir_list_and_warnings(abs_path)
        _check_copyright(init_dirs)
