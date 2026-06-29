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
"""Tests for exit codes."""


from pathlib import Path

from .exit_code import EXIT_CODE_HELP, ExitCode


def test_exit_code_help_exist() -> None:
    """Test if all exit codes have help message."""
    for name, code in ExitCode.__dict__.items():
        if name.startswith("__"):
            continue
        assert (
            code in EXIT_CODE_HELP
        ), f"Exit code {name} ({code}) does not have help message."


def test_exit_code_help_url_exist() -> None:
    """Test if all exit codes have help URL."""
    # Get all exit code help URLs
    dir_path = Path(__file__).parents[4] / "docs/source/ref-exit-codes"
    files = {int(f.stem): f for f in dir_path.glob("*.rst") if f.stem.isdigit()}

    # Check if all exit codes
    for name, code in ExitCode.__dict__.items():
        if name.startswith("__"):
            continue

        # Assert file exists
        assert code in files, f"Exit code {name} ({code}) does not have help URL."

        # Retrieve the title from the help URL
        f = files[code]
        title = f.read_text().split("\n")[1]

        # Assert the title is correct
        assert (
            title == f" [{code}] {name}"
        ), f"Exit code {name} ({code}) help URL has incorrect title in {str(f)}"
