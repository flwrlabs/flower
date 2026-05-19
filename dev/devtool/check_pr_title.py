# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Used to check a given PR title format."""

import re
import sys
from pathlib import Path
from typing import Any

_toml: Any

try:
    import tomllib

    _toml = tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli

    _toml = tomli


def _load_config() -> dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "changelog_config.toml"
    with config_path.open("rb") as file:
        config: dict[str, Any] = _toml.load(file)
        return config


def _validate_title(pr_title: str, config: dict[str, Any]) -> tuple[bool, str]:
    # Extract types, project, and scope from the config
    types = "|".join(config["type"])
    projects = "|".join(config["project"]) + "|\\*"
    scope = config["scope"]
    allowed_verbs = config["allowed_verbs"]

    # Construct the pattern
    pattern_template = config["pattern_template"]
    pattern = pattern_template.format(types=types, projects=projects, scope=scope)

    # Check title against pattern.
    match = re.search(pattern, pr_title)

    valid = True
    error = "it doesn't have the correct format"

    # This check is there to ignore dependabot PRs from title checks
    if pr_title.startswith("build"):
        return True, ""

    if not match:
        valid = False
    else:
        if match.group(4).split()[0] not in allowed_verbs:
            valid = False
            error = "the <PR_SUBJECT> doesn't start with a verb in the imperative mood"
        elif match.group(2) == "*" and match.group(3) is None:
            valid = False
            error = "the <PR_PROJECT> cannot be '*' without using the ':skip' flag"
    return valid, error


def main() -> None:
    """Validate a pull request title from the command line."""
    if len(sys.argv) < 2:
        raise ValueError("Usage: python -m devtool.check_pr_title '<title>'")

    pr_title = sys.argv[1]
    config = _load_config()
    valid, error = _validate_title(pr_title, config)

    if not valid:
        types = "|".join(config["type"])
        print(
            f"PR title `{pr_title}` is invalid, {error}.\n\nA PR title should "
            "be of the form:\n\n\t<PR_TYPE>(<PR_PROJECT>): <PR_SUBJECT>\n\n"
            f"Or, if the PR shouldn't appear in the changelog:\n\n\t<PR_TYPE>"
            f"(<PR_PROJECT>:skip): <PR_SUBJECT>\n\nwith <PR_TYPE> in [{types}],\n"
            f"<PR_PROJECT> in [{'|'.join(config['project']) + '|*'}] "
            "(where '*' is used when modifying multiple projects and should be used in "
            "conjunction with the ':skip' flag),\nand <PR_SUBJECT> starting with "
            "a capitalized verb in the imperative mood and without any punctuation "
            "at the end.\n\nA valid example is:\n\n\t`feat(framework): "
            "Add flwr build CLI command`\n\nOr, if the PR shouldn't appear in "
            "the changelog:\n\n\t`feat(framework:skip): Add new option to build CLI`\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
