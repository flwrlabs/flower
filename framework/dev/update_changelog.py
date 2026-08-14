# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Update a Framework release changelog using PR titles."""

import argparse
import json
import re
import subprocess
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import git
from git import Commit


REPO_NAME = "flwrlabs/flower"
MAX_WORKERS = 8

ROOT_DIR = Path(__file__).parents[2]

CHANGELOG_DIR = ROOT_DIR / "framework" / "docs" / "source" / "changelog"
CHANGELOG_INDEX = CHANGELOG_DIR / "index.md"

CACHE_DIR = ROOT_DIR / ".cache" / "update_changelog"
PR_CACHE_DIR = CACHE_DIR / "prs"
STATE_FILE = CACHE_DIR / "state.json"

with (ROOT_DIR / "dev" / "changelog_config.toml").open("rb") as toml_f:
    CONFIG = tomllib.load(toml_f)

TYPES = "|".join(CONFIG["type"])
PROJECTS = "|".join(CONFIG["project"]) + "|\\*"
SCOPE = CONFIG["scope"]
PATTERN = CONFIG["pattern_template"].format(types=TYPES, projects=PROJECTS, scope=SCOPE)

LOCAL_REPO = git.Repo(ROOT_DIR)

PR_TYPE_TO_SECTION = {
    "feat": "### New features",
    "docs": "### Documentation improvements",
    "break": "### Incompatible changes",
    "ci": "### Other changes",
    "fix": "### Other changes",
    "refactor": "### Other changes",
    "unknown": "### Unknown changes",
}

SKIPPED_CHANGELOG_PROJECTS = {
    "datasets",
    "hub",
    "intelligence",
}

IGNORED_TOPIC_LABELS = {"Maintainer", "Contributor", "Bot", "General"}
PR_NUMBER_PATTERN = re.compile(r"\(#(\d+)\)$")
COAUTHOR_PATTERN = re.compile(
    r"^Co-authored-by:\s*(.+?)\s*<[^>]+>\s*$", re.IGNORECASE | re.MULTILINE
)


def _get_previous_release_tag(source_sha: str) -> str:
    """Return the latest Framework release tag on the source's mainline history."""
    return cast(
        str,
        LOCAL_REPO.git.describe(
            "--first-parent",
            "--tags",
            "--abbrev=0",
            "--match=framework-*",
            source_sha,
        ),
    ).strip()


def _get_commits(previous_release_tag: str, source_sha: str) -> list[Commit]:
    """Return commits in the release range, ordered from oldest to newest."""
    commits = list(LOCAL_REPO.iter_commits(f"{previous_release_tag}..{source_sha}"))
    commits.reverse()
    return commits


def _get_pr_numbers(commits: list[Commit]) -> list[int]:
    """Extract unique PR numbers from squash commit summaries."""
    pr_numbers: list[int] = []
    seen: set[int] = set()
    for commit in commits:
        match = PR_NUMBER_PATTERN.search(cast(str, commit.summary))
        if match:
            number = int(match.group(1))
            if number not in seen:
                seen.add(number)
                pr_numbers.append(number)
    return pr_numbers


def _is_bot(name: str) -> bool:
    """Return whether a contributor name belongs to a bot."""
    return "[bot]" in name.casefold() or name.casefold() == "copilot"


def _get_contributors(commits: list[Commit]) -> list[str]:
    """Collect commit authors and co-authors in deterministic shortlog order."""
    contributors: set[str] = set()
    for commit in commits:
        if commit.author.name and not _is_bot(commit.author.name):
            contributors.add(commit.author.name)
        contributors.update(
            name.strip()
            for name in COAUTHOR_PATTERN.findall(cast(str, commit.message))
            if not _is_bot(name.strip())
        )
    return sorted(contributors)


def _fetch_pr(number: int) -> None:
    """Cache GitHub metadata and the full diff for a PR."""
    PR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metadata_file = PR_CACHE_DIR / f"{number}.json"
    diff_file = PR_CACHE_DIR / f"{number}.diff"

    if metadata_file.exists() and diff_file.exists():
        return

    metadata = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            str(number),
            "--repo",
            REPO_NAME,
            "--json",
            "number,title,body,url,author,labels,mergedAt",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    diff = subprocess.run(
        ["gh", "pr", "diff", str(number), "--repo", REPO_NAME],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    metadata_file.write_text(metadata, encoding="utf-8")
    diff_file.write_text(diff, encoding="utf-8")


def _load_pr(number: int) -> dict[str, Any]:
    """Load cached PR metadata."""
    with (PR_CACHE_DIR / f"{number}.json").open(encoding="utf-8") as cache_file:
        return cast(dict[str, Any], json.load(cache_file))


def _parse_pr_title(pr: dict[str, Any]) -> dict[str, str]:
    """Extract changelog grouping information from a pull request."""
    topic = next(
        (
            label["name"]
            for label in pr["labels"]
            if label["name"] not in IGNORED_TOPIC_LABELS
        ),
        "",
    )
    match = re.search(PATTERN, pr["title"])
    if match:
        return {
            "type": match.group(1),
            "project": match.group(2),
            "scope": match.group(3) or "",
            "subject": match.group(4),
            "topic": topic,
        }
    return {
        "type": "unknown",
        "project": "unknown",
        "scope": "unknown",
        "subject": "unknown",
        "topic": topic,
    }


def _format_pr_reference(pr: dict[str, Any]) -> str:
    """Format a pull request reference as a Markdown list item."""
    parts = pr["title"].strip().replace("*", "").split("`")
    formatted_parts = []
    for index, part in enumerate(parts):
        if not part.strip():
            continue
        if index % 2 == 0:
            formatted_parts.append(f"**{part.strip()}**")
        else:
            formatted_parts.append(f"`{part.strip()}`")
    formatted_title = " ".join(formatted_parts)
    return f"- {formatted_title} ([#{pr['number']}]({pr['url']}))"


def _contributor_entry(contributors: list[str]) -> str:
    """Build the contributor section for a release changelog."""
    shortlog = ", ".join(f"`{name}`" for name in contributors)
    return (
        "### Thanks to our contributors\n\n"
        "We would like to give our special thanks to all the contributors who made "
        "the new version of Flower possible (in `git shortlog` order):\n\n"
        f"{shortlog}"
    )


def _refresh_contributors(content: str, contributors: list[str]) -> str:
    """Replace the contributor section in a release changelog."""
    section = "### Thanks to our contributors"
    section_start = content.index(section)
    section_end = content.find("\n### ", section_start + len(section))
    if section_end == -1:
        section_end = len(content)
    return (
        f"{content[:section_start]}"
        f"{_contributor_entry(contributors)}\n"
        f"{content[section_end:]}"
    )


def _section_for_pr(parsed_title: dict[str, str]) -> str:
    """Return the changelog section for parsed PR metadata."""
    if parsed_title["topic"]:
        return f"### {parsed_title['topic']}"
    return PR_TYPE_TO_SECTION.get(parsed_title["type"], PR_TYPE_TO_SECTION["unknown"])


def _entry_sort_key(pr: dict[str, Any]) -> tuple[datetime, int]:
    """Return a stable chronological sort key for a PR."""
    merged_at = pr.get("mergedAt")
    merged = (
        datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
        if merged_at
        else datetime.min.replace(tzinfo=UTC)
    )
    return merged, pr["number"]


def _append_entry(content: str, section: str, entry: str) -> str:
    """Add an entry to a section, creating the section when necessary."""
    if section not in content:
        return f"{content.rstrip()}\n\n{section}\n\n{entry}\n"

    insert_at = content.index("\n", content.index(section)) + 1
    return f"{content[:insert_at]}\n{entry}\n{content[insert_at:]}"


def _update_release_file(
    version: str, contributors: list[str], prs: list[dict[str, Any]]
) -> None:
    """Create a release changelog or add uncovered PRs to an existing one."""
    release_file = CHANGELOG_DIR / f"v{version}.md"
    if release_file.exists():
        content = _refresh_contributors(
            release_file.read_text(encoding="utf-8"), contributors
        )
    else:
        sections = "\n\n".join(dict.fromkeys(PR_TYPE_TO_SECTION.values()))
        content = (
            f"## v{version} ({date.today()})\n\n"
            f"{_contributor_entry(contributors)}\n\n"
            f"{sections}\n"
        )

    for pr in sorted(prs, key=_entry_sort_key):
        if f"#{pr['number']}]" in content:
            continue
        parsed_title = _parse_pr_title(pr)
        if parsed_title["project"] in SKIPPED_CHANGELOG_PROJECTS:
            continue
        content = _append_entry(
            content, _section_for_pr(parsed_title), _format_pr_reference(pr)
        )

    release_file.write_text(f"{content.rstrip()}\n", encoding="utf-8")


def _update_index(version: str) -> None:
    """Include a release changelog at the top of the changelog index."""
    include = f"```{{include}} v{version}.md\n```"
    content = CHANGELOG_INDEX.read_text(encoding="utf-8")
    if include in content:
        return
    heading_end = content.index("\n", content.index("# Changelog")) + 1
    content = f"{content[:heading_end]}\n{include}\n{content[heading_end:]}"
    CHANGELOG_INDEX.write_text(content, encoding="utf-8")


def _load_previous_pr_numbers() -> set[int]:
    """Load PR numbers from the previous state, if present."""
    if not STATE_FILE.exists():
        return set()
    with STATE_FILE.open(encoding="utf-8") as state_file:
        return set(json.load(state_file)["pr_numbers"])


def _write_state(
    version: str,
    source_sha: str,
    previous_release_tag: str,
    pr_numbers: list[int],
    previous_pr_numbers: set[int],
) -> None:
    """Write the current release range state."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    state = {
        "version": version,
        "source_sha": source_sha,
        "previous_release_tag": previous_release_tag,
        "pr_numbers": pr_numbers,
        "new_pr_numbers": [
            number for number in pr_numbers if number not in previous_pr_numbers
        ],
    }
    STATE_FILE.write_text(f"{json.dumps(state, indent=2)}\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Target release version")
    parser.add_argument(
        "--source-sha", required=True, help="Exact release source commit SHA"
    )
    return parser.parse_args()


def main() -> None:
    """Update a release changelog for an exact source commit."""
    args = _parse_args()

    print("Fetching origin and tags...")
    LOCAL_REPO.git.fetch("origin", "--tags")

    previous_release_tag = _get_previous_release_tag(args.source_sha)
    commits = _get_commits(previous_release_tag, args.source_sha)
    pr_numbers = _get_pr_numbers(commits)
    contributors = _get_contributors(commits)
    previous_pr_numbers = _load_previous_pr_numbers()

    print(f"Caching {len(pr_numbers)} pull requests...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(_fetch_pr, pr_numbers))
    prs = [_load_pr(number) for number in pr_numbers]

    _write_state(
        args.version,
        args.source_sha,
        previous_release_tag,
        pr_numbers,
        previous_pr_numbers,
    )
    _update_release_file(args.version, contributors, prs)
    _update_index(args.version)
    print(f"Updated changelog for v{args.version}.")


if __name__ == "__main__":
    main()
