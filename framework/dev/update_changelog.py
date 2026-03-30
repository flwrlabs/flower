# mypy: ignore-errors
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
"""Update the changelog using PR titles."""


import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from sys import argv
from typing import Optional

import git
from git import Commit
from github import Github
from github.PullRequest import PullRequest
from github.Repository import Repository
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


REPO_NAME = "flwrlabs/flower"
ROOT_DIR = Path(__file__).parents[2]  # Path to the root of the repository
CHANGELOG_FILE = ROOT_DIR / "framework" / "docs" / "source" / "ref-changelog.md"
CHANGELOG_SECTION_HEADER = "### Changelog entry"

# Load the TOML configuration
with (ROOT_DIR / "dev" / "changelog_config.toml").open("rb") as toml_f:
    CONFIG = tomllib.load(toml_f)

# Extract types, project, and scope from the config
TYPES = "|".join(CONFIG["type"])
PROJECTS = "|".join(CONFIG["project"]) + "|\\*"
SCOPE = CONFIG["scope"]
ALLOWED_VERBS = CONFIG["allowed_verbs"]

# Construct the pattern
PATTERN_TEMPLATE = CONFIG["pattern_template"]
PATTERN = PATTERN_TEMPLATE.format(types=TYPES, projects=PROJECTS, scope=SCOPE)

# Local git repository
LOCAL_REPO = git.Repo(search_parent_directories=True)

# Map PR types to sections in the changelog
PR_TYPE_TO_SECTION = {
    "feat": "### New features",
    "docs": "### Documentation improvements",
    "break": "### Incompatible changes",
    "ci": "### Other changes",
    "fix": "### Other changes",
    "refactor": "### Other changes",
    "unknown": "### Unknown changes",
}
SKIPPED_CHANGELOG_PROJECTS = {"datasets", "hub", "intelligence"}

# Maximum number of workers in the thread pool
MAX_WORKERS = int(argv[2]) if len(argv) > 2 else 10
TAG_PATTERN = re.compile(r"(?:v|framework-)(?P<version>\d+\.\d+\.\d+)$")


@dataclass(frozen=True)
class ReleaseTag:
    """Structured representation of a release git tag and changelog version."""

    git_tag: str
    version: Version

    @property
    def changelog_version(self) -> str:
        """Return the version string used in changelog headings."""
        return f"v{self.version}"


def _parse_release_tag(tag: str) -> Optional[ReleaseTag]:
    """Parse a supported git release tag."""
    match = TAG_PATTERN.fullmatch(tag)
    if match is None:
        return None
    return ReleaseTag(git_tag=tag, version=Version(match.group("version")))


def _get_latest_release_tag(gh_api: Github) -> tuple[Repository, Optional[ReleaseTag]]:
    """Retrieve the latest supported git release tag from the repository."""
    repo = gh_api.get_repo(REPO_NAME)
    parsed_tags = [
        (tag_ref, release_tag)
        for tag_ref in LOCAL_REPO.tags
        if (release_tag := _parse_release_tag(tag_ref.name)) is not None
    ]
    if not parsed_tags:
        return repo, None
    _, latest_release_tag = max(
        parsed_tags, key=lambda item: item[0].commit.committed_datetime
    )
    return repo, latest_release_tag


def _add_shortlog(new_changelog_version: str, shortlog: str) -> None:
    """Update the markdown file with the new version or update existing logs."""
    token = f"<!---TOKEN_{new_changelog_version}-->"
    entry = (
        "\n### Thanks to our contributors\n\n"
        "We would like to give our special thanks to all the contributors "
        "who made the new version of Flower possible "
        f"(in `git shortlog` order):\n\n{shortlog} {token}"
    )
    current_date = date.today()

    with open(CHANGELOG_FILE, encoding="utf-8") as file:
        content = file.readlines()

    token_exists = any(token in line for line in content)
    shortlog_exists = True

    with open(CHANGELOG_FILE, "w", encoding="utf-8") as file:
        for line in content:
            if token in line:
                token_exists = True
                file.write(line)
            elif "## Unreleased" in line and not token_exists:
                # Add the new entry under "## Unreleased"
                file.write(f"## {new_changelog_version} ({current_date})\n{entry}\n")
                shortlog_exists = False
                token_exists = True
            else:
                file.write(line)

    if shortlog_exists:
        print("Shortlog already exists in the changelog, skipping addition.")
    else:
        print("Shortlog added to the changelog.")


def _git_commits_since_tag(git_tag: str) -> list[Commit]:
    """Get a set of commits since a given git tag."""
    return list(LOCAL_REPO.iter_commits(f"{git_tag}..origin/main"))


def _get_contributors_from_commits(api: Github, commits: list[Commit]) -> set[str]:
    """Get a set of contributors from a set of commits."""
    # Get authors and co-authors from the commits
    contributors: set[str] = set()
    coauthor_names_emails: set[tuple[str, str]] = set()
    coauthor_pattern = r"Co-authored-by:\s*(.+?)\s*<(.+?)>"

    for commit in commits:
        if commit.author.name is None:
            continue
        if "[bot]" in commit.author.name:
            continue
        # Find co-authors in the commit message
        matches: list[str] = re.findall(coauthor_pattern, commit.message)

        contributors.add(commit.author.name)
        if matches:
            coauthor_names_emails.update(matches)

    # Get full names of the GitHub usernames
    def _get_user(username: str, email: str) -> Optional[str]:
        try:
            user = api.get_user(username)
            if user.email == email:
                return user.name
            print(f"Email mismatch for user {username}: {email} != {user.email}")
        except Exception:  # pylint: disable=broad-exception-caught
            print(
                f"Failed to retrieve GitHub profile for user '{username}' <{email}>. "
                f"Using '{username}' directly in the changelog."
            )
            return username
        return None

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for name in executor.map(lambda x: _get_user(*x), coauthor_names_emails):
            if name:
                contributors.add(name)

    # Remove Copilot from the contributors list if present
    contributors.discard("Copilot")
    return contributors


def _get_pull_requests_since_tag(
    api: Github, repo: Repository, release_tag: ReleaseTag
) -> tuple[str, set[PullRequest]]:
    """Get a list of pull requests merged into the main branch since a given git tag."""
    prs = set()

    print(f"Retrieving commits since tag '{release_tag.git_tag}'...")
    commits = _git_commits_since_tag(release_tag.git_tag)

    print("Retrieving contributors...")
    contributors = _get_contributors_from_commits(api, commits)
    print(f"Found following contributors:\n{', '.join(sorted(contributors))}\n")

    print("Retrieving pull requests...")
    commit_shas = {commit.hexsha for commit in commits}
    for pr_info in repo.get_pulls(
        state="closed", sort="updated", direction="desc", base="main"
    ):
        if pr_info.merge_commit_sha in commit_shas:
            prs.add(pr_info)
        if len(prs) == len(commit_shas):
            break

    shortlog = ", ".join([f"`{name}`" for name in sorted(contributors)])
    return shortlog, prs


def _format_pr_reference(title: str, number: int, url: str) -> str:
    """Format a pull request reference as a markdown list item."""
    parts = title.strip().replace("*", "").split("`")
    formatted_parts = []

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Even index parts are normal text, ensure we do not add extra bold if empty
            if part.strip():
                formatted_parts.append(f"**{part.strip()}**")
            else:
                formatted_parts.append("")
        else:
            # Odd index parts are inline code
            formatted_parts.append(f"`{part.strip()}`")

    # Join parts with spaces but avoid extra spaces
    formatted_title = " ".join(filter(None, formatted_parts))
    return f"- {formatted_title} ([#{number}]({url}))"


def _extract_changelog_entry(
    pr_info: PullRequest,
) -> dict[str, str]:
    """Extract the changelog entry from a pull request's body."""
    # Use regex search to find matches
    match = re.search(PATTERN, pr_info.title)

    # Extract the topic label
    topic = ""
    for label in pr_info.labels:
        if label.name not in ["Maintainer", "Contributor", "Bot", "General"]:
            topic = label.name
            break

    if match:
        # Extract components from the regex groups
        pr_type = match.group(1)
        pr_project = match.group(2)
        pr_scope = match.group(3)  # Correctly capture optional sub-scope
        pr_subject = match.group(
            4
        )  # Capture subject starting with uppercase and no terminal period
        return {
            "type": pr_type,
            "project": pr_project,
            "scope": pr_scope,
            "subject": pr_subject,
            "topic": topic,
        }

    return {
        "type": "unknown",
        "project": "unknown",
        "scope": "unknown",
        "subject": "unknown",
    }


def _update_changelog(
    prs: set[PullRequest],
    previous_changelog_version: str,
    new_changelog_version: Optional[str],
) -> bool:
    """Update the changelog file with entries from provided pull requests."""
    with open(CHANGELOG_FILE, "r+", encoding="utf-8") as file:
        content = file.read()
        unreleased_index = content.find(
            "\n## Unreleased\n"
        )  # Avoid finding `## Unreleased` in other text
        if unreleased_index == -1 and new_changelog_version:
            unreleased_index = content.find(
                f"## {new_changelog_version}"
            )  # Try to find the new tag if Unreleased not found
        elif unreleased_index != -1:
            unreleased_index += 1  # Skip the newline (\n) character

        if unreleased_index == -1:
            print("Unreleased header not found in the changelog.")
            return False

        # Find the end of the Unreleased section
        end_index = content.find(
            f"## {previous_changelog_version}", unreleased_index + 1
        )
        if end_index == -1:
            print(
                "Previous changelog version "
                f"'{previous_changelog_version}' not found in the changelog."
            )
            return False

        for section in PR_TYPE_TO_SECTION.values():
            if content.find(section, unreleased_index, end_index) == -1:
                content = content[:end_index] + f"\n{section}\n\n" + content[end_index:]
                end_index = content.find(f"## {previous_changelog_version}", end_index)
        topic_to_section = {}

        prs_sorted = sorted(
            prs,
            key=lambda pr: (
                pr.merged_at or pr.updated_at or pr.closed_at,
                pr.number,
            ),
        )

        for pr_info in prs_sorted:
            parsed_title = _extract_changelog_entry(pr_info)

            # Skip if the PR is already in changelog
            if f"#{pr_info.number}]" in content:
                continue

            # Skip PRs for non-framework projects.
            if parsed_title["project"] in SKIPPED_CHANGELOG_PROJECTS:
                continue

            # Create the topic if not found
            if (topic := parsed_title.get("topic")) and topic not in topic_to_section:
                section = f"### {topic}"
                if content.find(section, unreleased_index, end_index) == -1:
                    content = (
                        content[:end_index] + f"\n{section}\n\n" + content[end_index:]
                    )
                    end_index = content.find(
                        f"## {previous_changelog_version}", end_index
                    )
                topic_to_section[topic] = section

            # Find section to insert
            pr_type = parsed_title.get("type", "unknown")
            if topic:
                section = topic_to_section[topic]
            else:
                section = PR_TYPE_TO_SECTION.get(pr_type, "### Unknown changes")
            insert_index = content.find(section, unreleased_index, end_index)

            pr_reference = _format_pr_reference(
                pr_info.title, pr_info.number, pr_info.html_url
            )
            content = _insert_entry_no_desc(
                content,
                pr_reference,
                insert_index,
            )

            # Find the end of the Unreleased section
            end_index = content.find(f"## {previous_changelog_version}", end_index)

        # Check for repeated PRs
        _check_repeated_prs(content)

        # Finalize content update
        file.seek(0)
        file.write(content)
        file.truncate()
    return True


def _check_repeated_prs(content: str) -> None:
    """Check for repeated PRs in the changelog."""
    found_pairs = re.findall(
        r"\[#(\d+)\]\(https://github.com/flwrlabs/flower/pull/(\d+)\)", content
    )

    count_prs = {}
    for pr, pr_http in found_pairs:
        if pr_http != pr:
            print(f"PR #{pr} has inconsistent http link.")
        if pr not in count_prs:
            count_prs[pr] = 0
        count_prs[pr] += 1
    for pr, count in count_prs.items():
        if count > 1:
            print(f"PR #{pr} is repeated {count} times.")


def _insert_entry_no_desc(
    content: str, pr_reference: str, unreleased_index: int
) -> str:
    """Insert a changelog entry for a pull request with no specific description."""
    insert_index = content.find("\n", unreleased_index) + 1
    content = (
        content[:insert_index] + "\n" + pr_reference + "\n" + content[insert_index:]
    )
    return content


def _fetch_origin() -> None:
    """Fetch the latest changes from the origin."""
    LOCAL_REPO.remote("origin").fetch()


def main() -> None:
    """Update changelog using the descriptions of PRs since the latest tag."""
    start = time.time()

    # Initialize GitHub Client with provided token (as argument)
    gh_api = Github(argv[1])

    # Fetch the latest changes from the origin
    print("Fetching the latest changes from the origin...")
    _fetch_origin()

    # Get the repository and the latest tag
    print("Retrieving the latest tag...")
    repo, latest_release_tag = _get_latest_release_tag(gh_api)
    if not latest_release_tag:
        print("No tags found in the repository.")
        return

    # Get the shortlog and the pull requests since the latest tag
    shortlog, prs = _get_pull_requests_since_tag(gh_api, repo, latest_release_tag)

    # Update the changelog
    print("Updating the changelog...")
    new_changelog_version = (
        f"v{latest_release_tag.version.major}.{latest_release_tag.version.minor + 1}.0"
    )
    if _update_changelog(
        prs,
        latest_release_tag.changelog_version,
        new_changelog_version,
    ):
        _add_shortlog(new_changelog_version, shortlog)
        print(f"Changelog updated successfully in {time.time() - start:.2f}s.")


if __name__ == "__main__":
    main()
