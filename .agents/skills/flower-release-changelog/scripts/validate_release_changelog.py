#!/usr/bin/env python3
"""Validate PR inventory, heading cleanup, link ordering, and incompatible-section placement in a Flower release changelog."""

import argparse
import re
from collections import Counter
from pathlib import Path


ALLOWED_SECTIONS = {
    "Thanks to our contributors",
    "What's new?",
    "Incompatible changes",
}
SECTION_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
PR_RE = re.compile(
    r"\[#\d+\]\(https://github\.com/flwrlabs/flower/pull/(\d+)/?\)"
)
ITEM_RE = re.compile(r"^- \*\*.+?\*\* \(([^\n]+)\)\s*$", re.MULTILINE)


def parse_prs(values: list[str]) -> set[int]:
    """Parse space- or comma-separated PR numbers."""
    parts = (
        part.strip().removeprefix("#")
        for value in values
        for part in value.split(",")
    )
    try:
        return {int(part) for part in parts if part}
    except ValueError as exc:
        raise SystemExit(f"Invalid PR number: {exc}") from exc


def section_body(text: str, name: str) -> str:
    """Return one level-three section body, or an empty string when absent."""
    match = re.search(rf"^###\s+{re.escape(name)}\s*$", text, re.MULTILINE)
    if not match:
        return ""
    next_section = SECTION_RE.search(text, match.end())
    return text[match.end() : next_section.start() if next_section else None]


def validate(text: str, expected: set[int], incompatible: set[int]) -> list[str]:
    """Return inventory, placement, and generated-heading errors."""
    errors: list[str] = []
    headings = SECTION_RE.findall(text)
    generated = [heading for heading in headings if heading not in ALLOWED_SECTIONS]
    if generated:
        errors.append("generated headings remain: " + ", ".join(generated))

    actual = [int(number) for number in PR_RE.findall(text)]
    counts = Counter(actual)
    missing = expected - set(actual)
    extra = set(actual) - expected
    duplicates = {number for number, count in counts.items() if count > 1}
    if missing:
        errors.append("missing PRs: " + format_prs(missing))
    if extra:
        errors.append("unexpected PRs: " + format_prs(extra))
    if duplicates:
        errors.append("duplicate PRs: " + format_prs(duplicates))

    for links in ITEM_RE.findall(text):
        item_prs = [int(number) for number in PR_RE.findall(links)]
        if item_prs != sorted(item_prs):
            found = ", ".join(f"#{number}" for number in item_prs)
            errors.append("PRs are not ascending: " + found)

    incompatible_body = section_body(text, "Incompatible changes")
    actual_incompatible = {int(number) for number in PR_RE.findall(incompatible_body)}
    if actual_incompatible != incompatible:
        errors.append(
            "incompatible PRs differ: expected "
            f"{format_prs(incompatible) or 'none'}; found "
            f"{format_prs(actual_incompatible) or 'none'}"
        )
    return errors


def format_prs(numbers: set[int]) -> str:
    """Format PR numbers for diagnostics."""
    return ", ".join(f"#{number}" for number in sorted(numbers))


def main() -> int:
    """Run the checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--expected-prs", nargs="+", required=True)
    parser.add_argument("--expected-incompatible-prs", nargs="*", default=[])
    args = parser.parse_args()

    try:
        text = args.release.read_text(encoding="utf-8")
    except OSError as exc:
        parser.error(str(exc))

    expected = parse_prs(args.expected_prs)
    incompatible = parse_prs(args.expected_incompatible_prs)
    if not incompatible <= expected:
        parser.error("incompatible PRs must also appear in --expected-prs")

    errors = validate(text, expected, incompatible)
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        return 1
    print(f"OK: {len(expected)} PRs accounted for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
