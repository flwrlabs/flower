# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Generate UI metadata for framework documentation.

Example usage:

    python ./framework/docs/generate_docs_ui_metadata.py \
    --docs-ui-config ./dev/docs-ui-config.yml \
    --output ./framework/docs/build/docs-ui-metadata.json
"""


import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from packaging.version import Version

MIN_VERSION = Version("1.17.0")


def _collect_versions() -> list[dict[str, str]]:
    # Extract versions from tags (`vX.Y.Z` or `framework-X.Y.Z`)
    tags = subprocess.run(
        ["git", "tag", "-l", "v*.*.*", "framework-*.*.*"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    versions: list[Version] = []
    for tag in tags:
        if (ver := Version(tag.removeprefix("framework-"))) >= MIN_VERSION:
            versions.append(ver)
    versions.sort(reverse=True)

    # Create version items (name + URL) for the UI
    version_items = [{"name": "main", "url": "main"}]
    added_names = {"main"}
    for version in versions:
        # Get display name and URL for the version
        name = f"v{version.release[0]}.{version.release[1]}.x"
        url = f"{version.release[0]}.{version.release[1]}"

        if name not in added_names:
            added_names.add(name)
            version_items.append({"name": name, "url": url})

    return version_items


def _load_announcement(config_path: Path) -> dict[str, Any]:
    try:
        with config_path.open(encoding="utf-8") as file:
            announcement = yaml.safe_load(file)["announcement"]
            return {
                "enabled": bool(announcement["enabled"]),
                "html": str(announcement["html"]).strip(),
            }
    except Exception as e:
        raise RuntimeError(f"Failed to load announcement from {config_path}") from e


def main() -> None:
    """Generate docs UI metadata."""
    parser = argparse.ArgumentParser(description="Generate docs UI metadata")
    parser.add_argument(
        "--docs-ui-config",
        type=Path,
        required=True,
        help="Path to docs UI config YAML",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for generated UI metadata JSON",
    )
    args = parser.parse_args()

    metadata_payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "versions": _collect_versions(),
        "announcement": _load_announcement(args.docs_ui_config),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metadata_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
