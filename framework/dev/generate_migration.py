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
"""Generate Alembic migration revision without requiring a persistent database.

This tool creates a temporary SQLite database, upgrades it to the current head
revision, then runs autogenerate to detect schema changes by comparing the
database against the current table definitions in the schema/ directory.

The temporary database file is automatically cleaned up after the migration
script is generated in the versions/ directory.

Usage:
    python -m dev.generate_migration "Add user preferences table"
    python -m dev.generate_migration --head ee@head "Add EE table"
"""


import argparse
import subprocess
from pathlib import Path

DEFAULT_HEAD = "flwr@head"


def _remove_existing_temp_db(db_path: Path) -> None:
    """Remove a stale temporary database before running Alembic."""
    try:
        db_path.unlink()
    except FileNotFoundError:
        return
    except OSError as e:
        raise SystemExit(
            f"Failed to remove existing temporary database '{db_path}': {e}"
        ) from e


def main() -> None:
    """Parse arguments and generate migration revision."""
    parser = argparse.ArgumentParser(
        description="Generate an Alembic migration revision."
    )
    parser.add_argument("message", help="Migration description message.")
    parser.add_argument(
        "--head",
        default=DEFAULT_HEAD,
        help=(
            "Branch head to extend (default: %(default)s). "
            "Use 'ee@head' to target the EE branch."
        ),
    )
    args = parser.parse_args()

    # Clean up any leftover state.db from a previous failed run
    db_path = Path("state.db")
    if db_path.exists():
        _remove_existing_temp_db(db_path)

    try:
        # Upgrade all branches to their latest revisions
        print("Upgrading temporary database to head revisions...")
        subprocess.run(
            ["alembic", "upgrade", "heads"],
            check=True,
            capture_output=False,
        )

        # Generate migration targeting the specified branch head
        print(f"Generating migration: {args.message}")
        subprocess.run(
            [
                "alembic",
                "revision",
                "--autogenerate",
                "--head",
                args.head,
                "-m",
                args.message,
            ],
            check=True,
            capture_output=False,
        )

        print("Migration generated successfully!")

    finally:
        # Clean up the state.db file
        if db_path.exists():
            try:
                db_path.unlink()
                print("Cleaned up temporary database file.")
            except OSError as e:
                print(
                    f"Warning: Failed to clean up temporary database file: {e}",
                )


if __name__ == "__main__":
    main()
