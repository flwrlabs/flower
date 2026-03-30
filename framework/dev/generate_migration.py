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

Example:
    python -m dev.generate_migration "Add user preferences table"
"""


from pathlib import Path
import subprocess
import sys


def main() -> None:
    """Parse arguments and generate migration revision."""
    if len(sys.argv) < 2:
        print("Usage: python -m dev.generate_migration <message>")
        print()
        print("Example:")
        print('  python -m dev.generate_migration "Add new_column to run table"')
        sys.exit(1)

    message = sys.argv[1]

    try:
        # Run alembic upgrade head - blocks until complete
        print("Upgrading temporary database to head revision...")
        subprocess.run(
            ["alembic", "upgrade", "head"],
            check=True,
            capture_output=False,
        )

        # Run alembic revision autogenerate - only runs after upgrade completes
        print(f"Generating migration: {message}")
        subprocess.run(
            ["alembic", "revision", "--autogenerate", "-m", message],
            check=True,
            capture_output=False,
        )

        print("Migration generated successfully!")

    finally:
        # Clean up the state.db file
        db_path = Path("state.db")
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
