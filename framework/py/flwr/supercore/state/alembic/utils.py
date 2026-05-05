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
"""Helpers for running and validating Alembic migrations."""


from collections.abc import Callable
from logging import INFO
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import MetaData, create_engine, inspect, pool
from sqlalchemy.engine import Engine

from flwr.common.exit import ExitCode, flwr_exit
from flwr.common.logger import log
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.state.schema.linkstate_tables import create_linkstate_metadata
from flwr.supercore.state.schema.objectstore_tables import create_objectstore_metadata

# Type alias for metadata provider functions
MetadataProvider = Callable[[], MetaData]

# Registry for additional metadata providers (e.g., from ee module)
_metadata_providers: list[MetadataProvider] = []

# Registry for additional version locations (e.g., from ee module)
_version_locations: list[Path] = []

ALEMBIC_DIR = Path(__file__).resolve().parent
ALEMBIC_VERSION_TABLE = "alembic_version"
FLWR_STATE_BASELINE_REVISION = "8e65d8ae60b0"  # Never change this
FLWR_STATE_LATEST_REVISIONS = "heads"


def register_metadata_provider(provider: MetadataProvider) -> None:
    """Register an additional metadata provider for Alembic migrations.

    This allows external modules to register their table definitions so
    they are included in the combined metadata used by Alembic for
    migrations.

    Parameters
    ----------
    provider : MetadataProvider
        A callable that returns a SQLAlchemy MetaData object containing
        table definitions to be included in migrations.
    """
    # Avoid duplicate registration to keep the registry idempotent
    if provider not in _metadata_providers:
        _metadata_providers.append(provider)


def register_version_location(version_dir: Path) -> None:
    """Register an additional versions directory for Alembic migrations.

    This allows external modules to register their migration scripts so
    they are included when running Alembic migrations.

    Parameters
    ----------
    version_dir : Path
        Path to a directory containing Alembic migration scripts.
    """
    # Avoid duplicate registration to keep the registry idempotent
    if version_dir not in _version_locations:
        _version_locations.append(version_dir)


def get_combined_metadata() -> MetaData:
    """Combine all Flower state metadata objects into a single MetaData instance.

    This ensures Alembic can track all tables across CoreState, LinkState,
    ObjectStore, and any registered external modules.

    Returns
    -------
    MetaData
        Combined SQLAlchemy MetaData with all Flower state tables.
    """
    # Start with linkstate tables
    metadata = create_linkstate_metadata()

    # Add corestate tables
    corestate_metadata = create_corestate_metadata()
    for table in corestate_metadata.tables.values():
        table.to_metadata(metadata)

    # Add objectstore tables
    objectstore_metadata = create_objectstore_metadata()
    for table in objectstore_metadata.tables.values():
        table.to_metadata(metadata)

    # Add tables from registered external providers
    for provider in _metadata_providers:
        extra_metadata = provider()
        for table in extra_metadata.tables.values():
            if table.name in metadata.tables:
                raise ValueError(
                    f"Table name collision: '{table.name}' from provider "
                    f"'{provider.__module__}.{provider.__qualname__}' "
                    f"conflicts with an existing table. External providers"
                    "must use unique table names."
                )
            table.to_metadata(metadata)

    return metadata


def run_migrations(engine: Engine) -> None:
    """Run pending Alembic migrations, handling pre-Alembic legacy databases.

    Expected scenarios:
    - If alembic_version exists: run migrations normally.
    - If DB is empty, e.g. when newly created: run migrations normally.
    - If DB is pre-Alembic and schema is mismatched: exit with guidance.
    - If DB is pre-Alembic and schema matches baseline: stamp, then upgrade.
    """
    config = build_alembic_config(engine)
    has_version_table = _has_alembic_version_table(engine)

    # Standard database with version tracking: just upgrade.
    if has_version_table:
        command.upgrade(config, FLWR_STATE_LATEST_REVISIONS)
        return

    table_names = _get_user_table_names(engine)

    # Empty/new database: run all migrations from scratch.
    if not table_names:
        command.upgrade(config, FLWR_STATE_LATEST_REVISIONS)
        return

    # Pre-Alembic database detected without version tracking: verify database matches
    # baseline schema before stamping version and upgrading.
    is_valid, error_msg = _verify_legacy_schema_matches_baseline(engine)

    # This is an edge case and unlikely to happen since SuperLink requires a specific
    # schema to operate normally.
    if not is_valid:
        flwr_exit(
            ExitCode.SUPERLINK_DATABASE_SCHEMA_MISMATCH,
            "Detected a pre-Alembic Flower state database, but its schema does not "
            f"match the baseline migration (revision {FLWR_STATE_BASELINE_REVISION}). "
            "Back up the database and either migrate it manually to the baseline "
            "schema or start with a fresh database. "
            f"{error_msg}",
        )

    log(
        INFO,
        "Detected pre-Alembic state database without alembic_version; stamping to %s "
        "before upgrading.",
        FLWR_STATE_BASELINE_REVISION,
    )
    stamp_existing_database(engine, FLWR_STATE_BASELINE_REVISION)
    command.upgrade(config, FLWR_STATE_LATEST_REVISIONS)
    log(INFO, "Flower state database stamped and upgraded successfully!")


def build_alembic_config(engine: Engine) -> Config:
    """Create Alembic config with script location, DB URL, and version locations."""
    config = Config()
    config.set_main_option("script_location", str(ALEMBIC_DIR))
    # render_as_string(hide_password=False) preserves the real password;
    # str(engine.url) masks it as '***' which breaks Alembic's connection.
    # The .replace("%", "%%") escapes percent-encoded characters (e.g. %3D)
    # so ConfigParser doesn't treat them as interpolation placeholders.
    url = engine.url.render_as_string(hide_password=False).replace("%", "%%")
    config.set_main_option("sqlalchemy.url", url)

    # Combine base version location with any registered external locations
    base_versions = ALEMBIC_DIR / "versions"
    all_version_locations = [base_versions] + _version_locations
    version_locations_str = " ".join(str(loc) for loc in all_version_locations)
    config.set_main_option("version_locations", version_locations_str)

    return config


def _get_user_table_names(engine: Engine) -> set[str]:
    """Return non-internal table names for the given engine."""
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    # Exclude SQLite internal tables (for example, sqlite_sequence)
    return {name for name in table_names if not name.startswith("sqlite_")}


def _has_alembic_version_table(engine: Engine) -> bool:
    """Return True if the Alembic version table exists."""
    inspector = inspect(engine)
    return inspector.has_table(ALEMBIC_VERSION_TABLE)


def _get_baseline_metadata() -> MetaData:
    """Create an in-memory DB at baseline revision and reflect its schema.

    Uses an in-memory SQLite database instead of a temporary file to avoid requiring
    filesystem write access. Note that this function is only invoked for pre-Alembic
    databases.

    The implementation uses StaticPool and passes an active connection via
    config.attributes to Alembic's env.py. This ensures the same in-memory database
    instance is used throughout migration and reflection, since each new connection to
    sqlite:///:memory: creates a separate empty database.
    """
    # Create an in-memory SQLite database with StaticPool to ensure connection reuse.
    # This is needed because in-memory databases are instance-specific per connection.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=pool.StaticPool,
    )
    try:
        # Open a connection and pass it to Alembic to ensure the in-memory database
        # persists throughout the migration process. Without this, Alembic would
        # create a new connection (and thus a new empty database) from the URL.
        with engine.begin() as connection:
            config = build_alembic_config(engine)
            # Store the connection in config.attributes so env.py can use it directly.
            # This prevents Alembic from creating a new connection and losing our data.
            config.attributes["connection"] = connection
            command.upgrade(config, FLWR_STATE_BASELINE_REVISION)

        # Reflect the baseline schema from the in-memory database.
        # At this point, the StaticPool ensures we're still connected to the same
        # database instance that contains the migrated tables.
        baseline_metadata = MetaData()
        baseline_metadata.reflect(
            bind=engine,
            only=lambda table_name, _: table_name != ALEMBIC_VERSION_TABLE,
        )
    finally:
        engine.dispose()

    return baseline_metadata


def _verify_legacy_schema_matches_baseline(engine: Engine) -> tuple[bool, str]:
    """Verify legacy schema matches baseline tables and columns.

    Only missing tables/columns are reported as errors.

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message). If valid, error_message is empty.
    """
    inspector = inspect(engine)

    # Get the baseline schema by running migrations up to the baseline revision
    # in a temporary database and reflecting its schema
    baseline_metadata = _get_baseline_metadata()
    existing_tables = set(inspector.get_table_names())

    # Filter out SQLite internal tables and alembic_version
    existing_tables = {
        t
        for t in existing_tables
        if not t.startswith("sqlite_") and t != ALEMBIC_VERSION_TABLE
    }

    # Exclude alembic_version from baseline comparison
    expected_tables = {
        t for t in baseline_metadata.tables.keys() if t != ALEMBIC_VERSION_TABLE
    }
    missing_tables = expected_tables - existing_tables

    if missing_tables:
        table_list = ", ".join(sorted(existing_tables))
        missing_str = ", ".join(sorted(missing_tables))
        return False, (
            f"Detected tables: [{table_list}]. "
            f"Missing baseline tables: [{missing_str}]."
        )

    # Verify columns for each expected table
    for table_name in expected_tables:
        table = baseline_metadata.tables[table_name]
        expected_columns = {col.name for col in table.columns}
        actual_columns = {col["name"] for col in inspector.get_columns(table_name)}

        missing_cols = expected_columns - actual_columns
        if missing_cols:
            missing_cols_str = ", ".join(sorted(missing_cols))
            return False, (
                f"Table '{table_name}' missing columns: [{missing_cols_str}]."
            )

    return True, ""


def stamp_existing_database(
    engine: Engine, revision: str = FLWR_STATE_BASELINE_REVISION
) -> None:
    """Stamp an existing legacy database to the baseline Alembic revision."""
    command.stamp(build_alembic_config(engine), revision)
