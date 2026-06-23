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
"""Tests for Alembic migration helpers."""


import unittest
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.engine import URL, Connection, Engine

from flwr.common.constant import SubStatus
from flwr.supercore.constant import TaskType
from flwr.supercore.exit import ExitCode
from flwr.supercore.state.alembic.utils import (
    ALEMBIC_DIR,
    ALEMBIC_VERSION_TABLE,
    FLWR_STATE_BASELINE_REVISION,
    _get_baseline_metadata,
    _metadata_providers,
    _version_locations,
    build_alembic_config,
    get_combined_metadata,
    register_metadata_provider,
    register_version_location,
    run_migrations,
)


class TestAlembicRun(unittest.TestCase):
    """Test Alembic migration helper utilities."""

    def setUp(self) -> None:
        """Create temporary directory for test databases."""
        self.original_locations = _version_locations.copy()
        self.temp_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        _version_locations.clear()
        _version_locations.extend(self.original_locations)
        self.temp_dir.cleanup()

    def create_engine(self, db_name: str = "state.db") -> Engine:
        """Create a SQLAlchemy engine for a test database."""
        db_path = self.temp_path / db_name
        return create_engine(f"sqlite:///{db_path}")

    def create_mock_engine(self, dialect_name: str) -> tuple[MagicMock, MagicMock]:
        """Create a mock SQLAlchemy engine and connection."""
        engine = MagicMock()
        connection = MagicMock(spec=Connection)
        engine.dialect.name = dialect_name
        engine.connect.return_value.__enter__.return_value = connection
        return engine, connection

    def create_advisory_lock_event_recorder(
        self, events: list[str]
    ) -> Callable[[object, object], None]:
        """Return a side effect that records advisory lock SQL statements."""

        def record_event(statement: object, _params: object) -> None:
            sql = str(statement)
            if "pg_advisory_lock" in sql:
                events.append("lock")
            if "pg_advisory_unlock" in sql:
                events.append("unlock")

        return record_event

    def upgrade_to_revision(self, engine: Engine, revision: str) -> None:
        """Upgrade the test database to the specified Alembic revision."""
        command.upgrade(build_alembic_config(engine), revision)

    def downgrade_to_revision(self, engine: Engine, revision: str) -> None:
        """Downgrade the test database to the specified Alembic revision."""
        command.downgrade(build_alembic_config(engine), revision)

    def build_run_row(  # pylint: disable=too-many-arguments
        self,
        run_id: int,
        *,
        fab_id: str,
        fab_version: str,
        fab_hash: str,
        pending_at: str,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Build a run row with defaults suited for migration backfill tests."""
        row: dict[str, Any] = {
            "run_id": run_id,
            "fab_id": fab_id,
            "fab_version": fab_version,
            "fab_hash": fab_hash,
            "override_config": "{}",
            "pending_at": pending_at,
            "starting_at": "",
            "running_at": "",
            "finished_at": "",
            "usage_reported_at": "",
            "sub_status": None,
            "details": None,
            "federation": "fed",
            "federation_config": None,
            "run_type": "serverapp",
            "flwr_aid": "aid",
            "bytes_sent": 0,
            "bytes_recv": 0,
            "clientapp_runtime": 0.0,
        }
        row.update(overrides)
        return row

    def insert_runs(
        self, connection: Connection, runs: Sequence[dict[str, Any]]
    ) -> None:
        """Insert run rows into the test database."""
        connection.execute(
            text(
                """
                INSERT INTO run (
                    run_id, fab_id, fab_version, fab_hash, override_config,
                    pending_at, starting_at, running_at, finished_at,
                    usage_reported_at, sub_status, details, federation,
                    federation_config, run_type, flwr_aid, bytes_sent,
                    bytes_recv, clientapp_runtime
                ) VALUES (
                    :run_id, :fab_id, :fab_version, :fab_hash, :override_config,
                    :pending_at, :starting_at, :running_at, :finished_at,
                    :usage_reported_at, :sub_status, :details, :federation,
                    :federation_config, :run_type, :flwr_aid, :bytes_sent,
                    :bytes_recv, :clientapp_runtime
                )
                """
            ),
            list(runs),
        )

    def assert_timestamp_equal(self, actual: Any, expected: str) -> None:
        """Assert DB timestamp values match regardless of storage formatting."""

        def normalize(value: Any) -> str:
            timestamp = str(value).replace("T", " ")
            timestamp = timestamp.replace("+00:00", "")
            return timestamp.split(".", maxsplit=1)[0]

        self.assertEqual(normalize(actual), normalize(expected))

    def test_run_migrations_sets_revision(self) -> None:
        """Ensure migrations advance the database to the latest head."""
        # Prepare
        engine = self.create_engine()
        try:
            # Execute & Assert
            # Initially, there should be no alembic_version table or revision.
            self.assertEqual(get_current_revisions(engine), set())
            self.assertTrue(check_migrations_pending(engine))

            run_migrations(engine)

            # After migration, alembic_version should be set to the latest heads.
            current = get_current_revisions(engine)
            script = ScriptDirectory.from_config(build_alembic_config(engine))
            self.assertEqual(current, set(script.get_heads()))
            # No pending migrations should remain.
            self.assertFalse(check_migrations_pending(engine))
        finally:
            engine.dispose()

    @patch("flwr.supercore.state.alembic.utils._run_migration_workflow")
    def test_run_migrations_uses_postgresql_advisory_lock(
        self, mock_run_migrations: MagicMock
    ) -> None:
        """Ensure PostgreSQL migrations are serialized with an advisory lock."""
        # Prepare
        events: list[str] = []
        engine, connection = self.create_mock_engine("postgresql")
        connection.in_transaction.return_value = True
        connection.execute.side_effect = self.create_advisory_lock_event_recorder(
            events
        )
        connection.commit.side_effect = lambda: events.append("commit")
        mock_run_migrations.side_effect = lambda _engine, _bind: events.append(
            "migrate"
        )

        # Execute
        run_migrations(engine)

        # Assert
        self.assertEqual(
            events, ["lock", "commit", "migrate", "commit", "unlock", "commit"]
        )
        self.assertEqual(connection.commit.call_count, 3)
        mock_run_migrations.assert_called_once_with(engine, connection)

    @patch("flwr.supercore.state.alembic.utils._run_migration_workflow")
    def test_run_migrations_releases_postgresql_advisory_lock_on_error(
        self, mock_run_migrations: MagicMock
    ) -> None:
        """Ensure PostgreSQL advisory locks are released when migrations fail."""
        # Prepare
        events: list[str] = []
        engine, connection = self.create_mock_engine("postgresql")
        connection.in_transaction.return_value = True
        connection.execute.side_effect = self.create_advisory_lock_event_recorder(
            events
        )
        mock_run_migrations.side_effect = RuntimeError("migration failed")

        # Execute & Assert
        with self.assertRaisesRegex(RuntimeError, "migration failed"):
            run_migrations(engine)

        self.assertEqual(events, ["lock", "unlock"])
        connection.rollback.assert_called_once_with()
        mock_run_migrations.assert_called_once_with(engine, connection)

    @patch("flwr.supercore.state.alembic.utils._run_migration_workflow")
    def test_run_migrations_does_not_lock_non_postgresql_backends(
        self, mock_run_migrations: MagicMock
    ) -> None:
        """Ensure non-PostgreSQL backends keep the existing migration behavior."""
        # Prepare
        engine, _ = self.create_mock_engine("sqlite")

        # Execute
        run_migrations(engine)

        # Assert
        engine.connect.assert_not_called()
        mock_run_migrations.assert_called_once_with(engine, engine)

    def test_migrated_schema_matches_metadata(self) -> None:
        """Verify that migrations match current SQLAlchemy metadata."""
        # Prepare
        metadata = get_combined_metadata()
        engine = self.create_engine()
        try:
            # Execute: create a fresh database and run migrations
            run_migrations(engine)
            with engine.connect() as connection:
                context = MigrationContext.configure(
                    connection,
                    opts={
                        "compare_type": True,
                        "compare_server_default": True,
                    },
                )
                # Compare the migrated database schema against the metadata
                diffs = compare_metadata(context, metadata)
            # Assert
            self.assertEqual(diffs, [])
        finally:
            engine.dispose()

    def test_primary_task_backfill_populates_historical_runs(self) -> None:
        """Ensure historical runs get backfilled primary tasks during migration."""
        engine = self.create_engine("primary_task_backfill.db")
        try:
            self.upgrade_to_revision(engine, "b277e6f3656c")
            with engine.begin() as connection:
                self.insert_runs(
                    connection,
                    [
                        self.build_run_row(
                            run_id=101,
                            fab_id="publisher/pending",
                            fab_version="1.0.0",
                            fab_hash="fab-pending",
                            pending_at="2026-04-27T10:00:00+00:00",
                            federation="fed-a",
                            flwr_aid="aid-a",
                        ),
                        self.build_run_row(
                            run_id=102,
                            fab_id="publisher/sim",
                            fab_version="2.0.0",
                            fab_hash="fab-sim",
                            pending_at="2026-04-27T11:00:00+00:00",
                            starting_at="2026-04-27T11:01:00+00:00",
                            running_at="2026-04-27T11:02:00+00:00",
                            finished_at="2026-04-27T11:03:00+00:00",
                            sub_status="completed",
                            details="done",
                            federation="fed-b",
                            federation_config="{}",
                            run_type="simulation",
                            flwr_aid="aid-b",
                            bytes_sent=7,
                            bytes_recv=8,
                            clientapp_runtime=9.5,
                        ),
                        self.build_run_row(
                            run_id=103,
                            fab_id="publisher/failed",
                            fab_version="3.0.0",
                            fab_hash="fab-failed",
                            pending_at="2026-04-27T12:00:00+00:00",
                            finished_at="2026-04-27T12:05:00+00:00",
                            sub_status="failed",
                            details="boom",
                            federation="fed-c",
                            flwr_aid="aid-c",
                            bytes_sent=1,
                            bytes_recv=2,
                            clientapp_runtime=3.0,
                        ),
                    ],
                )

            self.upgrade_to_revision(engine, "heads")

            with engine.connect() as connection:
                runs = {
                    row["run_id"]: row
                    for row in connection.execute(
                        text(
                            """
                            SELECT run_id, primary_task_id
                            FROM run
                            ORDER BY run_id
                            """
                        )
                    ).mappings()
                }
                tasks = {
                    row["task_id"]: row
                    for row in connection.execute(
                        text(
                            """
                            SELECT task_id, type, run_id, fab_hash, model_ref,
                                   pending_at, starting_at, running_at, finished_at,
                                   sub_status, details
                            FROM task
                            ORDER BY task_id
                            """
                        )
                    ).mappings()
                }

            # Assert: Primary task inserted for run 101
            task = tasks[runs[101]["primary_task_id"]]
            self.assertEqual(task["type"], TaskType.SERVER_APP)
            self.assertEqual(task["run_id"], 101)
            self.assert_timestamp_equal(task["pending_at"], "2026-04-27T10:00:00+00:00")
            self.assertIsNone(task["starting_at"])
            self.assertIsNone(task["running_at"])
            self.assertIsNone(task["finished_at"])

            # Assert: Primary task inserted for run 102
            task = tasks[runs[102]["primary_task_id"]]
            self.assertEqual(task["type"], TaskType.SIMULATION)
            self.assertEqual(task["run_id"], 102)
            self.assert_timestamp_equal(
                task["starting_at"], "2026-04-27T11:01:00+00:00"
            )
            self.assert_timestamp_equal(task["running_at"], "2026-04-27T11:02:00+00:00")
            self.assert_timestamp_equal(
                task["finished_at"], "2026-04-27T11:03:00+00:00"
            )
            self.assertEqual(task["sub_status"], "completed")
            self.assertEqual(task["details"], "done")

            # Assert: Primary task inserted for run 103
            task = tasks[runs[103]["primary_task_id"]]
            self.assertEqual(task["type"], TaskType.SERVER_APP)
            self.assertEqual(task["run_id"], 103)
            self.assert_timestamp_equal(
                task["finished_at"], "2026-04-27T12:05:00+00:00"
            )
            self.assertEqual(task["sub_status"], "failed")
            self.assertEqual(task["details"], "boom")
        finally:
            engine.dispose()

    def test_primary_task_backfill_stops_running_runs(self) -> None:
        """Ensure STARTING/RUNNING runs are migrated to FINISHED:STOPPED."""
        engine = self.create_engine("primary_task_backfill_stopped.db")
        try:
            self.upgrade_to_revision(engine, "b277e6f3656c")
            with engine.begin() as connection:
                self.insert_runs(
                    connection,
                    [
                        self.build_run_row(
                            201,
                            fab_id="publisher/live",
                            fab_version="1.0.0",
                            fab_hash="fab-live",
                            pending_at="2026-04-27T13:00:00+00:00",
                            starting_at="2026-04-27T13:01:00+00:00",
                            running_at="2026-04-27T13:02:00+00:00",
                            federation="fed-live",
                            flwr_aid="aid-live",
                        )
                    ],
                )

            self.upgrade_to_revision(engine, "heads")

            with engine.connect() as connection:
                primary_task = (
                    connection.execute(
                        text(
                            """
                        SELECT
                            r.primary_task_id,
                            t.type,
                            t.starting_at,
                            t.running_at,
                            t.finished_at,
                            t.sub_status,
                            t.details
                        FROM run AS r
                        JOIN task AS t ON t.task_id = r.primary_task_id
                        WHERE r.run_id = :run_id
                        """
                        ),
                        {"run_id": 201},
                    )
                    .mappings()
                    .one()
                )

            self.assertIsNotNone(primary_task["primary_task_id"])
            self.assertEqual(primary_task["type"], TaskType.SERVER_APP)
            self.assert_timestamp_equal(
                primary_task["starting_at"], "2026-04-27T13:01:00+00:00"
            )
            self.assert_timestamp_equal(
                primary_task["running_at"], "2026-04-27T13:02:00+00:00"
            )
            self.assertTrue(primary_task["finished_at"])
            self.assertEqual(primary_task["sub_status"], SubStatus.STOPPED)
            self.assertEqual(
                primary_task["details"], "Run stopped during server upgrade."
            )
        finally:
            engine.dispose()

    def test_squashed_migration_downgrade_restores_parent_schema(self) -> None:
        """Ensure downgrade from the squashed head returns to the nonce revision."""
        engine = self.create_engine("squashed_migration_downgrade.db")
        try:
            self.upgrade_to_revision(engine, "b277e6f3656c")
            with engine.begin() as connection:
                self.insert_runs(
                    connection,
                    [
                        self.build_run_row(
                            run_id=301,
                            fab_id="publisher/downgrade",
                            fab_version="1.0.0",
                            fab_hash="fab-downgrade",
                            pending_at="2026-04-27T10:00:00+00:00",
                            starting_at="2026-04-27T10:01:00+00:00",
                            running_at="2026-04-27T10:02:00+00:00",
                            finished_at="2026-04-27T10:03:00+00:00",
                            sub_status=SubStatus.COMPLETED,
                            details="done",
                            federation="fed",
                            flwr_aid="aid",
                        )
                    ],
                )
                connection.execute(
                    text(
                        """
                        INSERT INTO logs (timestamp, run_id, node_id, log)
                        VALUES (:timestamp, :run_id, :node_id, :log)
                        """
                    ),
                    {
                        "timestamp": 1.5,
                        "run_id": 301,
                        "node_id": 9,
                        "log": "log entry",
                    },
                )

            self.upgrade_to_revision(engine, "heads")
            self.downgrade_to_revision(engine, "b277e6f3656c")

            inspector = inspect(engine)
            self.assertTrue(inspector.has_table("nonce_store"))
            self.assertTrue(inspector.has_table("token_store"))
            self.assertFalse(inspector.has_table("task"))
            self.assertFalse(inspector.has_table("task_logs"))
            self.assertFalse(inspector.has_table("task_message"))

            run_columns = {column["name"] for column in inspector.get_columns("run")}
            self.assertIn("pending_at", run_columns)
            self.assertNotIn("primary_task_id", run_columns)

            with engine.connect() as connection:
                query = """
                SELECT pending_at, starting_at, running_at, finished_at,
                       sub_status, details
                FROM run
                WHERE run_id = :run_id
                """
                params = {"run_id": 301}
                run = connection.execute(text(query), params).mappings().one()
                query = """
                SELECT log
                FROM logs
                WHERE run_id = :run_id AND node_id = 0
                """
                copied_log = connection.execute(text(query), params).scalar_one()

            self.assert_timestamp_equal(run["pending_at"], "2026-04-27T10:00:00+00:00")
            self.assert_timestamp_equal(run["starting_at"], "2026-04-27T10:01:00+00:00")
            self.assert_timestamp_equal(run["running_at"], "2026-04-27T10:02:00+00:00")
            self.assert_timestamp_equal(run["finished_at"], "2026-04-27T10:03:00+00:00")
            self.assertEqual(run["sub_status"], SubStatus.COMPLETED)
            self.assertEqual(run["details"], "done")
            self.assertEqual(copied_log, "log entry")
        finally:
            engine.dispose()

    def test_legacy_database_is_stamped_and_upgraded_successfully(self) -> None:
        """Ensure legacy databases without alembic_version is stamped and upgraded."""
        # Prepare
        engine = self.create_engine()
        try:
            # Execute & Assert
            # Simulate pre-Alembic behavior: create tables at baseline schema. By
            # construction, there is no alembic_version table or revision.
            baseline_metadata = _get_baseline_metadata()
            baseline_metadata.create_all(engine)
            self.assertEqual(get_current_revisions(engine), set())
            self.assertFalse(inspect(engine).has_table(ALEMBIC_VERSION_TABLE))

            run_migrations(engine)

            # After migration, alembic_version should be set to the latest heads with
            # no pending migrations.
            current = get_current_revisions(engine)
            script = ScriptDirectory.from_config(build_alembic_config(engine))
            self.assertEqual(current, set(script.get_heads()))
            self.assertFalse(check_migrations_pending(engine))
        finally:
            engine.dispose()

    def test_legacy_mismatch_with_missing_tables_exits_with_guidance(self) -> None:
        """Ensure mismatched legacy schemas should fail and exit with a clear error."""
        # Prepare
        engine = self.create_engine()
        try:
            # Create only a subset of baseline tables to simulate incomplete legacy DB
            # Note that this is unlikely to happen as SuperLink requires a specific
            # schema, but we test it for completeness.
            baseline_metadata = _get_baseline_metadata()
            tables_to_create = list(baseline_metadata.tables.values())[:3]
            for table in tables_to_create:
                table.create(engine)

            # Execute & Assert
            with patch(
                "flwr.supercore.state.alembic.utils.flwr_exit",
                side_effect=SystemExit(1),
            ) as mock_exit:
                with self.assertRaises(SystemExit):
                    run_migrations(engine)

                # Verify flwr_exit was called with correct arguments
                mock_exit.assert_called_once()
                call_args = mock_exit.call_args
                self.assertEqual(
                    call_args[0][0], ExitCode.SUPERLINK_DATABASE_SCHEMA_MISMATCH
                )
                # Verify error message mentions missing baseline tables
                error_msg = call_args[0][1]
                self.assertIn("missing baseline tables", error_msg.lower())
        finally:
            engine.dispose()

    def test_legacy_mismatch_with_missing_columns_exits_with_guidance(self) -> None:
        """Ensure legacy schemas with missing columns should fail and exit with a clear
        error."""
        # Prepare
        engine = self.create_engine()
        try:
            # Create node table with only some columns (missing required ones)
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "CREATE TABLE node (node_id INTEGER, status TEXT)"
                )
            # Create other tables with baseline schemas
            baseline_metadata = _get_baseline_metadata()
            for table_name in baseline_metadata.tables:
                if table_name != "node":
                    baseline_metadata.tables[table_name].create(engine)

            # Execute & Assert
            with patch(
                "flwr.supercore.state.alembic.utils.flwr_exit",
                side_effect=SystemExit(1),
            ) as mock_exit:
                with self.assertRaises(SystemExit):
                    run_migrations(engine)

                # Verify flwr_exit was called
                mock_exit.assert_called_once()
                call_args = mock_exit.call_args
                self.assertEqual(
                    call_args[0][0], ExitCode.SUPERLINK_DATABASE_SCHEMA_MISMATCH
                )
                # Verify error message mentions missing columns
                error_msg = call_args[0][1]
                self.assertIn("missing columns", error_msg.lower())
        finally:
            engine.dispose()

    def test_legacy_database_with_extra_tables_and_columns_succeeds(self) -> None:
        """Ensure legacy databases with extra tables/columns can be migrated.

        This tests backward compatibility: a legacy DB might have extra tables or
        columns that were added manually. The verification should be permissive
        and only fail on MISSING baseline tables/columns.
        """
        # Prepare
        engine = self.create_engine()
        try:
            # Create baseline schema
            baseline_metadata = _get_baseline_metadata()
            baseline_metadata.create_all(engine)

            # Commit the transaction to flush tables
            engine.dispose()
            engine = self.create_engine()

            # Add extra table and column to simulate forward-compatible scenario
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "CREATE TABLE custom_user_table (id INTEGER)"
                )
                # Add extra column to existing table
                inspector = inspect(engine)
                if inspector.has_table("node"):
                    connection.exec_driver_sql(
                        "ALTER TABLE node ADD COLUMN custom_field TEXT"
                    )

            # Execute: should succeed and stamp/upgrade successfully
            run_migrations(engine)

            current = get_current_revisions(engine)
            script = ScriptDirectory.from_config(build_alembic_config(engine))
            # Assert
            self.assertEqual(current, set(script.get_heads()))
            self.assertFalse(check_migrations_pending(engine))
        finally:
            engine.dispose()

    def test_run_migrations_upgrades_to_all_heads(self) -> None:
        """Ensure migrations upgrade to every head across registered branches."""
        # Prepare: add a synthetic migration branch without relying on EE imports
        _version_locations.clear()
        extra_versions = self.temp_path / "external_versions"
        extra_versions.mkdir()
        write_revision_file(
            extra_versions / "rev_external_branch.py",
            revision="external_branch_001",
            down_revision=FLWR_STATE_BASELINE_REVISION,
        )
        register_version_location(extra_versions)

        engine = self.create_engine()
        try:
            # Execute
            run_migrations(engine)

            # Assert: both the core head and synthetic branch head are current
            current = get_current_revisions(engine)
            script = ScriptDirectory.from_config(build_alembic_config(engine))
            heads = set(script.get_heads())

            self.assertEqual(current, heads)
            self.assertIn("external_branch_001", current)
            self.assertGreater(len(current), 1)
            self.assertFalse(check_migrations_pending(engine))
        finally:
            engine.dispose()


def write_revision_file(path: Path, revision: str, down_revision: str) -> None:
    """Write a minimal Alembic revision file for tests."""
    path.write_text(
        f'''"""Synthetic migration branch for tests.

Revision ID: {revision}
Revises: {down_revision}
Create Date: 2026-03-23 00:00:00.000000
"""

from collections.abc import Sequence


revision: str = "{revision}"
down_revision: str | Sequence[str] | None = "{down_revision}"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Apply the synthetic test migration."""


def downgrade() -> None:
    """Revert the synthetic test migration."""
''',
        encoding="utf-8",
    )


def get_current_revisions(engine: Engine) -> set[str]:
    """Return the current Alembic revisions for the given database."""
    with engine.connect() as connection:
        context = MigrationContext.configure(connection)
        return set(context.get_current_heads())


def check_migrations_pending(engine: Engine) -> bool:
    """Return True if the database is not on the latest migration heads."""
    current = get_current_revisions(engine)
    script = ScriptDirectory.from_config(build_alembic_config(engine))
    heads = set(script.get_heads())
    if not current:
        return True
    return current != heads


class TestMetadataProviderRegistry(unittest.TestCase):
    """Test external metadata provider registration and collision detection."""

    def setUp(self) -> None:
        """Save the current state of the metadata providers registry."""
        self.original_providers = _metadata_providers.copy()

    def tearDown(self) -> None:
        """Restore the metadata providers registry to its original state."""
        _metadata_providers.clear()
        _metadata_providers.extend(self.original_providers)

    def test_external_provider_table_is_included_in_combined_metadata(self) -> None:
        """Ensure external provider tables are included in combined metadata."""

        # Prepare: Define a mock external metadata provider with a unique table
        def mock_external_provider() -> MetaData:
            metadata = MetaData()
            Table(
                "external_custom_table",
                metadata,
                Column("id", Integer, primary_key=True),
                Column("name", String),
            )
            return metadata

        # Execute: Register the provider
        register_metadata_provider(mock_external_provider)

        # Assert: The external table should be in the combined metadata
        combined = get_combined_metadata()
        self.assertIn("external_custom_table", combined.tables)
        # Verify the table has expected columns
        table = combined.tables["external_custom_table"]
        column_names = {col.name for col in table.columns}
        self.assertEqual(column_names, {"id", "name"})

    def test_external_provider_collision_raises_error(self) -> None:
        """Ensure table name collisions from external providers raise ValueError."""

        # Prepare: Define a mock provider that collides with existing 'node' table
        # (node is defined in linkstate_tables.py)
        def mock_colliding_provider() -> MetaData:
            metadata = MetaData()
            Table(
                "node",  # This table already exists in linkstate_tables
                metadata,
                Column("id", Integer, primary_key=True),
            )
            return metadata

        # Execute: Register the colliding provider
        register_metadata_provider(mock_colliding_provider)

        # Assert: Getting combined metadata should raise ValueError
        with self.assertRaises(ValueError) as context:
            get_combined_metadata()

        # Verify error message identifies the collision
        error_msg = str(context.exception)
        self.assertIn("node", error_msg)
        self.assertIn("collision", error_msg.lower())
        self.assertIn("mock_colliding_provider", error_msg)


class TestVersionLocationRegistry(unittest.TestCase):
    """Test external version location registration and build_alembic_config."""

    def setUp(self) -> None:
        """Save the current state of the version locations registry."""
        self.original_locations = _version_locations.copy()
        self.temp_dir = TemporaryDirectory()  # pylint: disable=consider-using-with
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Restore the version locations registry to its original state."""
        _version_locations.clear()
        _version_locations.extend(self.original_locations)
        self.temp_dir.cleanup()

    def test_register_version_location_adds_to_registry_idempotently(self) -> None:
        """Ensure register_version_location adds path to registry without duplicates."""
        # Prepare
        external_versions = Path("/some/external/versions")
        initial_count = len(_version_locations)

        # Execute: Register the same path twice
        register_version_location(external_versions)
        register_version_location(external_versions)

        # Assert: Path is in registry exactly once
        self.assertIn(external_versions, _version_locations)
        self.assertEqual(len(_version_locations), initial_count + 1)

    def test_build_alembic_config_includes_all_version_locations(self) -> None:
        """Ensure build_alembic_config includes base and registered version
        locations."""
        # Prepare
        db_path = self.temp_path / "state.db"
        engine = create_engine(f"sqlite:///{db_path}")
        external_versions = Path("/external/versions")
        register_version_location(external_versions)

        try:
            # Execute
            config = build_alembic_config(engine)

            # Assert: version_locations includes both base and external paths
            version_locations = config.get_main_option("version_locations")
            assert version_locations is not None
            base_versions = str(ALEMBIC_DIR / "versions")
            self.assertIn(base_versions, version_locations)
            self.assertIn(str(external_versions), version_locations)
        finally:
            engine.dispose()

    def test_build_alembic_config_preserves_url_password_and_percent_encoding(
        self,
    ) -> None:
        """Ensure Alembic receives the unmasked URL with escaped percent signs."""
        # Prepare: use a URL object so the test does not depend on a PostgreSQL driver.
        url = URL.create(
            "postgresql",
            username="flower",
            password="pa%ss=word",
            host="db.example",
            database="state",
        )
        engine = cast(Engine, SimpleNamespace(url=url))

        # Execute
        config = build_alembic_config(engine)

        # Assert: get_main_option should not raise interpolation errors and should
        # return the exact URL Alembic needs to connect.
        expected_url = url.render_as_string(hide_password=False)
        config_url = config.get_main_option("sqlalchemy.url")
        self.assertEqual(config_url, expected_url)
        assert config_url is not None
        self.assertIn("pa%25ss%3Dword", config_url)
        self.assertNotIn("***", config_url)
