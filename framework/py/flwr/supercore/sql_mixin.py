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
"""Mixin providing common SQL connection and initialization logic via SQLAlchemy."""


import re
from abc import ABC
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from logging import DEBUG, ERROR, WARNING
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, MetaData, create_engine, event, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from flwr.common.logger import log
from flwr.supercore.constant import FLWR_IN_MEMORY_SQLITE_DB_URL, SQLITE_PRAGMAS
from flwr.supercore.state.alembic.utils import run_migrations

_current_session: ContextVar[Session | None] = ContextVar(
    "current_sqlalchemy_session",
    default=None,
)


def _set_sqlite_pragmas(dbapi_conn: Any, _connection_record: Any) -> None:
    """Set SQLite pragmas for performance and correctness."""
    cursor = dbapi_conn.cursor()
    for pragma, value in SQLITE_PRAGMAS:
        cursor.execute(f"PRAGMA {pragma} = {value};")
    cursor.close()


def _log_query(  # pylint: disable=W0613,R0913,R0917
    conn: Any,
    cursor: Any,
    statement: str,
    parameters: Any,
    context: Any,
    executemany: bool,
) -> None:
    """Log SQL queries via Flower logger."""
    log(DEBUG, {"query": statement, "params": parameters})


class SqlMixin(ABC):
    """Mixin providing common SQLite connection and initialization logic.

    This mixin uses SQLAlchemy Core API for SQLite database access. It accepts either a
    database file path or a SQLite URL, automatically converting file paths to SQLite
    URLs.
    """

    def __init__(self, database_path: str) -> None:
        """Initialize the SqlMixin.

        Parameters
        ----------
        database_path : str
            Database location specifier. Can be:
            - A file path (relative or absolute): "state.db", "/var/data/state.db"
            - The special value ":memory:" for an in-memory database
            - A SQLite URL: "sqlite:///absolute/path/to/db.db"

            File paths are automatically converted to absolute paths and formatted
            as SQLite URLs. Empty or whitespace-only strings default to ":memory:".

        Warnings
        --------
        Providing an empty or whitespace-only string will log a warning and fall
        back to an in-memory database. For temporary databases, explicitly use
        ":memory:" to avoid warnings.
        """
        if not database_path or not database_path.strip():
            log(
                WARNING,
                "Empty `database_path` provided, defaulting to in-memory SQLite "
                "database",
            )
            database_path = ":memory:"

        # Auto-convert file path to SQLAlchemy SQLite URL if needed
        if database_path == ":memory:":
            self.database_url = FLWR_IN_MEMORY_SQLITE_DB_URL
        elif not database_path.startswith("sqlite://"):
            # Treat as file path, convert to absolute and create SQLite URL
            abs_path = Path(database_path).resolve()
            self.database_url = f"sqlite:///{abs_path}"
        else:
            # Already a SQLite URL
            self.database_url = database_path

        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Provide a transactional database session context.

        Yields a SQLAlchemy Session that automatically commits on success or rolls
        back on exceptions. Re-entrant: nested calls reuse the same session. Use for
        multi-statement transactions; prefer `query()` for single statements.

        Yields
        ------
        Session
            SQLAlchemy session. Commits on context exit, rolls back on exceptions.

        Examples
        --------
            with self.session() as session:
                session.execute(text("DELETE FROM t WHERE id = :id"), {"id": 1})
                session.execute(text("INSERT INTO t2 SELECT * FROM t"))
        """
        existing = _current_session.get()

        # Re-entrant: reuse the session if in a session context, no begin()
        if existing is not None:
            yield existing
            return

        if self._session_factory is None:
            raise AttributeError("Database not initialized. Call initialize() first.")

        # Create new session; outermost scope owns the transaction
        session = self._session_factory()
        token = _current_session.set(session)

        try:
            with session.begin():
                yield session
        finally:
            _current_session.reset(token)
            session.close()

    def get_metadata(self) -> MetaData | None:
        """Return the MetaData object for this class.

        Subclasses can override this to provide their SQLAlchemy MetaData.
        The base implementation returns None.

        Returns
        -------
        MetaData | None
            SQLAlchemy MetaData object for this class.
        """
        return None

    def initialize(self, log_queries: bool = False) -> list[str]:
        """Connect to the DB and create tables if needed.

        This method creates the SQLAlchemy engine and session factory,
        and creates tables returned by `get_metadata()`.

        Parameters
        ----------
        log_queries : bool
            Log each query which is executed.

        Returns
        -------
        list[str]
            The list of all tables in the DB.
        """
        # Create engine with SQLite-specific settings
        engine_kwargs: dict[str, Any] = {
            # SQLite needs check_same_thread=False for multi-threaded access
            "connect_args": {"check_same_thread": False}
        }
        self._engine = create_engine(self.database_url, **engine_kwargs)

        # Set SQLite pragmas via event listener for optimal performance and correctness
        event.listen(self._engine, "connect", _set_sqlite_pragmas)

        if log_queries:
            # Set up query logging via event listener
            event.listen(self._engine, "before_cursor_execute", _log_query)

        # Create session factory
        self._session_factory = sessionmaker(bind=self._engine)

        # Create database
        metadata: MetaData | None = self.get_metadata()
        if metadata and self.database_url == FLWR_IN_MEMORY_SQLITE_DB_URL:
            # In-memory databases: create tables directly from SQLAlchemy metadata
            metadata.create_all(self._engine)
        else:
            # File-based databases: use Alembic migrations for schema versioning
            run_migrations(self._engine)

        # Get all table names using inspector
        inspector = inspect(self._engine)
        return inspector.get_table_names()

    def query(
        self,
        query: str,
        data: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return the results as list of dicts.

        TRANSACTION SEMANTICS:
        ----------------------
        If called outside a session context, each call to query() runs in its own
        isolated transaction that is automatically committed. This is suitable for
        single SQL statements.

        If called within a session() context, query() reuses the existing session
        and transaction. This enables atomic multi-query operations:

            with self.session() as session:
                self.query("UPDATE ...", {...})    # Shares same transaction
                self.query("INSERT ...", {...})    # Shares same transaction
                # Both succeed or fail together

        You can also use session.execute() directly for the same effect:

            with self.session() as session:
                session.execute(text("UPDATE ..."), {...})
                session.execute(text("INSERT ..."), {...})

        Parameters
        ----------
        query : str
            SQL query string with named parameter placeholders.
            Use :name syntax for parameters: "SELECT * FROM t WHERE a = :a AND b = :b"
        data : Sequence[dict[str, Any]] | dict[str, Any] | None
            Query parameters using named parameter syntax:
            - Single execution: pass dict, e.g., {"a": value1, "b": value2}
            - Batch execution: pass sequence of dicts, e.g., [{"a": 1}, {"a": 2}]

        Returns
        -------
        list[dict[str, Any]]
            Query results as a list of dictionaries.

        Examples
        --------
        # Single query with named parameters (auto-committed transaction)
        rows = self.query(
            "SELECT * FROM node WHERE node_id = :id AND status = :status",
            {"id": node_id, "status": status}
        )

        # Batch insert with named parameters (auto-committed transaction)
        rows = self.query(
            "INSERT INTO node (node_id, status) VALUES (:id, :status)",
            [{"id": 1, "status": "online"}, {"id": 2, "status": "offline"}]
        )

        # Multi-statement transaction - query() calls share the same session
        with self.session():
            # Both statements succeed or fail together
            self.query("DELETE FROM token_store WHERE active_until < :time", {...})
            self.query("UPDATE run SET status = :status WHERE id = :id", {...})

        # Nested session() - query() calls share the same session
        with self.session():
            self.query("DELETE FROM token_store WHERE active_until < :time", {...})
            with self.session():
                self.query("UPDATE run SET status = :status WHERE id = :id", {...})
        """
        if self._engine is None:
            raise AttributeError(
                "LinkState is not initialized. Call initialize() first."
            )

        if data is None:
            data = {}

        # Clean up whitespace to make the logs nicer
        query = re.sub(r"\s+", " ", query.strip())

        try:
            # Wrap query in text() to enable SQLAlchemy named parameter syntax (:param).
            sql = text(query)

            def execute_and_fetch(session: Session) -> list[dict[str, Any]]:
                """Execute query and fetch results from the given session."""
                # Execute query (results live in database cursor).
                # There is no need to check for batch vs single execution;
                # SQLAlchemy handles both cases automatically.
                result = session.execute(sql, data)

                # Fetch results into Python memory before commit.
                # mappings() returns dict-like rows (works for SELECT and RETURNING).
                if result.returns_rows:  # type: ignore
                    return [dict(row) for row in result.mappings()]

                # For statements without RETURNING (INSERT/UPDATE/DELETE),
                # returns_rows is False, so we return empty list.
                return []

            # Not in a session context, create a new session context for this query
            with self.session() as session:
                return execute_and_fetch(session)

        except SQLAlchemyError as exc:
            log(ERROR, {"query": query, "data": data, "exception": exc})
            raise
