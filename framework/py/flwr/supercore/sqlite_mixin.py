# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Mixin providing common SQLite connection and initialization logic."""


import re
import sqlite3
from abc import ABC
from collections.abc import Sequence
from logging import DEBUG, ERROR
from typing import Any

from flwr.common.logger import log

DictOrTuple = tuple[Any, ...] | dict[str, Any]


class SqliteMixin(ABC):
    """Mixin providing common SQLite connection and initialization logic."""

    def __init__(self, database_path: str) -> None:
        self.database_path = database_path
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Get the SQLite connection."""
        if self._conn is None:
            raise AttributeError("Database not initialized. Call initialize() first.")
        return self._conn

    def get_sql_statements(self) -> tuple[str, ...]:
        """Return SQL statements for this class.

        Subclasses can override this to provide their SQL CREATE statements.
        The base implementation returns an empty tuple.

        Returns
        -------
        tuple[str, ...]
            SQL CREATE TABLE/INDEX statements for this class.
        """
        return ()

    def initialize(self, log_queries: bool = False) -> list[tuple[str]]:
        """Connect to the DB, enable FK support, and create tables if needed.

        This method executes SQL statements returned by `get_sql_statements()`.

        Parameters
        ----------
        log_queries : bool
            Log each query which is executed.

        Returns
        -------
        list[tuple[str]]
            The list of all tables in the DB.

        Examples
        --------
        Override `get_sql_statements()` in your subclass:

        .. code:: python

            def get_sql_statements(self) -> tuple[str, ...]:
                return (
                    SQL_CREATE_TABLE_FOO,
                    SQL_CREATE_TABLE_BAR,
                )

        To include parent SQL statements, call super():

        .. code:: python

            def get_sql_statements(self) -> tuple[str, ...]:
                return super().get_sql_statements() + (
                    SQL_CREATE_TABLE_FOO,
                    SQL_CREATE_TABLE_BAR,
                )
        """
        self._conn = sqlite3.connect(self.database_path, timeout=600.0)
        # Enable Write-Ahead Logging (WAL) for better concurrency
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA synchronous = NORMAL;")
        self._conn.execute("PRAGMA busy_timeout = 600000;")
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute("PRAGMA cache_size = -64000;")  # 64MB cache
        self._conn.execute("PRAGMA temp_store = MEMORY;")  # In-memory temp tables
        self._conn.execute("PRAGMA mmap_size = 268435456;")  # 256MB memory-mapped I/O
        self._conn.row_factory = dict_factory

        if log_queries:
            self._conn.set_trace_callback(lambda q: log(DEBUG, q))

        # Create tables and indexes
        cur = self._conn.cursor()
        for sql in self.get_sql_statements():
            cur.execute(sql)
        res = cur.execute("SELECT name FROM sqlite_schema;")
        return res.fetchall()

    def query(
        self,
        query: str,
        data: Sequence[DictOrTuple] | DictOrTuple | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a SQL query and return the results as list of dicts."""
        if self._conn is None:
            raise AttributeError("LinkState is not initialized.")

        if data is None:
            data = []

        # Clean up whitespace to make the logs nicer
        query = re.sub(r"\s+", " ", query)

        try:
            with self._conn:
                if (
                    len(data) > 0
                    and isinstance(data, (tuple | list))
                    and isinstance(data[0], (tuple | dict))
                ):
                    rows = self._conn.executemany(query, data)
                else:
                    rows = self._conn.execute(query, data)

                # Extract results before committing to support
                #   INSERT/UPDATE ... RETURNING
                # style queries
                result = rows.fetchall()
        except KeyError as exc:
            log(ERROR, {"query": query, "data": data, "exception": exc})

        return result


def dict_factory(
    cursor: sqlite3.Cursor,
    row: sqlite3.Row,
) -> dict[str, Any]:
    """Turn SQLite results into dicts.

    Less efficent for retrival of large amounts of data but easier to use.
    """
    fields = [column[0] for column in cursor.description]
    return dict(zip(fields, row, strict=True))
