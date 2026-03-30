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
"""Tests for SqlMixin."""


import unittest

from sqlalchemy import Column, Integer, MetaData, Table
from sqlalchemy.exc import IntegrityError

from .sql_mixin import SqlMixin


class DummyDbSqlAlchemy(SqlMixin):
    """Simple subclass for testing SqlMixin behavior with SQLAlchemy."""

    def get_metadata(self) -> MetaData:
        """Return MetaData with test table definition."""
        metadata = MetaData()
        Table(
            "test",
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("value", Integer),
        )
        return metadata

    def cleanup_negative_values(self) -> int:
        """Delete rows with negative values and return count deleted."""
        rows = self.query("SELECT COUNT(*) AS cnt FROM test WHERE value < 0")
        count: int = rows[0]["cnt"]
        if count > 0:
            self.query("DELETE FROM test WHERE value < 0")
        return count

    def insert_and_cleanup(self, value: int) -> int:
        """Insert a value and cleanup negative values atomically."""
        with self.session():
            self.query("INSERT INTO test (value) VALUES (:value)", {"value": value})
            deleted = self.cleanup_negative_values()
        return deleted


class TestSqlMixin(unittest.TestCase):
    """Test SqlMixin session and transaction behavior."""

    def setUp(self) -> None:
        """Set up test database for each test."""
        self.db = DummyDbSqlAlchemy(":memory:")
        self.db.initialize()

    def test_session_commits_all_queries_atomitcally(self) -> None:
        """Test that all queries in a session are committed as a single transaction."""
        # Insert initial test data
        self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": 100})

        # Multiple query() calls within a session should share the same transaction
        with self.db.session():
            # First query: insert a value
            self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": 200})

            # Second query: verify the value was inserted
            rows = self.db.query(
                "SELECT value FROM test WHERE value = :value", {"value": 200}
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["value"], 200)

            # Third query: update the value
            self.db.query(
                "UPDATE test SET value = :new WHERE value = :old",
                {"old": 200, "new": 300},
            )

            # Fourth query: verify the update
            rows = self.db.query(
                "SELECT value FROM test WHERE value = :value", {"value": 300}
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["value"], 300)

        # Verify all changes were committed
        rows = self.db.query("SELECT value FROM test ORDER BY value")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["value"], 100)
        self.assertEqual(rows[1]["value"], 300)

    def test_nested_sessions(self) -> None:
        """Test that nested session() calls reuse the same session."""
        # Nested session contexts should reuse the same session
        with self.db.session() as outer_session:
            self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": 101})

            with self.db.session() as inner_session:
                # Inner session should be the same object as outer session
                self.assertIs(inner_session, outer_session)

                # Insert in nested context
                self.db.query(
                    "INSERT INTO test (value) VALUES (:value)", {"value": 201}
                )

            # After inner context, can still use outer session
            self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": 301})

        # Verify all inserts were committed as one transaction
        rows = self.db.query("SELECT value FROM test ORDER BY value")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["value"], 101)
        self.assertEqual(rows[1]["value"], 201)
        self.assertEqual(rows[2]["value"], 301)

    def test_query_without_session(self) -> None:
        """Test that query() works independently when not in a session context."""
        # Each query() call should be its own transaction
        self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": 211})
        self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": 212})
        self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": 213})

        # Verify all inserts were committed independently
        rows = self.db.query("SELECT value FROM test ORDER BY value")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["value"], 211)
        self.assertEqual(rows[1]["value"], 212)
        self.assertEqual(rows[2]["value"], 213)

    def test_session_rollback_on_exception(self) -> None:
        """Test that exceptions in a session cause rollback for all nested queries."""
        # Insert initial test data
        self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": 10})

        # Exception should rollback all nested query() calls
        with self.assertRaises(ValueError):
            with self.db.session():
                # First query: insert a value
                self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": 20})

                # Second query: verify the value was inserted (within transaction)
                rows = self.db.query(
                    "SELECT value FROM test WHERE value = :value", {"value": 20}
                )
                self.assertEqual(len(rows), 1)

                # Raise a simulated error to trigger rollback before final commit
                raise ValueError("Simulated business logic error")

        # Verify the transaction was rolled back
        rows = self.db.query("SELECT value FROM test")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], 10)

    def test_session_rollback_on_database_error(self) -> None:
        """Test that database errors cause rollback for all nested queries."""
        # Insert initial test data
        self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": 111})

        # Database error should rollback all nested query() calls
        with self.assertRaises(IntegrityError):
            with self.db.session():
                # First query: insert a value with explicit id
                self.db.query(
                    "INSERT INTO test (id, value) VALUES (:id, :value)",
                    {"id": 999, "value": 200},
                )

                # Second query: verify the value was inserted (within transaction)
                rows = self.db.query(
                    "SELECT value FROM test WHERE value = :value", {"value": 200}
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["value"], 200)

                # Third query: attempt to insert duplicate primary key
                # (will raise IntegrityError)
                self.db.query(
                    "INSERT INTO test (id, value) VALUES (:id, :value)",
                    {"id": 999, "value": 300},  # Same id=999, violates PRIMARY KEY
                )

        # Verify the entire transaction was rolled back (neither 200 nor 300 exist)
        rows = self.db.query("SELECT value FROM test ORDER BY value")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], 111)

    def test_session_reuse_across_methods(self) -> None:
        """Test that session is reused when method A calls method B within a session."""
        # Insert initial test data with negative values
        self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": -1})
        self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": -2})

        # Call a method that wraps query() and another method in a session
        deleted = self.db.insert_and_cleanup(value=500)

        # Verify the cleanup happened
        self.assertEqual(deleted, 2)

        # Verify only positive value remains (both insert & cleanup committed together)
        rows = self.db.query("SELECT value FROM test ORDER BY value")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["value"], 500)

    def test_session_reuse_across_methods_rollback(self) -> None:
        """Test that rollback affects both method A and method B queries."""
        # Insert initial test data
        self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": -1})
        self.db.query("INSERT INTO test (value) VALUES (:value)", {"value": 108})

        # Method A creates session, calls query, then calls method B which also queries
        # If an error occurs after method B, everything should rollback
        with self.assertRaises(ValueError):
            with self.db.session():
                # Insert a new value (method A's query)
                self.db.query(
                    "INSERT INTO test (value) VALUES (:value)", {"value": 200}
                )

                # Call method B which deletes negative values
                deleted = self.db.cleanup_negative_values()
                self.assertEqual(deleted, 1)

                # Verify within transaction: -1 is gone, 200 exists
                rows = self.db.query("SELECT value FROM test ORDER BY value")
                self.assertEqual(len(rows), 2)  # 108 and 200
                self.assertEqual(rows[0]["value"], 108)
                self.assertEqual(rows[1]["value"], 200)

                # Simulate error after method B completes
                raise ValueError("Error after cleanup")

        # Verify complete rollback: original state restored (-1 and 108 both exist)
        rows = self.db.query("SELECT value FROM test ORDER BY value")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["value"], -1)  # Deletion was rolled back
        self.assertEqual(rows[1]["value"], 108)
