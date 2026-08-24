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
"""SQLite-based CoreState implementation."""

import secrets
import sqlite3
from logging import DEBUG, WARNING
from typing import cast

from flwr.common import now
from flwr.common.constant import (
    FLWR_APP_TOKEN_LENGTH,
    HEARTBEAT_CLIENTAPP_LEASE,
    HEARTBEAT_INITIAL_GRACE_PERIOD,
)
from flwr.common.logger import log
from flwr.supercore.sqlite_mixin import SqliteMixin
from flwr.supercore.utils import int64_to_uint64, mask_string, uint64_to_int64

from ..object_store import ObjectStore
from .corestate import CoreState

SQL_CREATE_TABLE_TOKEN_STORE = """
CREATE TABLE IF NOT EXISTS token_store (
    run_id                  INTEGER PRIMARY KEY,
    token                   TEXT UNIQUE NOT NULL,
    active_until            REAL
);
"""


class SqliteCoreState(CoreState, SqliteMixin):
    """SQLite-based CoreState implementation."""

    def __init__(self, database_path: str, object_store: ObjectStore) -> None:
        super().__init__(database_path)
        self._object_store = object_store

    @property
    def object_store(self) -> ObjectStore:
        """Return the ObjectStore instance used by this CoreState."""
        return self._object_store

    def get_sql_statements(self) -> tuple[str, ...]:
        """Return SQL statements needed for CoreState tables."""
        return (SQL_CREATE_TABLE_TOKEN_STORE,)

    def create_token(self, run_id: int) -> str | None:
        """Create a token for the given run ID."""
        token = secrets.token_hex(FLWR_APP_TOKEN_LENGTH)  # Generate a random token
        current = now().timestamp()
        active_until = current + HEARTBEAT_INITIAL_GRACE_PERIOD
        query = """
            INSERT INTO token_store (run_id, token, active_until)
            VALUES (:run_id, :token, :active_until);
        """
        data = {
            "run_id": uint64_to_int64(run_id),
            "token": token,
            "active_until": active_until,
        }
        try:
            self.query(query, data)
        except sqlite3.IntegrityError:
            log(
                DEBUG,
                "ClientApp token request denied: run_id=%s already has a token",
                run_id,
            )
            return None  # Token already created for this run ID
        log(
            DEBUG,
            "ClientApp token created: run_id=%s token=%s initial_grace_s=%s",
            run_id,
            mask_string(token),
            HEARTBEAT_INITIAL_GRACE_PERIOD,
        )
        return token

    def verify_token(self, run_id: int, token: str) -> bool:
        """Verify a token for the given run ID."""
        self._cleanup_expired_tokens()
        query = "SELECT token FROM token_store WHERE run_id = :run_id;"
        data = {"run_id": uint64_to_int64(run_id)}
        rows = self.query(query, data)
        if not rows:
            return False
        return cast(str, rows[0]["token"]) == token

    def delete_token(self, run_id: int) -> None:
        """Delete the token for the given run ID."""
        query = "DELETE FROM token_store WHERE run_id = :run_id;"
        data = {"run_id": uint64_to_int64(run_id)}
        self.query(query, data)
        log(DEBUG, "ClientApp token deleted after completion: run_id=%s", run_id)

    def get_run_id_by_token(self, token: str) -> int | None:
        """Get the run ID associated with a given token."""
        self._cleanup_expired_tokens()
        query = "SELECT run_id FROM token_store WHERE token = :token;"
        data = {"token": token}
        rows = self.query(query, data)
        if not rows:
            return None
        return int64_to_uint64(rows[0]["run_id"])

    def acknowledge_app_heartbeat(self, token: str) -> bool:
        """Acknowledge an app heartbeat with the provided token."""
        # Clean up expired tokens
        self._cleanup_expired_tokens()

        # Update the active_until field
        current = now().timestamp()
        active_until = current + HEARTBEAT_CLIENTAPP_LEASE
        query = """
            UPDATE token_store
            SET active_until = :active_until
            WHERE token = :token
            RETURNING run_id;
        """
        data = {"active_until": active_until, "token": token}
        rows = self.query(query, data)
        if rows:
            log(
                DEBUG,
                "ClientApp heartbeat accepted: run_id=%s token=%s lease_s=%s",
                int64_to_uint64(rows[0]["run_id"]),
                mask_string(token),
                HEARTBEAT_CLIENTAPP_LEASE,
            )
        else:
            log(
                WARNING,
                "ClientApp heartbeat rejected: token=%s is unknown or expired",
                mask_string(token),
            )
        return len(rows) > 0

    def _cleanup_expired_tokens(self) -> None:
        """Remove expired tokens and perform additional cleanup.

        This method is called before token operations to ensure integrity.
        Subclasses can override `_on_tokens_expired` to add custom cleanup logic.
        """
        current = now().timestamp()

        with self.conn:
            # Delete expired tokens and get their run_ids and active_until timestamps
            query = """
                DELETE FROM token_store
                WHERE active_until < :current
                RETURNING run_id, active_until;
            """
            rows = self.conn.execute(query, {"current": current}).fetchall()
            expired_records = [
                (int64_to_uint64(row["run_id"]), row["active_until"]) for row in rows
            ]

            # Hook for subclasses
            if expired_records:
                log(
                    WARNING,
                    "Expired %s ClientApp token(s): run_ids=%s",
                    len(expired_records),
                    [run_id for run_id, _ in expired_records],
                )
                self._on_tokens_expired(expired_records)

    def _on_tokens_expired(self, expired_records: list[tuple[int, float]]) -> None:
        """Handle cleanup of expired tokens.

        Override in subclasses to add custom cleanup logic.

        Parameters
        ----------
        expired_records : list[tuple[int, float]]
            List of tuples containing (run_id, active_until timestamp)
            for expired tokens.
        """
