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
"""Tests for shared SQLAlchemy schema types."""

from datetime import UTC, datetime

from sqlalchemy import Column, MetaData, Table, create_engine, select
from sqlalchemy.schema import CreateTable

from flwr.supercore.state.schema.types import UTCDateTime


def test_utc_datetime_compiles_as_timestamp_on_sqlite() -> None:
    """Ensure the custom type keeps the existing SQLite column schema."""
    engine = create_engine("sqlite:///:memory:")
    table = Table("sample", MetaData(), Column("created_at", UTCDateTime()))

    ddl = str(CreateTable(table).compile(engine))

    assert "created_at TIMESTAMP" in ddl


def test_utc_datetime_preserves_sqlite_timestamp_text() -> None:
    """Ensure SQLite binds keep the previous offset-preserving text format."""
    engine = create_engine("sqlite:///:memory:")
    table = Table("sample", MetaData(), Column("created_at", UTCDateTime()))
    table.create(engine)

    with engine.begin() as conn:
        conn.execute(
            table.insert().values(created_at=datetime(2026, 1, 1, 12, 0, tzinfo=UTC))
        )
        stored_value = conn.exec_driver_sql(
            "SELECT created_at FROM sample"
        ).scalar_one()

    assert stored_value == "2026-01-01 12:00:00+00:00"


def test_utc_datetime_reads_old_sqlite_timestamp_text_as_utc() -> None:
    """Ensure old SQLite timestamp rows read back as UTC-aware datetimes."""
    engine = create_engine("sqlite:///:memory:")
    table = Table("sample", MetaData(), Column("created_at", UTCDateTime()))
    table.create(engine)

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "INSERT INTO sample (created_at) VALUES (?)",
            ("2026-01-01 12:00:00+00:00",),
        )
        stored_value = conn.execute(select(table.c.created_at)).scalar_one()

    assert stored_value == datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
