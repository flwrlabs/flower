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
"""Tests for CoreState declarative models."""

from typing import Any

import pytest
from sqlalchemy import Column, Table

from flwr.supercore.state.schema.corestate_models import FlwrBase
from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata


def _server_default(column: Column[Any]) -> str | None:
    """Return a comparable representation of a column server default."""
    if column.server_default is None:
        return None
    return str(getattr(column.server_default, "arg", column.server_default))


def _column_signature(column: Column[Any]) -> tuple[object, ...]:
    """Return the schema-relevant parts of a column."""
    return (
        column.name,
        type(column.type),
        column.nullable,
        column.primary_key,
        column.unique,
        _server_default(column),
        tuple(
            sorted(
                (
                    foreign_key.column.table.name,
                    foreign_key.column.name,
                    foreign_key.ondelete,
                )
                for foreign_key in column.foreign_keys
            )
        ),
    )


def _index_signature(table: Table) -> set[tuple[object, ...]]:
    """Return the schema-relevant parts of table indexes."""
    return {
        (
            index.name,
            tuple(column.name for column in index.columns),
            index.unique,
        )
        for index in table.indexes
    }


def _primary_key_signature(table: Table) -> tuple[str, ...]:
    """Return the primary-key column names for a table."""
    return tuple(column.name for column in table.primary_key.columns)


@pytest.mark.parametrize(
    "table_name", ["nonce_store", "fab", "run_series", "series_context", "series_runs"]
)
def test_declarative_model_matches_core_metadata(table_name: str) -> None:
    """Ensure declarative metadata preserves the Core table schema."""
    core_table = create_corestate_metadata().tables[table_name]
    model_table = FlwrBase.metadata.tables[table_name]

    assert model_table.name == core_table.name
    assert [_column_signature(column) for column in model_table.columns] == [
        _column_signature(column) for column in core_table.columns
    ]
    assert _primary_key_signature(model_table) == _primary_key_signature(core_table)
    assert _index_signature(model_table) == _index_signature(core_table)
