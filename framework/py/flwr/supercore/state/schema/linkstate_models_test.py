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
"""Tests for LinkState declarative models."""

from typing import Any, cast

import pytest
from sqlalchemy import Column, Table, UniqueConstraint

from flwr.supercore.state.schema.linkstate_models import (
    Context,
    LinkStateBase,
    LogsTable,
    MessageIns,
    MessageRes,
    Node,
    Run,
)
from flwr.supercore.state.schema.linkstate_tables import create_linkstate_metadata

LINKSTATE_TABLE_NAMES = {
    "node",
    "run",
    "logs",
    "context",
    "message_ins",
    "message_res",
}


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


def _unique_constraint_signature(table: Table) -> set[tuple[str, ...]]:
    """Return the column names for every unique constraint."""
    return {
        tuple(column.name for column in constraint.columns)
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }


@pytest.mark.parametrize("table_name", sorted(LINKSTATE_TABLE_NAMES))
def test_declarative_metadata_matches_linkstate_metadata(table_name: str) -> None:
    """Ensure declarative metadata preserves the LinkState table schema."""
    linkstate_table = create_linkstate_metadata().tables[table_name]
    model_table = LinkStateBase.metadata.tables[table_name]

    assert [_column_signature(column) for column in model_table.columns] == [
        _column_signature(column) for column in linkstate_table.columns
    ]
    assert _index_signature(model_table) == _index_signature(linkstate_table)
    assert _unique_constraint_signature(model_table) == _unique_constraint_signature(
        linkstate_table
    )


def test_declarative_metadata_covers_all_linkstate_tables() -> None:
    """Ensure declarative metadata covers every LinkState table."""
    assert set(create_linkstate_metadata().tables) == LINKSTATE_TABLE_NAMES
    assert set(LinkStateBase.metadata.tables) == LINKSTATE_TABLE_NAMES


@pytest.mark.parametrize(
    ("model", "identity_column"),
    [
        (Node, "node_id"),
        (Run, "run_id"),
        (Context, "run_id"),
        (MessageIns, "message_id"),
        (MessageRes, "message_id"),
    ],
)
def test_mapper_uses_existing_unique_identity(
    model: type[LinkStateBase], identity_column: str
) -> None:
    """Ensure mapped tables use an existing unique identity column."""
    assert [column.name for column in model.__mapper__.primary_key] == [identity_column]


def test_logs_table_remains_unmapped_without_safe_identity() -> None:
    """Ensure logs stays a table instead of using a nullable composite identity."""
    mapped_table_names = {
        cast(Table, mapper.local_table).name
        for mapper in LinkStateBase.registry.mappers
    }

    assert LogsTable.primary_key.columns.values() == []
    assert "logs" not in mapped_table_names
