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
"""SQLAlchemy Core Table definitions for ObjectStore."""


from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Integer,
    LargeBinary,
    MetaData,
    PrimaryKeyConstraint,
    String,
    Table,
)


def create_objectstore_metadata() -> MetaData:
    """Create and return MetaData with ObjectStore table definitions."""
    metadata = MetaData()

    # --------------------------------------------------------------------------
    #  Table: objects
    # --------------------------------------------------------------------------
    Table(
        "objects",
        metadata,
        Column("object_id", String, primary_key=True, nullable=True),
        Column("content", LargeBinary),
        Column(
            "is_available",
            Integer,
            nullable=False,
            server_default="0",
        ),
        Column("ref_count", Integer, nullable=False, server_default="0"),
        CheckConstraint("is_available IN (0, 1)", name="ck_objects_is_available"),
    )

    # --------------------------------------------------------------------------
    #  Table: object_children
    # --------------------------------------------------------------------------
    Table(
        "object_children",
        metadata,
        Column(
            "parent_id",
            String,
            ForeignKey("objects.object_id", ondelete="CASCADE"),
            nullable=False,
        ),
        Column(
            "child_id",
            String,
            ForeignKey("objects.object_id", ondelete="CASCADE"),
            nullable=False,
        ),
        PrimaryKeyConstraint("parent_id", "child_id"),
    )

    # --------------------------------------------------------------------------
    #  Table: run_objects
    # --------------------------------------------------------------------------
    Table(
        "run_objects",
        metadata,
        Column("run_id", Integer, nullable=False),
        Column(
            "object_id",
            String,
            ForeignKey("objects.object_id", ondelete="CASCADE"),
            nullable=False,
        ),
        PrimaryKeyConstraint("run_id", "object_id"),
    )

    return metadata
