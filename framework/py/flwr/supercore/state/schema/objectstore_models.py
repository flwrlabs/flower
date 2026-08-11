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
"""SQLAlchemy declarative model definitions for ObjectStore."""

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Integer,
    LargeBinary,
    MetaData,
    PrimaryKeyConstraint,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class ObjectStoreBase(DeclarativeBase):
    """Base class for ObjectStore state models."""

    metadata = MetaData()


class StoredObject(ObjectStoreBase):
    """Represent stored object content and reference counts."""

    __tablename__ = "objects"
    __table_args__ = (
        CheckConstraint("is_available IN (0, 1)", name="ck_objects_is_available"),
        CheckConstraint("ref_count >= 0", name="ck_objects_ref_count_nonnegative"),
    )

    object_id: Mapped[str | None] = mapped_column(
        String, primary_key=True, nullable=True
    )
    content: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    is_available: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="0"
    )
    ref_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")


class ObjectChild(ObjectStoreBase):
    """Represent a parent-child relationship between stored objects."""

    __tablename__ = "object_children"
    __table_args__ = (PrimaryKeyConstraint("parent_id", "child_id"),)

    parent_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("objects.object_id", ondelete="CASCADE"),
        nullable=False,
    )
    child_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("objects.object_id", ondelete="CASCADE"),
        nullable=False,
    )


class RunObject(ObjectStoreBase):
    """Represent an object registered for a run."""

    __tablename__ = "run_objects"
    __table_args__ = (PrimaryKeyConstraint("run_id", "object_id"),)

    run_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    object_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("objects.object_id", ondelete="CASCADE"),
        nullable=False,
    )


class ObjectStoreLock(ObjectStoreBase):
    """Represent an ObjectStore transaction lock."""

    __tablename__ = "objectstore_locks"
    __table_args__ = (PrimaryKeyConstraint("lock_id"),)

    lock_id: Mapped[str] = mapped_column(String, nullable=False)
    lock_value: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
