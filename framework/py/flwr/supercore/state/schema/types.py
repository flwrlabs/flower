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
"""Shared SQLAlchemy schema types for state tables."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import TIMESTAMP, BigInteger
from sqlalchemy.types import TypeDecorator

from flwr.supercore.utils import int64_to_uint64, uint64_to_int64


class UInt64(TypeDecorator[int]):  # pylint: disable=too-many-ancestors,abstract-method
    """Store unsigned 64-bit identifiers using signed BIGINT bit patterns."""

    impl = BigInteger
    cache_ok = True

    @property
    def python_type(self) -> type[int]:
        """Return the Python type represented by this column type."""
        return int

    def process_bind_param(self, value: int | None, _dialect: Any) -> int | None:
        """Validate and convert an unsigned identifier before binding."""
        if value is None:
            return None
        if not isinstance(value, int) or not 0 <= value < (1 << 64):
            raise ValueError("UInt64 requires an integer in [0, 2**64).")
        return uint64_to_int64(value)

    def process_result_value(self, value: int | None, _dialect: Any) -> int | None:
        """Convert a stored signed identifier back to its unsigned value."""
        return int64_to_uint64(value) if value is not None else None


class UTCDateTime(
    TypeDecorator[datetime]
):  # pylint: disable=too-many-ancestors,abstract-method
    """Store timezone-aware datetimes consistently across supported dialects."""

    impl = TIMESTAMP(timezone=True)
    cache_ok = True

    @property
    def python_type(self) -> type[datetime]:
        """Return the Python type represented by this column type."""
        return datetime

    def process_bind_param(
        self, value: datetime | None, _dialect: Any
    ) -> datetime | None:
        """Pass through non-SQLite bind values to the wrapped timestamp type."""
        return value

    def process_result_value(
        self, value: datetime | None, _dialect: Any
    ) -> datetime | None:
        """Pass through non-SQLite result values from the wrapped timestamp type."""
        return value

    def bind_processor(self, dialect: Any) -> Any:
        """Return a bind processor preserving the previous SQLite text format."""
        if dialect.name != "sqlite":
            return super().bind_processor(dialect)

        def process(value: datetime | None) -> str | None:
            if value is None:
                return None
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            else:
                value = value.astimezone(UTC)
            return value.isoformat(sep=" ")

        return process

    def result_processor(self, dialect: Any, coltype: Any) -> Any:
        """Return a result processor producing UTC-aware datetimes on SQLite."""
        if dialect.name != "sqlite":
            return super().result_processor(dialect, coltype)

        def process(value: datetime | str | None) -> datetime | None:
            if value is None:
                return None
            if isinstance(value, str):
                value = datetime.fromisoformat(value)
            if value.tzinfo is None:
                return value.replace(tzinfo=UTC)
            return value.astimezone(UTC)

        return process
