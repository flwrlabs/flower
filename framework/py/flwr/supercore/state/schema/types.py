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

from sqlalchemy import TIMESTAMP
from sqlalchemy.types import TypeDecorator


class UTCDateTime(TypeDecorator[datetime]):  # pylint: disable=too-many-ancestors
    """Store timezone-aware datetimes consistently across supported dialects."""

    impl = TIMESTAMP(timezone=True)
    cache_ok = True

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
