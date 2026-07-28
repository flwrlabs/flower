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
"""SQLAlchemy declarative model definitions for CoreState."""

from sqlalchemy import Float, Index, LargeBinary, MetaData, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class FlwrBase(DeclarativeBase):
    """Base class for Flower OSS state models."""

    metadata = MetaData()


class NonceStore(FlwrBase):
    """Represent stored nonces for replay protection."""

    __tablename__ = "nonce_store"
    __table_args__ = (Index("idx_nonce_store_expires_at", "expires_at"),)

    namespace: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    nonce: Mapped[str] = mapped_column(String, primary_key=True, nullable=False)
    expires_at: Mapped[float] = mapped_column(Float, nullable=False)


class Fab(FlwrBase):
    """Represent stored FAB contents."""

    __tablename__ = "fab"

    fab_hash: Mapped[str] = mapped_column(String, primary_key=True)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    verifications: Mapped[str] = mapped_column(String, nullable=False)
