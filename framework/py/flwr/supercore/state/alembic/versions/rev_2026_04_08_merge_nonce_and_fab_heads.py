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
"""Merge nonce_store and fab migration heads.

Revision ID: 4c6b1d2e9a7f
Revises: f1a9c6d4b2e1, 33e2f70642b1
Create Date: 2026-04-08 15:10:00.000000
"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "4c6b1d2e9a7f"
down_revision: str | Sequence[str] | None = ("f1a9c6d4b2e1", "33e2f70642b1")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Merge revision only: no schema changes.


def downgrade() -> None:
    """Downgrade schema."""
    # Merge revision only: no schema changes.
