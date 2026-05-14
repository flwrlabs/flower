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
"""Merge flwr state heads.

Revision ID: 154388986f5f
Revises: aac61834ee69, 8b40f767ddcb
Create Date: 2026-05-14 14:06:19.316073
"""

from collections.abc import Sequence


# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "154388986f5f"
down_revision: str | Sequence[str] | None = ("aac61834ee69", "8b40f767ddcb")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # No-op merge migration: this revision only merges multiple heads.


def downgrade() -> None:
    """Downgrade schema."""
    # No-op merge migration: this revision only merges multiple heads.
