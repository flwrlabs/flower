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
"""Rename conversation table to space.

Revision ID: a8f6aa405dfa
Revises: 10abac2005b4
Create Date: 2026-11-04 21:24:52.061177
"""
from collections.abc import Sequence

from alembic import op

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "a8f6aa405dfa"
down_revision: str | Sequence[str] | None = "10abac2005b4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.rename_table("conversation", "space")


def downgrade() -> None:
    """Downgrade schema."""
    op.rename_table("space", "conversation")
