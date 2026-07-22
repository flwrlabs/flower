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
"""Merge flwr migration heads.

Revision ID: cfda9accf4ec
Revises: c17f3d9b8a42, 0341f1ecc9c2
Create Date: 2026-07-19 21:02:26.729102
"""
from collections.abc import Sequence

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "cfda9accf4ec"
down_revision: str | Sequence[str] | None = ("c17f3d9b8a42", "0341f1ecc9c2")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
