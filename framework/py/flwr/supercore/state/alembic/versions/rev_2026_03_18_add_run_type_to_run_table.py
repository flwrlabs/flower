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
"""Add run_type and federation_config to run table.

Revision ID: c8f4f6e2c1ad
Revises: 8e65d8ae60b0
Create Date: 2026-03-18 15:10:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from flwr.supercore.constant import RunType

# pylint: disable=no-member

# revision identifiers, used by Alembic.
revision: str = "c8f4f6e2c1ad"
down_revision: str | Sequence[str] | None = "8e65d8ae60b0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "run",
        sa.Column(
            "run_type",
            sa.String(),
            nullable=False,
            server_default=RunType.SERVER_APP,
        ),
    )
    op.add_column("run", sa.Column("federation_config", sa.String(), nullable=True))
    op.execute(
        sa.text(
            "UPDATE run SET run_type = :run_type WHERE length(federation_options) > 0"
        ).bindparams(run_type=RunType.SIMULATION)
    )
    op.drop_column("run", "federation_options")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "run", sa.Column("federation_options", sa.LargeBinary(), nullable=True)
    )
    op.drop_column("run", "federation_config")
    op.drop_column("run", "run_type")
