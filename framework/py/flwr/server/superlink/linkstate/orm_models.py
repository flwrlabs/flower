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
"""SQLAlchemy ORM models for LinkState tables."""

from sqlalchemy.orm import DeclarativeBase, Mapped

from flwr.supercore.state.schema.linkstate_tables import create_linkstate_metadata

LINKSTATE_METADATA = create_linkstate_metadata()
NODE_TABLE = LINKSTATE_METADATA.tables["node"]


class LinkStateBase(DeclarativeBase):
    """Base class for LinkState ORM models."""

    metadata = LINKSTATE_METADATA


class LinkStateNode(LinkStateBase):
    """Represent a LinkState node."""

    __table__ = NODE_TABLE
    __mapper_args__ = {"primary_key": [NODE_TABLE.c.node_id]}

    node_id: Mapped[int]
    status: Mapped[str]
    public_key: Mapped[bytes]
