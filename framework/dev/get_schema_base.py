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
"""SQLAlchemy Base for schema visualization tools (paracelsus, ERD generation, etc.).

This file is for development/documentation purposes only and is not part of the
production build. It combines all state table schemas into a single declarative base
for tools like paracelsus to generate schema diagrams.
"""

from sqlalchemy import MetaData
from sqlalchemy.orm import declarative_base

from flwr.supercore.state.schema.corestate_tables import create_corestate_metadata
from flwr.supercore.state.schema.linkstate_tables import create_linkstate_metadata
from flwr.supercore.state.schema.objectstore_tables import create_objectstore_metadata

combined_metadata = MetaData()

linkstate_metadata = create_linkstate_metadata()
corestate_metadata = create_corestate_metadata()
objectstore_metadata = create_objectstore_metadata()

# Copy tables into a single MetaData namespace
for md in (linkstate_metadata, corestate_metadata, objectstore_metadata):
    for table in md.tables.values():
        table.tometadata(combined_metadata)

Base = declarative_base(metadata=combined_metadata)

# EE tables (optional, only available when the EE module is installed)
try:
    from flwr.ee.state.alembic.tables import create_ee_metadata

    ee_combined_metadata = MetaData()
    for table in create_ee_metadata().tables.values():
        table.tometadata(ee_combined_metadata)
    EEBase = declarative_base(metadata=ee_combined_metadata)
except ImportError:
    EEBase = None
