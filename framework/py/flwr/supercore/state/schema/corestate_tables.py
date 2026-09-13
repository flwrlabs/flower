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
"""Compatibility wrapper for CoreState SQLAlchemy metadata."""

from sqlalchemy import MetaData

from flwr.supercore.state.schema.corestate_models import FlwrBase


def create_corestate_metadata() -> MetaData:
    """Create and return MetaData with CoreState table definitions."""
    metadata = MetaData()
    for table in FlwrBase.metadata.tables.values():
        table.to_metadata(metadata)
    return metadata
