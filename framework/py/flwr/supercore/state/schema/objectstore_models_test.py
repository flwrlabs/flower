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
"""Tests for ObjectStore declarative models."""

from flwr.supercore.state.schema.objectstore_models import (
    ObjectChild,
    ObjectStoreBase,
    ObjectStoreLock,
    RunObject,
    StoredObject,
)
from flwr.supercore.state.schema.objectstore_tables import create_objectstore_metadata

OBJECTSTORE_TABLE_NAMES = {
    "objects",
    "object_children",
    "run_objects",
    "objectstore_locks",
}


def test_declarative_metadata_covers_all_objectstore_tables() -> None:
    """Ensure ObjectStore metadata is fully defined by model metadata."""
    copied_table_names = set(create_objectstore_metadata().tables)
    model_table_names = set(ObjectStoreBase.metadata.tables)

    assert model_table_names == OBJECTSTORE_TABLE_NAMES
    assert copied_table_names == model_table_names


def test_objectstore_mappers_use_existing_primary_keys_as_identity_keys() -> None:
    """Ensure ORM mappers use durable existing primary keys."""
    assert [column.name for column in StoredObject.__mapper__.primary_key] == [
        "object_id"
    ]
    assert [column.name for column in ObjectChild.__mapper__.primary_key] == [
        "parent_id",
        "child_id",
    ]
    assert [column.name for column in RunObject.__mapper__.primary_key] == [
        "run_id",
        "object_id",
    ]
    assert [column.name for column in ObjectStoreLock.__mapper__.primary_key] == [
        "lock_id"
    ]
