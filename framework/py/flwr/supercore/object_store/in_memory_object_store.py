# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Flower in-memory ObjectStore implementation."""


import threading
from dataclasses import dataclass

from flwr.proto.message_pb2 import ObjectTree  # pylint: disable=E0611
from flwr.supercore.inflatable.inflatable_object import (
    get_object_id,
    is_valid_sha256_hash,
    iterate_object_tree,
)
from flwr.supercore.inflatable.inflatable_utils import validate_object_content

from .object_store import NoObjectInStoreError, ObjectStore


@dataclass
class ObjectEntry:
    """Data class representing an object entry in the store."""

    content: bytes
    is_available: bool
    child_object_ids: list[str]  # List of child object IDs
    ref_count: int  # Number of references (direct parents) to this object
    runs: set[int]  # Set of run IDs that used this object


class InMemoryObjectStore(ObjectStore):
    """In-memory implementation of the ObjectStore interface."""

    def __init__(self, verify: bool = True) -> None:
        self.verify = verify
        self.store: dict[str, ObjectEntry] = {}
        self.lock_store = threading.RLock()
        # Mapping each run ID to a set of object IDs that are used in that run
        self.run_objects_mapping: dict[int, set[str]] = {}

    def preregister(self, run_id: int, object_tree: ObjectTree) -> list[str]:
        """Identify and preregister missing objects."""
        new_objects = []
        if run_id not in self.run_objects_mapping:
            self.run_objects_mapping[run_id] = set()

        for tree_node in iterate_object_tree(object_tree):
            obj_id = tree_node.object_id
            # Verify object ID format (must be a valid sha256 hash)
            if not is_valid_sha256_hash(obj_id):
                raise ValueError(f"Invalid object ID format: {obj_id}")
            with self.lock_store:
                if obj_id not in self.store:
                    self.store[obj_id] = ObjectEntry(
                        content=b"",  # Initially empty content
                        is_available=False,  # Initially not available
                        child_object_ids=[  # List of child object IDs
                            child.object_id for child in tree_node.children
                        ],
                        ref_count=0,  # Reference count starts at 0
                        runs={run_id},  # Start with the current run ID
                    )

                    # Increment the reference count for all its children
                    # Post-order traversal ensures that children are registered
                    # before parents
                    for child_node in tree_node.children:
                        child_id = child_node.object_id
                        self.store[child_id].ref_count += 1

                    # Add the object ID to the run's mapping
                    self.run_objects_mapping[run_id].add(obj_id)

                    # Add to the list of new objects
                    new_objects.append(obj_id)
                else:
                    # Object is in store, retrieve it
                    obj_entry = self.store[obj_id]

                    # Add to the list of new objects if not available
                    if not obj_entry.is_available:
                        new_objects.append(obj_id)

                    # If the object is already registered but not in this run,
                    # add the run ID to its runs
                    if obj_id not in self.run_objects_mapping[run_id]:
                        obj_entry.runs.add(run_id)
                        self.run_objects_mapping[run_id].add(obj_id)

        return new_objects

    def get_object_tree(self, object_id: str) -> ObjectTree:
        """Get the object tree for a given object ID."""
        with self.lock_store:
            # Raise an exception if there's no object with the given ID
            if not (object_entry := self.store.get(object_id)):
                raise NoObjectInStoreError(
                    f"Object with ID '{object_id}' was not pre-registered."
                )

            # Build the object trees of all children
            try:
                child_trees = [
                    self.get_object_tree(child_id)
                    for child_id in object_entry.child_object_ids
                ]
            except NoObjectInStoreError as e:
                # Raise an error if any child object is missing
                # This indicates an integrity issue
                raise NoObjectInStoreError(
                    f"Object tree for object ID '{object_id}' contains missing "
                    "children. This may indicate a corrupted object store."
                ) from e

            # Create and return the ObjectTree for the current object
            return ObjectTree(object_id=object_id, children=child_trees)

    def put(self, object_id: str, object_content: bytes) -> None:
        """Put an object into the store."""
        if self.verify:
            # Verify object_id and object_content match
            object_id_from_content = get_object_id(object_content)
            if object_id != object_id_from_content:
                raise ValueError(f"Object ID {object_id} does not match content hash")

            # Validate object content
            validate_object_content(content=object_content)

        with self.lock_store:
            # Only allow adding the object if it has been preregistered
            if object_id not in self.store:
                raise NoObjectInStoreError(
                    f"Object with ID '{object_id}' was not pre-registered."
                )

            # Return if object is already present in the store
            if self.store[object_id].is_available:
                return

            # Update the object entry in the store
            self.store[object_id].content = object_content
            self.store[object_id].is_available = True

    def get(self, object_id: str) -> bytes | None:
        """Get an object from the store."""
        with self.lock_store:
            # Check if the object ID is pre-registered
            if object_id not in self.store:
                return None

            # Return content (if not yet available, it will b"")
            return self.store[object_id].content

    def delete(self, object_id: str) -> None:
        """Delete an object and its unreferenced descendants from the store."""
        with self.lock_store:
            # If the object is not in the store, nothing to delete
            if (object_entry := self.store.get(object_id)) is None:
                return

            # Delete the object if it has no references left
            if object_entry.ref_count == 0:
                del self.store[object_id]

                # Remove the object from the run's mapping
                for run_id in object_entry.runs:
                    self.run_objects_mapping[run_id].discard(object_id)

                # Decrease the reference count of its children
                for child_id in object_entry.child_object_ids:
                    self.store[child_id].ref_count -= 1

                    # Recursively try to delete the child object
                    self.delete(child_id)

    def delete_objects_in_run(self, run_id: int) -> None:
        """Delete all objects that were registered in a specific run."""
        with self.lock_store:
            if run_id not in self.run_objects_mapping:
                return
            for object_id in list(self.run_objects_mapping[run_id]):
                # Check if the object is still in the store
                if (object_entry := self.store.get(object_id)) is None:
                    continue

                # Remove the run ID from the object's runs
                object_entry.runs.discard(run_id)

                # Only message objects are allowed to have a `ref_count` of 0,
                # and every message object must have a `ref_count` of 0
                if object_entry.ref_count == 0:
                    # Delete the message object and its unreferenced descendants
                    self.delete(object_id)

            # Remove the run from the mapping
            del self.run_objects_mapping[run_id]

    def clear(self) -> None:
        """Clear the store."""
        with self.lock_store:
            self.store.clear()
            self.run_objects_mapping.clear()

    def __contains__(self, object_id: str) -> bool:
        """Check if an object_id is in the store."""
        with self.lock_store:
            return object_id in self.store

    def __len__(self) -> int:
        """Get the number of objects in the store."""
        with self.lock_store:
            return len(self.store)
