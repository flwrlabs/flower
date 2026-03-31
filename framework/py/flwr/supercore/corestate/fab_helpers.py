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
"""Helpers for in-memory FAB storage semantics."""


import hashlib
from collections.abc import Mapping, MutableMapping
from types import TracebackType
from typing import Protocol

from flwr.common.typing import Fab


class LockLike(Protocol):
    """Protocol for lock objects used with `with` statements."""

    def __enter__(self) -> bool:
        """Enter the lock context and return acquisition result."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the lock context."""


def _validate_and_clone_fab(fab: Fab) -> tuple[str, Fab]:
    """Validate hash and return a canonical FAB copy."""
    fab_hash = hashlib.sha256(fab.content).hexdigest()
    if fab.hash_str and fab.hash_str != fab_hash:
        raise ValueError(
            f"FAB hash mismatch: provided {fab.hash_str}, computed {fab_hash}"
        )
    return fab_hash, Fab(
        hash_str=fab_hash,
        content=fab.content,
        verifications=dict(fab.verifications),
    )


def _clone_fab(fab: Fab) -> Fab:
    """Return a defensive FAB copy."""
    return Fab(
        hash_str=fab.hash_str,
        content=fab.content,
        verifications=dict(fab.verifications),
    )


def store_fab_locked(
    lock: LockLike,
    store: MutableMapping[str, Fab],
    fab: Fab,
) -> str:
    """Store a FAB under canonical content hash while holding the lock."""
    fab_hash, canonical_fab = _validate_and_clone_fab(fab)
    with lock:
        # Keep launch behavior: last write wins for metadata under the same content
        # hash.
        store[fab_hash] = canonical_fab
    return fab_hash


def get_fab_locked(
    lock: LockLike, store: Mapping[str, Fab], fab_hash: str
) -> Fab | None:
    """Return a defensive FAB copy while holding the lock."""
    with lock:
        if (fab := store.get(fab_hash)) is None:
            return None
        # Launch tradeoff: do not recompute content hash on reads; rely on write-time
        # validation and hash-addressed lookup.
        return _clone_fab(fab)
