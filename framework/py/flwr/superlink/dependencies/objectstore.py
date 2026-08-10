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
"""FastAPI ObjectStore dependency for SuperLink routers."""

from __future__ import annotations

from typing import cast

from fastapi import Request

from flwr.supercore.error import ApiErrorCode, FlowerError
from flwr.supercore.object_store import ObjectStore, ObjectStoreFactory


def get_objectstore(request: Request) -> ObjectStore:
    """Return the SuperLink ObjectStore for the current request."""
    objectstore_factory = cast(
        ObjectStoreFactory | None,
        getattr(request.app.state, "objectstore_factory", None),
    )
    if objectstore_factory is None:
        raise FlowerError(
            ApiErrorCode.OBJECTSTORE_NOT_INITIALIZED,
            "SuperLink ObjectStoreFactory is not initialized.",
        )

    return objectstore_factory.store()
