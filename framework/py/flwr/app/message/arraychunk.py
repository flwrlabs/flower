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
"""ArrayChunk."""


from __future__ import annotations

from dataclasses import dataclass

from flwr.supercore.inflatable.inflatable_object import (
    InflatableObject,
    add_header_to_object_body,
    get_object_body,
)


@dataclass
class ArrayChunk(InflatableObject):
    """ArrayChunk type."""

    data: memoryview

    def deflate(self) -> bytes:
        """Deflate the ArrayChunk."""
        return add_header_to_object_body(object_body=self.data, obj=self)

    @classmethod
    def inflate(
        cls, object_content: bytes, children: dict[str, InflatableObject] | None = None
    ) -> ArrayChunk:
        """Inflate an ArrayChunk from bytes.

        Parameters
        ----------
        object_content : bytes
            The deflated object content of the ArrayChunk.

        children : Optional[dict[str, InflatableObject]] (default: None)
            Must be ``None``. ``ArrayChunk`` does not support child objects.
            Providing any children will raise a ``ValueError``.

        Returns
        -------
        ArrayChunk
            The inflated ArrayChunk.
        """
        if children:
            raise ValueError("`ArrayChunk` objects do not have children.")

        obj_body = get_object_body(object_content, cls)
        return cls(data=memoryview(obj_body))
