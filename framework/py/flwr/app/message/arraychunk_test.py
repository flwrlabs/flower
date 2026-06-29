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
"""Unit tests for ArrayChunk."""


import pytest

from flwr.app.message.arraychunk import ArrayChunk


def test_deflate_inflate() -> None:
    """Test deflate/inflate."""
    # Prepare
    data = memoryview(b"some data")
    ac = ArrayChunk(data)

    # Deflate
    ac_deflated = ac.deflate()

    # Inflate
    ac_inflated = ArrayChunk.inflate(ac_deflated)

    # Assert (objects are identical)
    assert ac.object_id == ac_inflated.object_id


def test_inflate_passing_children() -> None:
    """ArrayChunk do not have children."""
    # Prepare
    data = memoryview(b"some data")
    ac = ArrayChunk(data)

    # Deflate
    ac_deflated = ac.deflate()

    # Inflate
    with pytest.raises(ValueError):
        ArrayChunk.inflate(ac_deflated, children={"123": ac})
