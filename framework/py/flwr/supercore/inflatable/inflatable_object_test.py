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
"""Inflatable test."""


from __future__ import annotations

import hashlib
from unittest.mock import patch

import pytest

from ..constant import HEAD_BODY_DIVIDER, HEAD_VALUE_DIVIDER
from .inflatable_object import (
    InflatableObject,
    UnexpectedObjectContentError,
    _get_object_body,
    _get_object_head,
    add_header_to_object_body,
    get_all_nested_objects,
    get_descendant_object_ids,
    get_object_body,
    get_object_children_ids_from_object_content,
    get_object_head_values_from_object_content,
    get_object_id,
    get_object_type_from_object_content,
    is_valid_sha256_hash,
    no_object_id_recompute,
)
from .inflatable_utils import validate_object_content


class CustomDataClass(InflatableObject):
    """A dummy dataclass to test Inflatable features."""

    def __init__(
        self, data: bytes, children: list[InflatableObject] | None = None
    ) -> None:
        self.data = data
        self._children = children or []

    def deflate(self) -> bytes:
        """Deflate object."""
        obj_body = self.data
        return add_header_to_object_body(object_body=obj_body, obj=self)

    @classmethod
    def inflate(
        cls, object_content: bytes, children: dict[str, InflatableObject] | None = None
    ) -> CustomDataClass:
        """Inflate the object from bytes."""
        children_ids = get_object_children_ids_from_object_content(object_content)
        object_body = get_object_body(object_content, cls)
        input_children = None
        if children_ids:
            if children is None or len(children_ids) != len(children):
                raise ValueError("Invalid children for this object.")
            input_children = [children[child_id] for child_id in children_ids]
        elif children:
            raise ValueError("This object does not have children.")

        return cls(data=object_body, children=input_children)

    @property
    def children(self) -> dict[str, InflatableObject]:
        """Return children objects."""
        return {child.object_id: child for child in self._children}


def test_deflate_and_inflate() -> None:
    """Deflate a custom object and verify its ``object_id``.

    Then inflate it and verify the content is identical to the original object.
    """
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj_b = obj.deflate()

    # assert
    # Class name matches
    assert get_object_type_from_object_content(obj_b) == obj.__class__.__qualname__
    # Content length matches
    assert len(get_object_body(obj_b, CustomDataClass)) == len(data)

    # assert
    # both objects are identical
    assert get_object_id(obj_b) == obj.object_id

    # Inflate and check object payload is the same
    obj_ = CustomDataClass.inflate(obj_b)
    assert obj_.data == obj.data

    # Assert
    # Inflate passing children raises ValueError
    with pytest.raises(ValueError):
        CustomDataClass.inflate(obj_b, children={"1234": obj})


def test_get_object_id() -> None:
    """Test helper function to get object id from bytes."""
    some_bytes = b"hello world"
    expected = hashlib.sha256(some_bytes).hexdigest()
    assert get_object_id(some_bytes) == expected


def test_get_object_body() -> None:
    """Test helper function to extract object body from object content."""
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj_b = obj.deflate()

    assert _get_object_body(obj_b) == data
    assert get_object_body(obj_b, CustomDataClass) == data


@pytest.mark.parametrize(
    "children",
    [
        [],
        [CustomDataClass(b"child1 data")],
        [CustomDataClass(b"child1 data"), CustomDataClass(b"child2 data")],
    ],
)
def test_add_header_to_object_body(children: list[InflatableObject]) -> None:
    """Test helper function that adds the header to the object body."""
    data = b"this is a test"
    obj = CustomDataClass(data, children)
    obj_b = obj.deflate()

    # Expected object head
    head_str = f"%s{HEAD_VALUE_DIVIDER}%s{HEAD_VALUE_DIVIDER}%d" % (
        "CustomDataClass",
        ",".join(child.object_id for child in children),
        len(data),
    )
    exp_obj_head = head_str.encode()
    assert _get_object_head(obj_b) == exp_obj_head

    # Expected object content
    exp_obj_content = exp_obj_head + HEAD_BODY_DIVIDER + data
    assert add_header_to_object_body(data, obj) == exp_obj_content


@pytest.mark.parametrize(
    "children",
    [
        [],
        [CustomDataClass(b"child1 data")],
        [CustomDataClass(b"child1 data"), CustomDataClass(b"child2 data")],
    ],
)
def test_get_head_values_from_object_content(children: list[InflatableObject]) -> None:
    """Test helper function that extracts the values of the object head."""
    # Prepare
    data = b"this is a test"
    obj = CustomDataClass(data, children)
    obj_b = obj.deflate()

    # Execute
    obj_type, children_ids, body_len = get_object_head_values_from_object_content(obj_b)

    # Assert
    assert obj_type == "CustomDataClass"
    assert children_ids == [child.object_id for child in children]
    assert body_len == len(data)


def test_get_object_type_from_object_content() -> None:
    """Test helper function that extracts the name of the class of the deflated
    object."""
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj_b = obj.deflate()

    assert get_object_type_from_object_content(obj_b) == obj.__class__.__qualname__


def test_is_valid_sha256_hash_valid() -> None:
    """Test helper function that checks if a string is a valid SHA256 hash."""
    # Prepare
    valid_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    # Execute & assert
    assert is_valid_sha256_hash(valid_hash)


def test_is_valid_sha256_hash_invalid() -> None:
    """Test helper function that checks if a string is a valid SHA256 hash."""
    # Prepare
    invalid_hash = "invalid_hash"

    # Execute & assert
    assert not is_valid_sha256_hash(invalid_hash)


@pytest.mark.parametrize(
    "children",
    [
        [],
        [CustomDataClass(b"child1 data")],
        [CustomDataClass(b"child1 data"), CustomDataClass(b"child2 data")],
    ],
)
def test_get_desdendants(children: list[InflatableObject]) -> None:
    """Test computing list of object IDs for all descendants."""
    data = b"this is a test"
    obj = CustomDataClass(data, children)

    # Assert
    assert get_descendant_object_ids(obj) == {child.object_id for child in children}


def test_object_content_validator() -> None:
    """Test validator."""
    # A valid message
    data = b"this is a test"
    obj = CustomDataClass(data)
    obj_b = obj.deflate()

    validate_object_content(obj_b)

    # The message has a longer content than what's recorded in the head
    with pytest.raises(UnexpectedObjectContentError):
        validate_object_content(obj_b + b"extra_bytes")

    # The head specifies an object_type that's not supported
    obj_b_wrong_type = obj_b.replace(
        obj.__class__.__qualname__.encode("utf-8"), b"blabla"
    )
    with pytest.raises(UnexpectedObjectContentError):
        validate_object_content(obj_b_wrong_type)

    # The content doesn't have a head-body divider
    parts = obj_b.split(HEAD_BODY_DIVIDER, 1)
    head, body = parts
    with pytest.raises(UnexpectedObjectContentError):
        validate_object_content(head + body)

    # The head does not have three distinct parts
    head_decoded = head.decode(encoding="utf-8")
    head_parts = head_decoded.split(HEAD_VALUE_DIVIDER)
    obj_type, _, body_len = head_parts
    # construct head (but don't pass children part) and object
    obj_wrong_head = (obj_type + HEAD_VALUE_DIVIDER + body_len).encode(encoding="utf-8")
    almost_correct_obj = obj_wrong_head + HEAD_BODY_DIVIDER + body
    with pytest.raises(UnexpectedObjectContentError):
        validate_object_content(almost_correct_obj)

    # The message head carries some children IDs but these
    # aren't valid SHA256
    obj_wrong_head = (
        obj_type + HEAD_VALUE_DIVIDER + "abcd12345" + HEAD_VALUE_DIVIDER + body_len
    ).encode(encoding="utf-8")
    almost_correct_obj = obj_wrong_head + HEAD_BODY_DIVIDER + body
    with pytest.raises(UnexpectedObjectContentError):
        validate_object_content(almost_correct_obj)


def test_get_all_nested_object_ids() -> None:
    """Test getting all nested object IDs."""
    # Prepare
    child1 = CustomDataClass(b"child1 data")
    child2 = CustomDataClass(b"child2 data")
    obj = CustomDataClass(b"this is a test", children=[child1, child2])
    expected_objects = {
        child1.object_id: child1,
        child2.object_id: child2,
        obj.object_id: obj,
    }

    # Execute
    all_objects = get_all_nested_objects(obj)

    # Assert
    assert all_objects == expected_objects
    assert list(all_objects.keys()) == list(expected_objects.keys())


def test_no_object_id_recompute() -> None:
    """Test that no recompute of object ID is done."""
    # Prepare
    obj = CustomDataClass(b"this is a test")
    original_object_id = obj.object_id

    with patch(
        "flwr.supercore.inflatable.inflatable_object.get_object_id",
        side_effect=get_object_id,
    ) as mock_get_object_id:
        # Execute: Access object_id multiple times within the context manager
        with no_object_id_recompute():
            for _ in range(5):
                obj_id = obj.object_id  # Accessing object_id should not recompute it

        # Assert: the mock deflate was called only once
        assert mock_get_object_id.call_count == 1
        assert obj_id == original_object_id

    with patch(
        "flwr.supercore.inflatable.inflatable_object.get_object_id",
        side_effect=get_object_id,
    ) as mock_get_object_id:
        # Execute: Access object_id outside the context manager
        for _ in range(5):
            obj_id = obj.object_id

        # Assert: Accessing object_id outside the context manager recomputes it
        assert mock_get_object_id.call_count == 5
