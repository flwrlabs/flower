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
"""Unit tests for inflatable_utils.py."""


from unittest.mock import Mock

import pytest

from .inflatable_object import get_object_tree
from .inflatable_object_test import CustomDataClass
from .inflatable_utils import (
    inflatable_class_registry,
    pull_and_inflate_object_from_tree,
    pull_objects,
    push_object_contents_from_iterable,
)


def test_pull_and_inflate_object_from_tree() -> None:
    """Test pulling and inflating an object from a tree of CustomDataClass objects."""
    # Prepare: Create a tree of CustomDataClass objects
    grandchild = CustomDataClass(b"grandchild")
    child1 = CustomDataClass(b"child1", children=[grandchild])
    child2 = CustomDataClass(b"child2")
    root = CustomDataClass(b"root", children=[child1, child2])

    # Prepare: Mock the functions
    store = {
        grandchild.object_id: grandchild.deflate(),
        child1.object_id: child1.deflate(),
        child2.object_id: child2.deflate(),
        root.object_id: root.deflate(),
    }
    mock_pull_object = Mock()
    mock_pull_object.side_effect = lambda x: store[x]
    mock_confirm_message_received = Mock()

    # Execute: Call the function under test
    inflatable_class_registry[CustomDataClass.__qualname__] = CustomDataClass
    result = pull_and_inflate_object_from_tree(
        object_tree=get_object_tree(root),
        pull_object_fn=mock_pull_object,
        confirm_object_received_fn=mock_confirm_message_received,
        return_type=CustomDataClass,
    )
    del inflatable_class_registry[CustomDataClass.__qualname__]

    # Assert: Mypy or Pylance should recognize `result` as `CustomDataClass`
    assert result.data == root.data  # This should pass type checking

    # Assert: Check that the result matches the root object
    assert result.object_id == root.object_id
    mock_pull_object.assert_called()
    mock_confirm_message_received.assert_called_once_with(root.object_id)


def test_push_object_contents_from_iterable() -> None:
    """Test pushing object contents from an iterable."""
    # Prepare: Create a list of CustomDataClass objects
    fake_pairs = [(f"fake_obj_id_{i}", b"fake_content_{i}") for i in range(10)]
    # Prepare: Mock the functions
    mock_push_object = Mock()

    # Execute: Call the function under test
    push_object_contents_from_iterable(
        object_contents=fake_pairs,
        push_object_fn=mock_push_object,
    )

    # Assert: Check that the push function was called with the correct objects
    for obj_id, obj_content in fake_pairs:
        mock_push_object.assert_any_call(obj_id, obj_content)


def test_pull_objects_reraises_exception() -> None:
    """Test that exceptions from pull_object_fn are re-raised."""
    # Prepare: Create a list of fake object IDs
    fake_object_ids = [f"fake_obj_id_{i}" for i in range(10)]

    # Prepare: Mock the pull function to raise an exception
    test_exception = RuntimeError("Test exception from pull_object_fn")
    side_effects = [b"fake_content"] * 2 + [test_exception]
    mock_pull_object = Mock(side_effect=side_effects)

    # Execute and Assert: Verify the exception is re-raised
    with pytest.raises(RuntimeError, match=str(test_exception)):
        pull_objects(
            object_ids=fake_object_ids,
            pull_object_fn=mock_pull_object,
        )


def test_push_object_contents_from_iterable_reraises_exception() -> None:
    """Test that exceptions from push_object_fn are re-raised."""
    # Prepare: Create a list of fake object pairs
    fake_pairs = [(f"fake_obj_id_{i}", b"fake_content_{i}") for i in range(10)]

    # Prepare: Mock the push function to raise an exception
    test_exception = RuntimeError("Test exception from push_object_fn")
    side_effects = [None, None, test_exception]
    mock_push_object = Mock(side_effect=side_effects)

    # Execute and Assert: Verify the exception is re-raised
    with pytest.raises(RuntimeError, match=str(test_exception)):
        push_object_contents_from_iterable(
            object_contents=fake_pairs,
            push_object_fn=mock_push_object,
        )
