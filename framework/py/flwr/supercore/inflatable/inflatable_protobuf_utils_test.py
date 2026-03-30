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
"""Tests for InflatableObject helpers to communicate with gRPC servicers."""


import unittest
from itertools import product
from unittest.mock import Mock

import numpy as np
from parameterized import parameterized

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.proto.message_pb2 import (  # pylint: disable=E0611
    PullObjectRequest,
    PullObjectResponse,
    PushObjectRequest,
    PushObjectResponse,
)
from flwr.proto.node_pb2 import Node  # pylint: disable=E0611
from flwr.supercore.inflatable.inflatable_utils import (
    ObjectIdNotPreregisteredError,
    ObjectUnavailableError,
    pull_objects,
    push_objects,
)

from .inflatable_object import get_all_nested_objects
from .inflatable_protobuf_utils import (
    make_pull_object_fn_protobuf,
    make_push_object_fn_protobuf,
)

base_cases = [
    ({"a": ConfigRecord({"a": 123, "b": 123})},),  # Single w/o children
    (
        {
            "a": ConfigRecord({"a": 123, "b": 123}),
            "b": ConfigRecord({"a": 123, "b": 123}),
        },
    ),  # Two identical
    ({"a": ConfigRecord({"a": 123, "b": 123}), "b": ConfigRecord()},),  # Different
    (
        {
            "a": ConfigRecord({"a": 123, "b": 123}),
            "b": ArrayRecord([np.array([1, 2]), np.array([3, 4])]),
        },
    ),  # Mixed with children
]


class TestInflatableStubHelpers(unittest.TestCase):  # pylint: disable=R0902
    """Test helpers to push and pull InflatableObjects."""

    def setUp(self) -> None:
        """Initialize mock stub and server interceptor."""
        self.mock_store: dict[str, bytes] = {}
        self.mock_stub = Mock()

        def push_object(request: PushObjectRequest) -> PushObjectResponse:
            if request.object_id not in self.mock_store:
                return PushObjectResponse(stored=False)
            self.mock_store[request.object_id] = request.object_content
            return PushObjectResponse(stored=True)

        def pull_object(request: PullObjectRequest) -> PullObjectResponse:
            obj_content = self.mock_store.get(request.object_id, b"")
            return PullObjectResponse(
                object_content=obj_content,
                object_found=request.object_id in self.mock_store,
                object_available=obj_content != b"",
            )

        self.mock_stub.PushObject.side_effect = push_object
        self.mock_stub.PullObject.side_effect = pull_object
        node = Node(node_id=456)
        run_id = 1234
        self.push_object_fn = make_push_object_fn_protobuf(
            self.mock_stub.PushObject, node, run_id
        )
        self.pull_object_fn = make_pull_object_fn_protobuf(
            self.mock_stub.PullObject, node, run_id
        )

    @parameterized.expand(product([case[0] for case in base_cases], [True, False]))  # type: ignore
    def test_push_objects(
        self,
        records: dict[str, ArrayRecord | ConfigRecord | MetricRecord],
        filter_by_obj_ids: bool,
    ) -> None:
        """Test pushing objects with push_objects helper function."""
        # Prepare
        obj = Message(RecordDict(records), dst_node_id=123, message_type="query")
        # Prepare: Pre-register all objects
        all_objects = get_all_nested_objects(obj)
        expected_obj_count = len(all_objects)
        for obj_id in all_objects:
            self.mock_store[obj_id] = b""
        # Prepare: Filter by object IDs if specified
        object_ids_to_push = None
        if filter_by_obj_ids:
            object_ids_to_push = set(list(all_objects.keys())[:2])  # Take first two
            expected_obj_count = len(object_ids_to_push)

        # Execute
        push_objects(
            all_objects, self.push_object_fn, object_ids_to_push=object_ids_to_push
        )

        # Assert: Expected number of objects were pushed
        num_pushed_objects = sum(b != b"" for b in self.mock_store.values())
        assert self.mock_stub.PushObject.call_count == expected_obj_count
        assert num_pushed_objects == expected_obj_count

    @parameterized.expand(base_cases)  # type: ignore
    def test_pull_objects_success(
        self,
        records: dict[str, ArrayRecord | ConfigRecord | MetricRecord],
    ) -> None:
        """Test pulling objects with pull_objects helper function."""
        # Prepare
        obj = Message(RecordDict(records), dst_node_id=123, message_type="query")
        # Prepare: Pre-register all objects
        all_objects = get_all_nested_objects(obj)
        expected_obj_count = len(all_objects)
        for obj_id in all_objects:
            self.mock_store[obj_id] = b""
        # Prepare: Push objects
        push_objects(all_objects, self.push_object_fn, keep_objects=True)

        # Execute
        pulled_objects = pull_objects(list(all_objects.keys()), self.pull_object_fn)

        # Assert: Expected number of objects were pulled
        assert self.mock_stub.PullObject.call_count == expected_obj_count
        assert pulled_objects == {k: v.deflate() for k, v in all_objects.items()}

    @parameterized.expand(base_cases)  # type: ignore
    def test_pull_objects_no_preregistration_failure(
        self,
        records: dict[str, ArrayRecord | ConfigRecord | MetricRecord],
    ) -> None:
        """Test pulling objects without pre-registering them."""
        # Prepare
        obj = Message(RecordDict(records), dst_node_id=123, message_type="query")
        # Prepare: Pre-register all objects
        all_objects = get_all_nested_objects(obj)
        all_object_ids = list(all_objects.keys())
        all_objects.pop(
            obj.object_id
        )  # Remove one object to simulate no pre-registration
        for obj_id in all_objects:
            self.mock_store[obj_id] = b""
        # Prepare: Push objects
        push_objects(all_objects, self.push_object_fn)

        # Execute and assert
        with self.assertRaises(ObjectIdNotPreregisteredError):
            _ = pull_objects(all_object_ids, self.pull_object_fn)

    @parameterized.expand(base_cases)  # type: ignore
    def test_pull_objects_exceeding_max_time_failure(
        self,
        records: dict[str, ArrayRecord | ConfigRecord | MetricRecord],
    ) -> None:
        """Test pulling objects exceeding max time."""
        # Prepare
        obj = Message(RecordDict(records), dst_node_id=123, message_type="query")
        # Prepare: Pre-register all objects
        all_objects = get_all_nested_objects(obj)
        all_object_ids = list(all_objects.keys())
        all_objects.pop(obj.object_id)  # Remove one object to simulate unavailability
        for obj_id in all_object_ids:
            self.mock_store[obj_id] = b""
        # Prepare: Push objects
        push_objects(all_objects, self.push_object_fn)

        # Execute
        with self.assertRaises(ObjectUnavailableError):
            _ = pull_objects(
                all_object_ids,
                self.pull_object_fn,
                max_time=0.001,
                initial_backoff=0.0015,
            )

    @parameterized.expand(base_cases)  # type: ignore
    def test_pull_objects_exceeding_max_tries_failure(
        self,
        records: dict[str, ArrayRecord | ConfigRecord | MetricRecord],
    ) -> None:
        """Test pulling objects exceeding max tries."""
        # Prepare
        obj = Message(RecordDict(records), dst_node_id=123, message_type="query")
        # Prepare: Pre-register all objects
        all_objects = get_all_nested_objects(obj)
        all_object_ids = list(all_objects.keys())
        all_objects.pop(obj.object_id)  # Remove one object to simulate unavailability
        for obj_id in all_object_ids:
            self.mock_store[obj_id] = b""
        # Prepare: Push objects
        push_objects(all_objects, self.push_object_fn)

        # Execute
        with self.assertRaises(ObjectUnavailableError):
            _ = pull_objects(
                all_object_ids,
                self.pull_object_fn,
                max_tries_per_object=3,
                initial_backoff=0.0001,  # Small backoff to trigger retries quickly
            )
