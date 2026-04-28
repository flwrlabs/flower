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
"""Client-side message handler tests."""


import unittest
import uuid
from copy import copy

from flwr.client import Client
from flwr.client.typing import ClientFnExt
from flwr.common import (
    DEFAULT_TTL,
    Code,
    Context,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Metadata,
    Parameters,
    RecordDict,
    Status,
    now,
)
from flwr.common import recorddict_compat as compat
from flwr.common import typing
from flwr.common.constant import MessageTypeLegacy
from flwr.common.message import make_message

from .message_handler import handle_legacy_message_from_msgtype, validate_out_message


class ClientWithoutProps(Client):
    """Client not implementing get_properties."""

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Get empty parameters of the client with 'Success' status."""
        return GetParametersRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Simulate successful training, return no parameters, no metrics."""
        return FitRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
            num_examples=1,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Simulate successful evaluation, return no metrics."""
        return EvaluateRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            loss=1.0,
            num_examples=1,
            metrics={},
        )


class ClientWithProps(Client):
    """Client implementing get_properties."""

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """Get fixed properties of the client with 'Success' status."""
        return GetPropertiesRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            properties={"str_prop": "val", "int_prop": 1},
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Get empty parameters of the client with 'Success' status."""
        return GetParametersRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Simulate successful training, return no parameters, no metrics."""
        return FitRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
            num_examples=1,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Simulate successful evaluation, return no metrics."""
        return EvaluateRes(
            status=typing.Status(code=typing.Code.OK, message="Success"),
            loss=1.0,
            num_examples=1,
            metrics={},
        )


def _get_client_fn(client: Client) -> ClientFnExt:
    def client_fn(contex: Context) -> Client:  # pylint: disable=unused-argument
        return client

    return client_fn


def test_client_without_get_properties() -> None:
    """Test client implementing get_properties."""
    # Prepare
    client = ClientWithoutProps()
    recorddict = compat.getpropertiesins_to_recorddict(GetPropertiesIns({}))
    message = make_message(
        metadata=Metadata(
            run_id=123,
            message_id=str(uuid.uuid4()),
            group_id="some group ID",
            src_node_id=0,
            dst_node_id=1123,
            reply_to_message_id="",
            created_at=now().timestamp(),
            ttl=DEFAULT_TTL,
            message_type=MessageTypeLegacy.GET_PROPERTIES,
        ),
        content=recorddict,
    )

    # Execute
    actual_msg = handle_legacy_message_from_msgtype(
        client_fn=_get_client_fn(client),
        message=message,
        context=Context(
            run_id=2234, node_id=1123, node_config={}, state=RecordDict(), run_config={}
        ),
    )

    # Assert
    expected_get_properties_res = GetPropertiesRes(
        status=Status(
            code=Code.GET_PROPERTIES_NOT_IMPLEMENTED,
            message="Client does not implement `get_properties`",
        ),
        properties={},
    )
    expected_rs = compat.getpropertiesres_to_recorddict(expected_get_properties_res)
    expected_msg = make_message(
        metadata=Metadata(
            run_id=123,
            message_id="",
            group_id="some group ID",
            src_node_id=1123,
            dst_node_id=0,
            reply_to_message_id=message.metadata.message_id,
            created_at=now().timestamp(),
            # Computed based on Message(..., reply_to=[message])
            ttl=actual_msg.metadata.ttl,
            message_type=MessageTypeLegacy.GET_PROPERTIES,
        ),
        content=expected_rs,
    )

    assert actual_msg.content == expected_msg.content
    # metadata.created_at will differ so let's exclude it from checks
    attrs = vars(actual_msg.metadata)
    attrs_keys = list(attrs.keys())
    attrs_keys.remove("_created_at")
    # metadata.created_at will differ so let's exclude it from checks
    for attr in attrs_keys:
        assert getattr(actual_msg.metadata, attr) == getattr(
            expected_msg.metadata, attr
        )

    # Ensure the message created last has a higher timestamp
    assert actual_msg.metadata.created_at < expected_msg.metadata.created_at


def test_client_with_get_properties() -> None:
    """Test client not implementing get_properties."""
    # Prepare
    client = ClientWithProps()
    recorddict = compat.getpropertiesins_to_recorddict(GetPropertiesIns({}))
    message = make_message(
        metadata=Metadata(
            run_id=123,
            message_id=str(uuid.uuid4()),
            group_id="some group ID",
            src_node_id=0,
            dst_node_id=1123,
            reply_to_message_id="",
            created_at=now().timestamp(),
            ttl=DEFAULT_TTL,
            message_type=MessageTypeLegacy.GET_PROPERTIES,
        ),
        content=recorddict,
    )

    # Execute
    actual_msg = handle_legacy_message_from_msgtype(
        client_fn=_get_client_fn(client),
        message=message,
        context=Context(
            run_id=2234, node_id=1123, node_config={}, state=RecordDict(), run_config={}
        ),
    )

    # Assert
    expected_get_properties_res = GetPropertiesRes(
        status=Status(
            code=Code.OK,
            message="Success",
        ),
        properties={"str_prop": "val", "int_prop": 1},
    )
    expected_rs = compat.getpropertiesres_to_recorddict(expected_get_properties_res)
    expected_msg = make_message(
        metadata=Metadata(
            run_id=123,
            message_id="",
            group_id="some group ID",
            src_node_id=1123,
            dst_node_id=0,
            reply_to_message_id=message.metadata.message_id,
            created_at=now().timestamp(),
            # Computed based on Message(..., reply_to=[message])
            ttl=actual_msg.metadata.ttl,
            message_type=MessageTypeLegacy.GET_PROPERTIES,
        ),
        content=expected_rs,
    )

    assert actual_msg.content == expected_msg.content
    attrs = vars(actual_msg.metadata)
    attrs_keys = list(attrs.keys())
    attrs_keys.remove("_created_at")
    # metadata.created_at will differ so let's exclude it from checks
    for attr in attrs_keys:
        assert getattr(actual_msg.metadata, attr) == getattr(
            expected_msg.metadata, attr
        )

    # Ensure the message created last has a higher timestamp
    assert actual_msg.metadata.created_at < expected_msg.metadata.created_at


class TestMessageValidation(unittest.TestCase):
    """Test message validation."""

    def setUp(self) -> None:
        """Set up the message validation."""
        # Common setup for tests
        self.in_metadata = Metadata(
            run_id=123,
            message_id="qwerty",
            src_node_id=10,
            dst_node_id=20,
            reply_to_message_id="",
            group_id="group1",
            created_at=now().timestamp(),
            ttl=DEFAULT_TTL,
            message_type="evaluate",
        )

        self.valid_out_metadata = Metadata(
            run_id=123,
            message_id="",
            src_node_id=20,
            dst_node_id=10,
            reply_to_message_id="qwerty",
            group_id="group1",
            created_at=now().timestamp(),
            ttl=DEFAULT_TTL,
            message_type="evaluate",
        )
        self.common_content = RecordDict()

    def test_valid_message(self) -> None:
        """Test a valid message."""
        # Prepare
        valid_message = make_message(
            metadata=self.valid_out_metadata, content=RecordDict()
        )
        valid_message.metadata.__dict__["_message_id"] = valid_message.object_id

        # Assert
        self.assertTrue(validate_out_message(valid_message, self.in_metadata))

    def test_invalid_message_run_id(self) -> None:
        """Test invalid messages."""
        # Prepare
        msg = make_message(metadata=self.valid_out_metadata, content=RecordDict())

        # Execute
        invalid_metadata_list: list[Metadata] = []
        attrs = list(vars(self.valid_out_metadata).keys())
        for attr in attrs:
            if attr == "_delivered_at":
                continue
            if attr == "_ttl":  # Skip configurable ttl
                continue
            # Make an invalid metadata
            invalid_metadata = copy(self.valid_out_metadata)
            value = getattr(invalid_metadata, attr)
            if isinstance(value, int):
                value = 999
            elif isinstance(value, str):
                value = "999"
            elif isinstance(value, float):
                if attr == "_created_at":
                    # make it be in 1h the past
                    value = value - 3600
            setattr(invalid_metadata, attr, value)
            # Add to list
            invalid_metadata_list.append(invalid_metadata)

        # Assert
        for invalid_metadata in invalid_metadata_list:
            msg.__dict__["_metadata"] = invalid_metadata
            self.assertFalse(validate_out_message(msg, self.in_metadata))
