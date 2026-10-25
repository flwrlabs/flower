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


import uuid

from flwr.app import DEFAULT_TTL, Context, Metadata, RecordDict
from flwr.app.message import make_message
from flwr.client import Client, ClientFnExt
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Status,
)
from flwr.common.constant import MessageTypeLegacy
from flwr.compat.common import recorddict_compat as compat
from flwr.supercore.date import now

from .message_handler import handle_legacy_message_from_msgtype


class ClientWithoutProps(Client):
    """Client not implementing get_properties."""

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Get empty parameters of the client with 'Success' status."""
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Simulate successful training, return no parameters, no metrics."""
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
            num_examples=1,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Simulate successful evaluation, return no metrics."""
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=1.0,
            num_examples=1,
            metrics={},
        )


class ClientWithProps(Client):
    """Client implementing get_properties."""

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """Get fixed properties of the client with 'Success' status."""
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={"str_prop": "val", "int_prop": 1},
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Get empty parameters of the client with 'Success' status."""
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Simulate successful training, return no parameters, no metrics."""
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=Parameters(tensors=[], tensor_type=""),
            num_examples=1,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Simulate successful evaluation, return no metrics."""
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
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
