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
# ===============================================================================
"""Tests for the Flower CLI gRPC client interceptor."""

from collections import namedtuple
from unittest import TestCase
from unittest.mock import Mock

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.proto.control_pb2 import ListRunsRequest  # pylint: disable=E0611
from flwr.supercore.constant import FLWR_CLIENT_METADATA_KEY

from .cli_client_interceptor import CliClientInterceptor

_ClientCallDetails = namedtuple(
    "_ClientCallDetails",
    ["method", "timeout", "metadata", "credentials", "wait_for_ready", "compression"],
)


def _make_call_details(
    metadata: tuple[tuple[str, str | bytes], ...] = (),
) -> _ClientCallDetails:
    return _ClientCallDetails(
        method="/flwr.proto.Control/ListRuns",
        timeout=None,
        metadata=metadata,
        credentials=None,
        wait_for_ready=None,
        compression=None,
    )


class TestCliClientInterceptor(TestCase):
    """Unit tests for CliClientInterceptor."""

    def test_adds_client_metadata(self) -> None:
        """The interceptor should identify every CLI RPC."""
        interceptor = CliClientInterceptor()
        details = _make_call_details((("x-test", "value"),))
        call = Mock(spec=grpc.Call)
        captured: dict[str, list[tuple[str, str | bytes]]] = {}

        def continuation(
            client_call_details: grpc.ClientCallDetails,
            _request: GrpcMessage,
        ) -> Mock:
            captured["metadata"] = list(client_call_details.metadata or [])
            return call

        response = interceptor.intercept_unary_unary(
            continuation,
            details,
            ListRunsRequest(),
        )

        self.assertIs(response, call)
        self.assertEqual(
            captured["metadata"],
            [("x-test", "value"), (FLWR_CLIENT_METADATA_KEY, "cli")],
        )

    def test_replaces_existing_client_metadata(self) -> None:
        """The interceptor should keep the CLI value authoritative for the call."""
        interceptor = CliClientInterceptor()
        details = _make_call_details(
            ((FLWR_CLIENT_METADATA_KEY, "web_ui"),),
        )
        captured: dict[str, list[tuple[str, str | bytes]]] = {}

        def continuation(
            client_call_details: grpc.ClientCallDetails,
            _request: GrpcMessage,
        ) -> Mock:
            captured["metadata"] = list(client_call_details.metadata or [])
            return Mock(spec=grpc.Call)

        interceptor.intercept_unary_stream(
            continuation,
            details,
            ListRunsRequest(),
        )

        self.assertEqual(captured["metadata"], [(FLWR_CLIENT_METADATA_KEY, "cli")])
