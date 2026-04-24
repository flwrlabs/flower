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
"""Tests for runtime version metadata interceptors."""


from collections import namedtuple
from typing import cast
from unittest import TestCase
from unittest.mock import Mock

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.common.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
)
from flwr.common.runtime_version import build_runtime_version_metadata
from flwr.proto.serverappio_pb2 import GetNodesRequest  # pylint: disable=E0611
from flwr.supercore.interceptors import (
    RuntimeVersionClientInterceptor,
    RuntimeVersionServerInterceptor,
)

_ClientCallDetails = namedtuple(
    "_ClientCallDetails",
    ["method", "timeout", "metadata", "credentials", "wait_for_ready", "compression"],
)


class _HandlerCallDetails:
    def __init__(
        self,
        method: str,
        invocation_metadata: tuple[tuple[str, str | bytes], ...],
    ) -> None:
        self.method = method
        self.invocation_metadata = invocation_metadata


def _make_unary_handler() -> grpc.RpcMethodHandler:
    def _handler(_request: GrpcMessage, _context: grpc.ServicerContext) -> str:
        return "ok"

    return grpc.unary_unary_rpc_method_handler(_handler)


class TestRuntimeVersionClientInterceptor(TestCase):
    """Unit tests for RuntimeVersionClientInterceptor."""

    def test_attach_runtime_version_headers(self) -> None:
        """The interceptor should add the shared version metadata keys."""
        interceptor = RuntimeVersionClientInterceptor(component_name="simulation")
        details = _ClientCallDetails(
            method="/flwr.proto.ServerAppIo/GetNodes",
            timeout=None,
            metadata=((FLWR_PACKAGE_NAME_METADATA_KEY, "old"), ("x-test", "value")),
            credentials=None,
            wait_for_ready=None,
            compression=None,
        )
        captured: dict[str, list[tuple[str, str | bytes]]] = {}

        def continuation(
            client_call_details: grpc.ClientCallDetails,
            _request: GrpcMessage,
        ) -> str:
            captured["metadata"] = list(client_call_details.metadata or [])
            return "ok"

        response = interceptor.intercept_unary_unary(
            continuation=continuation,
            client_call_details=details,
            request=GetNodesRequest(run_id=1),
        )

        self.assertEqual(response, "ok")
        metadata = dict(captured["metadata"])
        self.assertEqual(metadata["x-test"], "value")
        self.assertIn(FLWR_PACKAGE_NAME_METADATA_KEY, metadata)
        self.assertIn(FLWR_PACKAGE_VERSION_METADATA_KEY, metadata)
        self.assertEqual(metadata[FLWR_COMPONENT_NAME_METADATA_KEY], "simulation")


class TestRuntimeVersionServerInterceptor(TestCase):
    """Unit tests for RuntimeVersionServerInterceptor."""

    def setUp(self) -> None:
        self.interceptor = RuntimeVersionServerInterceptor(
            connection_name="flwr-simulation <-> SuperLink ServerAppIo API",
            local_metadata=build_runtime_version_metadata(
                "superlink",
                package_name_value="flwr",
                package_version_value="1.29.0",
            ),
        )

    def test_missing_metadata_is_tolerated(self) -> None:
        """Missing runtime metadata should pass during rollout."""
        intercepted = self.interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails("/flwr.proto.ServerAppIo/GetNodes", ()),
        )

        response = cast(str, intercepted.unary_unary(GetNodesRequest(run_id=1), Mock()))
        self.assertEqual(response, "ok")

    def test_invalid_metadata_is_rejected(self) -> None:
        """Malformed runtime metadata should be rejected."""
        context = Mock()
        context.abort.side_effect = grpc.RpcError()
        intercepted = self.interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/GetNodes",
                (
                    (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
                    (FLWR_PACKAGE_VERSION_METADATA_KEY, "main"),
                    (FLWR_COMPONENT_NAME_METADATA_KEY, "simulation"),
                ),
            ),
        )

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.FAILED_PRECONDITION,
            "Invalid Flower version metadata for "
            "flwr-simulation <-> SuperLink ServerAppIo API. "
            "Peer Flower version metadata is invalid: 'main'.",
        )

    def test_incompatible_metadata_is_rejected(self) -> None:
        """Different major.minor versions should be rejected."""
        context = Mock()
        context.abort.side_effect = grpc.RpcError()
        intercepted = self.interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/GetNodes",
                (
                    (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
                    (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.30.1"),
                    (FLWR_COMPONENT_NAME_METADATA_KEY, "simulation"),
                ),
            ),
        )

        with self.assertRaises(grpc.RpcError):
            intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        context.abort.assert_called_once_with(
            grpc.StatusCode.FAILED_PRECONDITION,
            "Incompatible Flower version for "
            "flwr-simulation <-> SuperLink ServerAppIo API.\n"
            "Local superlink version 1.29.0 only accepts peers from the same "
            "major.minor release, but received simulation version 1.30.1.",
        )

    def test_compatible_metadata_is_accepted(self) -> None:
        """Same major.minor versions should pass."""
        intercepted = self.interceptor.intercept_service(
            lambda _: _make_unary_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/GetNodes",
                (
                    (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
                    (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.7"),
                    (FLWR_COMPONENT_NAME_METADATA_KEY, "simulation"),
                ),
            ),
        )

        response = cast(str, intercepted.unary_unary(GetNodesRequest(run_id=1), Mock()))
        self.assertEqual(response, "ok")
