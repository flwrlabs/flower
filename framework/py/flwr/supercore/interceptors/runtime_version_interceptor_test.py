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
from collections.abc import Iterator
from unittest import TestCase
from unittest.mock import Mock, patch

import grpc
from google.protobuf.message import Message as GrpcMessage

from flwr.proto.serverappio_pb2 import GetNodesRequest  # pylint: disable=E0611
from flwr.supercore.constant import (
    FLWR_COMPONENT_NAME_METADATA_KEY,
    FLWR_PACKAGE_NAME_METADATA_KEY,
    FLWR_PACKAGE_VERSION_METADATA_KEY,
    VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY,
)
from flwr.supercore.interceptors import (
    RuntimeVersionClientInterceptor,
    RuntimeVersionServerInterceptor,
)
from flwr.supercore.runtime_version_compatibility import RuntimeVersionMetadata

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


def _make_unary_stream_handler() -> grpc.RpcMethodHandler:
    def _handler(
        _request: GrpcMessage, _context: grpc.ServicerContext
    ) -> Iterator[str]:
        yield "a"
        yield "b"

    return grpc.unary_stream_rpc_method_handler(_handler)


class TestRuntimeVersionClientInterceptor(TestCase):
    """Unit tests for RuntimeVersionClientInterceptor."""

    def _make_call(self) -> Mock:
        call = Mock(spec=grpc.Call)
        call.trailing_metadata.return_value = ()
        return call

    def test_attach_runtime_version_headers(self) -> None:
        """The interceptor should add the shared version metadata keys."""
        interceptor = RuntimeVersionClientInterceptor(component_name="simulation")
        details = _ClientCallDetails(
            method="/flwr.proto.ServerAppIo/GetNodes",
            timeout=None,
            metadata=(("x-test", "value"),),
            credentials=None,
            wait_for_ready=None,
            compression=None,
        )
        captured: dict[str, list[tuple[str, str | bytes]]] = {}
        call = self._make_call()

        def continuation(
            client_call_details: grpc.ClientCallDetails,
            _request: GrpcMessage,
        ) -> Mock:
            captured["metadata"] = list(client_call_details.metadata or [])
            return call

        response = interceptor.intercept_unary_unary(
            continuation=continuation,
            client_call_details=details,
            request=GetNodesRequest(run_id=1),
        )

        self.assertIs(response, call)
        metadata = dict(captured["metadata"])
        self.assertEqual(metadata["x-test"], "value")
        self.assertIn(FLWR_PACKAGE_NAME_METADATA_KEY, metadata)
        self.assertIn(FLWR_PACKAGE_VERSION_METADATA_KEY, metadata)
        self.assertEqual(metadata[FLWR_COMPONENT_NAME_METADATA_KEY], "simulation")

    def test_attach_runtime_version_headers_rejects_preexisting_runtime_keys(
        self,
    ) -> None:
        """Fail fast when runtime-version keys are already present outbound."""
        interceptor = RuntimeVersionClientInterceptor(component_name="simulation")
        details = _ClientCallDetails(
            method="/flwr.proto.ServerAppIo/GetNodes",
            timeout=None,
            metadata=((FLWR_PACKAGE_NAME_METADATA_KEY, "old"), ("x-test", "value")),
            credentials=None,
            wait_for_ready=None,
            compression=None,
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "gRPC metadata already contains runtime version keys: flwr-package-name",
        ):
            interceptor.intercept_unary_unary(
                continuation=lambda _details, _request: self._make_call(),
                client_call_details=details,
                request=GetNodesRequest(run_id=1),
            )


class TestRuntimeVersionServerInterceptor(TestCase):
    """Unit tests for RuntimeVersionServerInterceptor."""

    def setUp(self) -> None:
        """Create a baseline interceptor for each test."""
        self.interceptor = RuntimeVersionServerInterceptor(
            connection_name="flwr-simulation <-> SuperLink ServerAppIo API",
            local_metadata=RuntimeVersionMetadata.from_local_component(
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

        context = Mock()
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        self.assertEqual(response, "ok")
        context.set_trailing_metadata.assert_not_called()

    def test_unparseable_peer_version_is_warned(self) -> None:
        """Explicit unparseable peer versions should be warned."""
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

        context = Mock()
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        self.assertEqual(response, "ok")
        context.set_trailing_metadata.assert_called_once()

    def test_incompatible_metadata_is_warned(self) -> None:
        """Different major.minor versions should still be warned."""
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

        context = Mock()
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        self.assertEqual(response, "ok")
        context.set_trailing_metadata.assert_called_once()

    def test_compatible_metadata_is_accepted(self) -> None:
        """Compatible peer version should not set trailing metadata for unary
        handlers."""
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

        context = Mock()
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        self.assertEqual(response, "ok")
        context.set_trailing_metadata.assert_not_called()

    def test_unary_stream_incompatible_metadata_is_warned(self) -> None:
        """Incompatible peer version should set trailing metadata for stream
        handlers."""
        intercepted = self.interceptor.intercept_service(
            lambda _: _make_unary_stream_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/PullTaskIns",
                (
                    (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
                    (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.30.1"),
                    (FLWR_COMPONENT_NAME_METADATA_KEY, "simulation"),
                ),
            ),
        )

        context = Mock()
        responses = list(intercepted.unary_stream(GetNodesRequest(run_id=1), context))
        self.assertEqual(responses, ["a", "b"])
        context.send_initial_metadata.assert_called_once()
        context.set_trailing_metadata.assert_called_once()

    def test_unary_stream_compatible_metadata_is_accepted(self) -> None:
        """Compatible peer version should not set trailing metadata for stream
        handlers."""
        intercepted = self.interceptor.intercept_service(
            lambda _: _make_unary_stream_handler(),
            _HandlerCallDetails(
                "/flwr.proto.ServerAppIo/PullTaskIns",
                (
                    (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
                    (FLWR_PACKAGE_VERSION_METADATA_KEY, "1.29.7"),
                    (FLWR_COMPONENT_NAME_METADATA_KEY, "simulation"),
                ),
            ),
        )

        context = Mock()
        responses = list(intercepted.unary_stream(GetNodesRequest(run_id=1), context))
        self.assertEqual(responses, ["a", "b"])
        context.set_trailing_metadata.assert_not_called()


class TestRuntimeVersionClientInterceptorUnaryStream(TestCase):
    """Unit tests for RuntimeVersionClientInterceptor.intercept_unary_stream."""

    def test_attach_runtime_version_headers_unary_stream(self) -> None:
        """The interceptor should add version metadata headers for stream calls."""
        interceptor = RuntimeVersionClientInterceptor(component_name="simulation")
        details = _ClientCallDetails(
            method="/flwr.proto.Fleet/PullTaskIns",
            timeout=None,
            metadata=(("x-test", "value"),),
            credentials=None,
            wait_for_ready=None,
            compression=None,
        )
        captured: dict[str, list[tuple[str, str | bytes]]] = {}

        def continuation(
            client_call_details: grpc.ClientCallDetails,
            _request: GrpcMessage,
        ) -> Mock:
            captured["metadata"] = list(client_call_details.metadata or [])
            mock_call = Mock()
            mock_call.initial_metadata.return_value = ()
            mock_call.trailing_metadata.return_value = ()
            mock_call.__iter__ = Mock(return_value=iter(["msg1", "msg2"]))
            return mock_call

        responses = list(
            interceptor.intercept_unary_stream(
                continuation=continuation,
                client_call_details=details,
                request=GetNodesRequest(run_id=1),
            )
        )

        self.assertEqual(responses, ["msg1", "msg2"])
        metadata = dict(captured["metadata"])
        self.assertEqual(metadata["x-test"], "value")
        self.assertIn(FLWR_PACKAGE_NAME_METADATA_KEY, metadata)
        self.assertIn(FLWR_PACKAGE_VERSION_METADATA_KEY, metadata)
        self.assertEqual(metadata[FLWR_COMPONENT_NAME_METADATA_KEY], "simulation")

    def test_log_incompatibility_from_initial_metadata(self) -> None:
        """The interceptor should log stream incompatibilities before completion."""
        interceptor = RuntimeVersionClientInterceptor(component_name="simulation")
        details = _ClientCallDetails(
            method="/flwr.proto.Fleet/PullTaskIns",
            timeout=None,
            metadata=(),
            credentials=None,
            wait_for_ready=None,
            compression=None,
        )
        mock_call = Mock()
        mock_call.initial_metadata.return_value = (
            (
                VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY,
                "runtime mismatch",
            ),
        )
        mock_call.trailing_metadata.return_value = ()
        mock_call.__iter__ = Mock(return_value=iter(["msg1"]))

        with patch(
            "flwr.supercore.interceptors.runtime_version_interceptor.log"
        ) as log_mock:
            responses = list(
                interceptor.intercept_unary_stream(
                    continuation=lambda _details, _request: mock_call,
                    client_call_details=details,
                    request=GetNodesRequest(run_id=1),
                )
            )

        self.assertEqual(responses, ["msg1"])
        mock_call.initial_metadata.assert_called_once()
        mock_call.add_done_callback.assert_called_once()
        log_mock.assert_called_once()

    def test_fallback_to_trailing_metadata_when_initial_metadata_is_empty(self) -> None:
        """The interceptor should keep the trailing-metadata fallback for streams."""
        interceptor = RuntimeVersionClientInterceptor(component_name="simulation")
        details = _ClientCallDetails(
            method="/flwr.proto.Fleet/PullTaskIns",
            timeout=None,
            metadata=(),
            credentials=None,
            wait_for_ready=None,
            compression=None,
        )
        mock_call = Mock()
        mock_call.initial_metadata.return_value = ()
        mock_call.trailing_metadata.return_value = (
            (
                VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY,
                "runtime mismatch",
            ),
        )
        mock_call.__iter__ = Mock(return_value=iter(["msg1"]))

        with patch(
            "flwr.supercore.interceptors.runtime_version_interceptor.log"
        ) as log_mock:
            interceptor.intercept_unary_stream(
                continuation=lambda _details, _request: mock_call,
                client_call_details=details,
                request=GetNodesRequest(run_id=1),
            )

            done_callback = mock_call.add_done_callback.call_args.args[0]
            done_callback(mock_call)

        log_mock.assert_called_once()
