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
from collections.abc import Iterable, Iterator
from typing import cast
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


def _make_call_details(
    method: str,
    metadata: tuple[tuple[str, str | bytes], ...] = (),
) -> _ClientCallDetails:
    return _ClientCallDetails(
        method=method,
        timeout=None,
        metadata=metadata,
        credentials=None,
        wait_for_ready=None,
        compression=None,
    )


def _make_runtime_metadata(version: str) -> tuple[tuple[str, str], ...]:
    return (
        (FLWR_PACKAGE_NAME_METADATA_KEY, "flwr"),
        (FLWR_PACKAGE_VERSION_METADATA_KEY, version),
        (FLWR_COMPONENT_NAME_METADATA_KEY, "simulation"),
    )


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


def _make_stream_call(
    messages: tuple[str, ...] = ("msg1", "msg2"),
    *,
    trailing_metadata: tuple[tuple[str, str | bytes], ...] = (),
) -> Mock:
    call = Mock()
    call.trailing_metadata.return_value = trailing_metadata
    call.__iter__ = Mock(return_value=iter(messages))
    return call


class TestRuntimeVersionClientInterceptor(TestCase):
    """Unit tests for RuntimeVersionClientInterceptor."""

    def setUp(self) -> None:
        """Create the default client interceptor under test."""
        self.interceptor = RuntimeVersionClientInterceptor(component_name="simulation")

    def _make_call(self) -> Mock:
        call = Mock(spec=grpc.Call)
        call.trailing_metadata.return_value = ()
        return call

    def test_attach_runtime_version_headers(self) -> None:
        """The interceptor should add the shared version metadata keys."""
        details = _make_call_details(
            "/flwr.proto.ServerAppIo/GetNodes",
            (("x-test", "value"),),
        )
        captured: dict[str, list[tuple[str, str | bytes]]] = {}
        call = self._make_call()

        def continuation(
            client_call_details: grpc.ClientCallDetails,
            _request: GrpcMessage,
        ) -> Mock:
            captured["metadata"] = list(client_call_details.metadata or [])
            return call

        response = self.interceptor.intercept_unary_unary(
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
        details = _make_call_details(
            "/flwr.proto.ServerAppIo/GetNodes",
            ((FLWR_PACKAGE_NAME_METADATA_KEY, "old"), ("x-test", "value")),
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "gRPC metadata already contains runtime version keys: flwr-package-name",
        ):
            self.interceptor.intercept_unary_unary(
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

    def _intercept(
        self,
        method: str,
        metadata: tuple[tuple[str, str | bytes], ...],
        *,
        stream: bool = False,
    ) -> grpc.RpcMethodHandler:
        handler = _make_unary_stream_handler() if stream else _make_unary_handler()
        return self.interceptor.intercept_service(
            lambda _: handler,
            _HandlerCallDetails(method, metadata),
        )

    def test_missing_metadata_is_tolerated(self) -> None:
        """Missing runtime metadata should pass during rollout."""
        intercepted = self._intercept(
            "/flwr.proto.ServerAppIo/GetNodes",
            (),
        )

        context = Mock()
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        self.assertEqual(response, "ok")
        context.set_trailing_metadata.assert_not_called()

    def test_unparseable_peer_version_is_warned(self) -> None:
        """Explicit unparseable peer versions should be warned."""
        intercepted = self._intercept(
            "/flwr.proto.ServerAppIo/GetNodes",
            _make_runtime_metadata("main"),
        )

        context = Mock()
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        self.assertEqual(response, "ok")
        context.set_trailing_metadata.assert_called_once()

    def test_incompatible_metadata_is_warned(self) -> None:
        """Different major.minor versions should still be warned."""
        intercepted = self._intercept(
            "/flwr.proto.ServerAppIo/GetNodes",
            _make_runtime_metadata("1.30.1"),
        )

        context = Mock()
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        self.assertEqual(response, "ok")
        context.set_trailing_metadata.assert_called_once()

    def test_compatible_metadata_is_accepted(self) -> None:
        """Compatible peer version should not set trailing metadata for unary
        handlers."""
        intercepted = self._intercept(
            "/flwr.proto.ServerAppIo/GetNodes",
            _make_runtime_metadata("1.29.7"),
        )

        context = Mock()
        response = intercepted.unary_unary(GetNodesRequest(run_id=1), context)
        self.assertEqual(response, "ok")
        context.set_trailing_metadata.assert_not_called()

    def test_unary_stream_incompatible_metadata_is_warned(self) -> None:
        """Incompatible peer version should set trailing metadata for stream
        handlers."""
        intercepted = self._intercept(
            "/flwr.proto.ServerAppIo/PullTaskIns",
            _make_runtime_metadata("1.30.1"),
            stream=True,
        )

        context = Mock()
        responses = list(intercepted.unary_stream(GetNodesRequest(run_id=1), context))
        self.assertEqual(responses, ["a", "b"])
        context.set_trailing_metadata.assert_called_once()

    def test_unary_stream_compatible_metadata_is_accepted(self) -> None:
        """Compatible peer version should not set trailing metadata for stream
        handlers."""
        intercepted = self._intercept(
            "/flwr.proto.ServerAppIo/PullTaskIns",
            _make_runtime_metadata("1.29.7"),
            stream=True,
        )

        context = Mock()
        responses = list(intercepted.unary_stream(GetNodesRequest(run_id=1), context))
        self.assertEqual(responses, ["a", "b"])
        context.set_trailing_metadata.assert_not_called()


class TestRuntimeVersionClientInterceptorUnaryStream(TestCase):
    """Unit tests for RuntimeVersionClientInterceptor.intercept_unary_stream."""

    def setUp(self) -> None:
        """Create the default unary-stream client interceptor under test."""
        self.interceptor = RuntimeVersionClientInterceptor(component_name="simulation")

    def _intercept(
        self,
        call: Iterable[str],
        metadata: tuple[tuple[str, str | bytes], ...] = (),
    ) -> Iterable[str]:
        return cast(
            Iterable[str],
            self.interceptor.intercept_unary_stream(
                continuation=lambda _details, _request: call,
                client_call_details=_make_call_details(
                    "/flwr.proto.Fleet/PullTaskIns",
                    metadata,
                ),
                request=GetNodesRequest(run_id=1),
            ),
        )

    def test_attach_runtime_version_headers_unary_stream(self) -> None:
        """The interceptor should add version metadata headers for stream calls."""
        details = _make_call_details(
            "/flwr.proto.Fleet/PullTaskIns",
            (("x-test", "value"),),
        )
        captured: dict[str, list[tuple[str, str | bytes]]] = {}

        def continuation(
            client_call_details: grpc.ClientCallDetails,
            _request: GrpcMessage,
        ) -> Mock:
            captured["metadata"] = list(client_call_details.metadata or [])
            return _make_stream_call()

        responses = list(
            self.interceptor.intercept_unary_stream(
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

    def test_log_incompatibility_from_trailing_metadata(self) -> None:
        """The interceptor should log stream incompatibilities from trailing
        metadata."""
        mock_call = _make_stream_call(
            trailing_metadata=(
                (VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY, "runtime mismatch"),
            ),
        )

        with patch(
            "flwr.supercore.interceptors.runtime_version_interceptor.log"
        ) as log_mock:
            self.interceptor.intercept_unary_stream(
                continuation=lambda _details, _request: mock_call,
                client_call_details=_make_call_details("/flwr.proto.Fleet/PullTaskIns"),
                request=GetNodesRequest(run_id=1),
            )

            done_callback = mock_call.add_callback.call_args.args[0]
            done_callback()

        mock_call.add_callback.assert_called_once()
        log_mock.assert_called_once()

    def test_log_incompatibility_when_callback_cannot_be_added(self) -> None:
        """The interceptor should log immediately if the RPC already terminated."""
        mock_call = _make_stream_call(
            trailing_metadata=(
                (VERSION_INCOMPATIBILITY_MESSAGE_METADATA_KEY, "runtime mismatch"),
            ),
        )
        mock_call.add_callback.return_value = False

        with patch(
            "flwr.supercore.interceptors.runtime_version_interceptor.log"
        ) as log_mock:
            response = self.interceptor.intercept_unary_stream(
                continuation=lambda _details, _request: mock_call,
                client_call_details=_make_call_details("/flwr.proto.Fleet/PullTaskIns"),
                request=GetNodesRequest(run_id=1),
            )

        self.assertIs(response, mock_call)
        mock_call.add_callback.assert_called_once()
        log_mock.assert_called_once()
