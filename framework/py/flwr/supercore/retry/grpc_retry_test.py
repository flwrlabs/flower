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
"""Tests for gRPC retry utilities."""


from unittest.mock import Mock, patch

import grpc
import pytest

from .grpc_retry import make_simple_grpc_retry_invoker


class _UnauthenticatedError(grpc.RpcError):  # type: ignore[misc]
    """gRPC error reporting an authentication failure."""

    def code(self) -> grpc.StatusCode:
        """Return the gRPC status code."""
        return grpc.StatusCode.UNAUTHENTICATED


@patch("flwr.supercore.retry.grpc_retry.os.kill")
def test_unauthenticated_does_not_signal_when_retries_disabled(
    mock_kill: Mock,
) -> None:
    """Late background RPC failures must not interrupt graceful shutdown."""
    retry_invoker = make_simple_grpc_retry_invoker()
    retry_invoker.max_tries = 1

    with pytest.raises(_UnauthenticatedError):
        retry_invoker.invoke(Mock(side_effect=_UnauthenticatedError()))

    mock_kill.assert_not_called()
