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
"""Tests for the SuperNode SuperExec authentication dependency."""

from unittest.mock import Mock, patch

from fastapi import Request

from flwr.supernode.nodestate import NodeState

from .superexec import SuperExecAuthDependency


def test_superexec_auth_dependency_delegates_to_shared_authentication() -> None:
    """Authenticate a SuperExec request using the supplied SuperNode state."""
    method = "/flwr.proto.Runtime/PullPendingTasks"
    request = Mock(spec=Request)
    state = Mock(spec=NodeState)
    dependency = SuperExecAuthDependency(method)

    with patch(
        "flwr.supernode.dependencies.superexec.authenticate_superexec_request"
    ) as authenticate:
        dependency(request, state)

    authenticate.assert_called_once_with(request, state, method)
