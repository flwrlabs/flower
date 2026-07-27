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
"""Token auth policy definitions for AppIo interfaces."""


from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodTokenPolicy:
    """Token requirement for a single unary RPC method."""

    requires_token: bool

    @staticmethod
    def no_auth() -> MethodTokenPolicy:
        """Return policy for methods that should remain unauthenticated."""
        return MethodTokenPolicy(requires_token=False)

    @staticmethod
    def token_required() -> MethodTokenPolicy:
        """Return policy for methods protected by token auth."""
        return MethodTokenPolicy(requires_token=True)


_RUNTIME_METHOD_AUTH_POLICY: dict[str, MethodTokenPolicy] = {
    "PullPendingTasks": MethodTokenPolicy.no_auth(),
    "ClaimTask": MethodTokenPolicy.no_auth(),
    "GetRun": MethodTokenPolicy.no_auth(),
    "SendTaskHeartbeat": MethodTokenPolicy.token_required(),
    "PullTaskInput": MethodTokenPolicy.token_required(),
    "PushTaskOutput": MethodTokenPolicy.token_required(),
    "PushObject": MethodTokenPolicy.token_required(),
    "PullObject": MethodTokenPolicy.token_required(),
    "ConfirmMessageReceived": MethodTokenPolicy.token_required(),
    "PushLogs": MethodTokenPolicy.token_required(),
    "PushMessages": MethodTokenPolicy.token_required(),
    "PullMessages": MethodTokenPolicy.token_required(),
    "GetNodes": MethodTokenPolicy.token_required(),
    "CreateTask": MethodTokenPolicy.token_required(),
    "PushTaskMessage": MethodTokenPolicy.token_required(),
    "PushTaskEvents": MethodTokenPolicy.token_required(),
    "PullTaskMessage": MethodTokenPolicy.token_required(),
    "RecordTaskUsage": MethodTokenPolicy.token_required(),
}


def _build_runtime_method_auth_policy(
    service_name: str,
) -> dict[str, MethodTokenPolicy]:
    """Build the token policy map for an AppIo service."""
    return {
        f"/flwr.proto.{service_name}/{method}": policy
        for method, policy in _RUNTIME_METHOD_AUTH_POLICY.items()
    }


SERVERAPPIO_METHOD_AUTH_POLICY = _build_runtime_method_auth_policy("ServerAppIo")
CLIENTAPPIO_METHOD_AUTH_POLICY = _build_runtime_method_auth_policy("ClientAppIo")
